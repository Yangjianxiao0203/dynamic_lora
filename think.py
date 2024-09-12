import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import math
import swanlab


# Set environment variables and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "/root/autodl-tmp/models/qwen/Qwen2-0___5B"
save_model = "dynamic_qwen_0.5_alpaca"


class DynamicLoRALayer(nn.Module):
    def __init__(self, original_linear, r_max):
        super().__init__()
        self.original_linear = original_linear
        self.r_max = r_max
        self.rank_threshold = 1e-5 * 3

        # Freeze the original weights
        for param in self.original_linear.parameters():
            param.requires_grad = False

        in_features = original_linear.in_features
        out_features = original_linear.out_features
        self.lora_A = nn.Parameter(torch.zeros(r_max, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r_max))
        self.scaling = 1 / r_max

        # Initialize LoRA matrices
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def get_sparse_weights(self):
        # No sparsification, just return the current weights
        return self.lora_A, self.lora_B

    def forward(self, x):
        # Forward pass through the original layer
        original_output = self.original_linear(x)

        sparse_A, sparse_B = self.get_sparse_weights()
        lora_intermediate = torch.einsum('bsi,ri->brs', x, sparse_A)  # (batch_size, r_max, seq_len)
        lora_output = torch.einsum('brs,or->bos', lora_intermediate, sparse_B)  # (batch_size, out_features, seq_len)
        lora_output = lora_output.permute(0, 2, 1)  # (batch_size, seq_len, out_features)

        return original_output + lora_output * self.scaling

    def reduce_rank_with_svd(self, verbose=True, scale=1/3):
        with torch.no_grad():
            # Compute the combined matrix from lora_B and lora_A
            combined = self.lora_B @ self.lora_A
            if verbose:
                print(f"lora a : {self.lora_A.shape}")
                print(f"lora b : {self.lora_B.shape}")
                print(f"weight: {combined.shape}")
                print(f"lora_a: {self.lora_A}")
                print(f"lora_b: {self.lora_B}")
                print(f"lora_a grad: {self.lora_A.grad}")
                print(f"lora_b grad: {self.lora_B.grad}")
            # Perform Singular Value Decomposition (SVD)
            U, S, Vh = torch.linalg.svd(combined, full_matrices=False)

            #如果S都是0，退出，不更新
            if torch.all(S == 0):
                if verbose:
                    print("All singular values are zero. Exiting without updating.")
                return  # Exit the function without updating
            if verbose:
                print(f"u shape: {U.shape}")
                print(f"s shape: {S.shape}")
                print(f"vh shape: {Vh.shape}")

            self.rank_threshold = 1/3 * self.rank_threshold
            num_greater_than_threshold = torch.sum(S > self.rank_threshold).item()
            if verbose:
                print(f"S: {S[:16]}")
                print(f"next rank: {num_greater_than_threshold}")
                print(f"rank threshole: {self.rank_threshold}")
                print(f"scale: {scale}")

            r_actual = min(self.lora_A.size(0), num_greater_than_threshold)
            if verbose:
                print(f"r actual: {r_actual}")
            reduced_A = Vh[:r_actual, :].T @ torch.diag(S[:r_actual])
            reduced_B = U[:, :r_actual] @ torch.diag(S[:r_actual])

            self.lora_A[:r_actual, :] = reduced_A.T
            self.lora_B[:, :r_actual] = reduced_B


def replace_with_dynamic_lora(model, r_max, keywords=[]):
    print(f"Replacing model layers with dynamic LoRA layers: r_max={r_max}")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if not any(keyword in name for keyword in keywords):
                continue
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            print(f"Replacing {parent_name}.{child_name} with DynamicLoRALayer")
            parent = model.get_submodule(parent_name)
            dynamic_lora = DynamicLoRALayer(module, r_max)
            setattr(parent, child_name, dynamic_lora)
    return model


def load_alpaca_dataset(tokenizer, max_length=512):
    dataset = load_dataset("tatsu-lab/alpaca")

    def tokenize_function(examples):
        prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
        texts = []
        for instruction, input_text, output_text in zip(examples["instruction"], examples["input"], examples["output"]):
            text = prompt.format(instruction=instruction, input=input_text, output=output_text)
            texts.append(text)
        return tokenizer(texts, padding="max_length", truncation=True, max_length=max_length)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)
    return tokenized_dataset["train"]


class AlpacaDataset(Dataset):
    def __init__(self, tokenized_dataset):
        self.tokenized_dataset = tokenized_dataset

    def __len__(self):
        return len(self.tokenized_dataset)

    def __getitem__(self, idx):
        item = self.tokenized_dataset[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"]),
            "attention_mask": torch.tensor(item["attention_mask"]),
            "labels": torch.tensor(item["input_ids"])
        }


def train(model, train_loader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    for index, batch in enumerate(train_loader):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        print(f"epoch {epoch}, batch {index}, loss: {loss.item()}")

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
    return total_loss / len(val_loader)


def update_lora_ranks(model,scale):
    total_rank = 0
    num_lora_layers = 0
    verbose= True
    for module in model.modules():
        if isinstance(module, DynamicLoRALayer):
            module.reduce_rank_with_svd(verbose,scale=scale)
            combined = module.lora_B @ module.lora_A
            singular_values = torch.linalg.svdvals(combined)
            current_rank = torch.sum(singular_values > 1e-5).item()
            total_rank += current_rank
            num_lora_layers += 1
            verbose= False

    avg_rank = total_rank / num_lora_layers if num_lora_layers > 0 else 0
    return avg_rank


def main():
    # swanlab.init(project="qwen-0.5B", experiment="experiment-1")

    r_max = 16
    batch_size = 8
    learning_rate = 1e-4
    num_epochs = 6

    # Load model and tokenizer
    print(f"Loading model {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Replace linear layers with dynamic LoRA layers
    keywords = ["q_proj", "k_proj", "v_proj"]
    model = replace_with_dynamic_lora(model, r_max, keywords=keywords)
    model.to(device)

    # Load and prepare dataset
    dataset = load_alpaca_dataset(tokenizer)
    dataset = dataset.select(range(100))
    train_dataset = AlpacaDataset(dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * num_epochs)

    # Training loop
    pre_loss = None
    scale = 1/3
    for epoch in range(num_epochs):
        avg_rank = update_lora_ranks(model,scale)
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss = train(model, train_loader, optimizer, scheduler, device, epoch)
        if pre_loss:
            scale = pre_loss / train_loss
        pre_loss = train_loss
        val_loss = evaluate(model, train_loader, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Avg LoRA Rank: {avg_rank:.2f}")

    # Save the final model
    model.save_pretrained(save_model)
    tokenizer.save_pretrained(save_model)


if __name__ == "__main__":
    main()
