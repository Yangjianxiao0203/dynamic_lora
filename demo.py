import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import LlamaForCausalLM, LlamaTokenizer,AutoTokenizer,AutoModelForCausalLM
from datasets import load_dataset
import math
import os

# Set environment variables and device
# os.environ['WANDB_PROJECT'] = "dynamirank-llama2-alpaca"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_path = "bert-base-uncased"
model_path = "/root/autodl-tmp/models/qwen/Qwen2-0___5B"
save_model = "dynamic_qwen_0.5_alpaca"


class DynamicLoRALayer(nn.Module):
    def __init__(self, original_linear, r_max):
        super().__init__()
        self.original_linear = original_linear

        # 冻结原始线性层的权重和偏置
        for param in self.original_linear.parameters():
            param.requires_grad = False

        # LoRA 低秩矩阵
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        self.lora_A = nn.Parameter(torch.zeros(r_max, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r_max))
        self.scaling = 1 / r_max
        self.sparsity_threshold = nn.Parameter(torch.tensor(0.01))

        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def get_sparse_weights(self):
        # Apply soft thresholding
        sparse_A = F.softshrink(self.lora_A, self.sparsity_threshold.item())
        sparse_B = F.softshrink(self.lora_B, self.sparsity_threshold.item())
        return sparse_A, sparse_B

    def forward(self, x):
        original_output = self.original_linear(x)

        sparse_A, sparse_B = self.get_sparse_weights()
        lora_intermediate = torch.einsum('bsi,ri->brs', x, sparse_A)  # (batch_size, r_max, seq_len)
        lora_output = torch.einsum('brs,or->bos', lora_intermediate, sparse_B)  # (batch_size, out_features, seq_len)

        lora_output = lora_output.permute(0, 2, 1)  # (batch_size, seq_len, out_features)
        # print(f"lora_output shape: {lora_output.shape}")
        return original_output + lora_output * self.scaling

    def estimate_rank(self):
        sparse_A, sparse_B = self.get_sparse_weights()
        combined = sparse_B @ sparse_A
        singular_values = torch.linalg.svdvals(combined)
        return torch.sum(singular_values > 1e-5).item()


def replace_with_dynamic_lora(model, r_max):
    print(f"replace model with dynamic lora layers: r {r_max}")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            print(f"replace {parent_name}.{child_name} with DynamicLoRALayer")
            parent = model.get_submodule(parent_name)
            dynamic_lora = DynamicLoRALayer(module, r_max)
            setattr(parent, child_name, dynamic_lora)
    return model


def load_alpaca_dataset(tokenizer, max_length=512):
    dataset = load_dataset("tatsu-lab/alpaca")

    def tokenize_function(examples):
        prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
        # texts = [prompt.format(**ex) for ex in examples]
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
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Add L1 regularization for sparsity
        l1_reg = sum(p.abs().sum() for name, p in model.named_parameters() if "lora_" in name)
        loss += 0.01 * l1_reg  # Adjust the coefficient as needed

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


def update_lora_ranks(model):
    total_rank = 0
    num_lora_layers = 0
    for module in model.modules():
        if isinstance(module, DynamicLoRALayer):
            current_rank = module.estimate_rank()
            total_rank += current_rank
            num_lora_layers += 1

            # Adjust sparsity threshold based on current rank
            if current_rank > module.lora_A.size(0) * 0.8:
                module.sparsity_threshold.data *= 1.1
            elif current_rank < module.lora_A.size(0) * 0.2:
                module.sparsity_threshold.data *= 0.9

    avg_rank = total_rank / num_lora_layers if num_lora_layers > 0 else 0
    return avg_rank


def main():
    # Initialize wandb
    # wandb.init(project="dynamirank-llama2-alpaca", name="experiment-1")
    # Hyperparameters
    r_max = 16
    batch_size = 8
    learning_rate = 1e-4
    num_epochs = 3

    # Load model and tokenizer
    print(f"start loading model {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"model {model_path} loaded")
    # Replace linear layers with dynamic LoRA layers
    model = replace_with_dynamic_lora(model, r_max)
    model.to(device)

    # Load and prepare dataset
    dataset = load_alpaca_dataset(tokenizer)
    train_dataset = AlpacaDataset(dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print("start loading optimizer and scheduler")
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * num_epochs)
    print("optimizer and scheduler loaded")
    print("training started")
    print("*" * 20)
    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss = train(model, train_loader, optimizer, scheduler, device, epoch)
        print(f"train_loss: {train_loss}")
        val_loss = evaluate(model, train_loader, device)  # Using train_loader as val_loader for simplicity
        print(f"val_loss: {val_loss}")
        avg_rank = update_lora_ranks(model)
        print(f"avg_rank: {avg_rank}")

        current_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "avg_lora_rank": avg_rank
        }
        print(current_log)
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Avg LoRA Rank: {avg_rank:.2f}")

    # Save the final model
    model.save_pretrained(save_model)
    tokenizer.save_pretrained(save_model)


if __name__ == "__main__":
    main()

