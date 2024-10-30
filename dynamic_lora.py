import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import math

from load_data.alpaca import Alpaca
from load_data.mmlu import MMLU


def snapshot_lora_ranks(model, step_num, epoch, loss, file_name="singular_values.jsonl"):
    with open(f'./records/{file_name}', 'a') as f:
        for name, module in model.named_modules():
            if isinstance(module, DynamicLoRALayer):
                combined = module.lora_B @ module.lora_A
                U, S, Vh = torch.linalg.svd(combined, full_matrices=False)
                # 记录唯一名字和前32个秩
                singular_values = S[:32].cpu().tolist()
                dic = {
                    "name": name,
                    "step_num": step_num,
                    "epoch": epoch,
                    "ranks": singular_values,
                    "loss": loss.item()
                }
                f.write(json.dumps(dic, ensure_ascii=False) + '\n')


class DynamicLoRALayer(nn.Module):
    def __init__(self, original_linear, r_max, name, alpha=32):
        super().__init__()
        self.original_linear = original_linear
        self.r_max = r_max
        # self.rank_threshold = 1e-5 * 3
        self.rank_threshold = 0

        # Freeze the original weights
        for param in self.original_linear.parameters():
            param.requires_grad = False

        in_features = original_linear.in_features
        out_features = original_linear.out_features
        self.lora_A = nn.Parameter(torch.zeros(r_max, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r_max))
        self.scaling = alpha / r_max
        self.name = name

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

    def reduce_rank_with_svd(self, verbose=True, scale=1 / 3):
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

            # 如果S都是0，退出，不更新
            if torch.all(S == 0):
                if verbose:
                    print("All singular values are zero. Exiting without updating.")
                return  # Exit the function without updating
            if verbose:
                print(f"u shape: {U.shape}")
                print(f"s shape: {S.shape}")
                print(f"vh shape: {Vh.shape}")

            self.rank_threshold = 1 / 3 * self.rank_threshold
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


class DynamicLoRAManager:
    def __init__(self, model, r_max, keywords, alpha):
        self.model = model
        self.r_max = r_max
        self.alpha = alpha
        self.keywords = keywords
        self.dynamic_lora_layers = {}
        self.device = next(model.parameters()).device

    def replace_with_dynamic_lora(self):
        print(f"Replacing model layers with dynamic LoRA layers: r_max={self.r_max}")
        for name, module in self.model.named_modules():
            for param in module.parameters():
                param.requires_grad = False
            if isinstance(module, nn.Linear):
                if not any(keyword in name for keyword in self.keywords):
                    continue
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                print(f"Replacing {parent_name}.{child_name} with DynamicLoRALayer")
                parent = self.model.get_submodule(parent_name)
                dynamic_lora = DynamicLoRALayer(module, self.r_max, name,self.alpha)
                setattr(parent, child_name, dynamic_lora)
                self.dynamic_lora_layers[name] = dynamic_lora
        print(f"Total trainable parameters: {count_trainable_params(self.model)}")

    def save_lora_layers(self, save_path):
        lora_state_dict = {}
        for name, lora_layer in self.dynamic_lora_layers.items():
            lora_state_dict[name] = {
                "lora_A": lora_layer.lora_A.detach().cpu(),
                "lora_B": lora_layer.lora_B.detach().cpu(),
            }
        torch.save(lora_state_dict, save_path)
        print(f"LoRA layers saved to {save_path}")

    def load_lora_layers(self, load_path):
        lora_state_dict = torch.load(load_path)
        # 相当于置换，把manager中的lora layer的权重，lora_A, lora_B 替换进去
        for name, lora_layer in self.dynamic_lora_layers.items():
            if name in lora_state_dict:
                lora_layer.lora_A.data = lora_state_dict[name]["lora_A"].to(self.device)
                lora_layer.lora_B.data = lora_state_dict[name]["lora_B"].to(self.device)
        print(f"LoRA layers loaded from {load_path}")

    def update_lora_ranks(self, scale):
        total_rank = 0
        num_lora_layers = 0
        verbose = True
        for module in self.dynamic_lora_layers.values():
            module.reduce_rank_with_svd(verbose, scale=scale)
            combined = module.lora_B @ module.lora_A
            singular_values = torch.linalg.svdvals(combined)
            current_rank = torch.sum(singular_values > 1e-5).item()
            total_rank += current_rank
            num_lora_layers += 1
            verbose = False
        avg_rank = total_rank / num_lora_layers if num_lora_layers > 0 else 0
        return avg_rank


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model, train_loader, optimizer, scheduler, device, epoch, dynamic_lora_manager=None, **kwargs):
    snapshot_path = kwargs.get("snapshot_path", None)
    model.train()
    total_loss = 0
    for index, batch in enumerate(train_loader):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        print(
            f"epoch {epoch}, batch {index}, loss: {loss.item() / input_ids.shape[0]}, batch size: {input_ids.shape[0]}")

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        if snapshot_path:
            snapshot_lora_ranks(model, index, epoch, loss, file_name=snapshot_path)

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


def main(model_path, keywords, snapshot_path=None, save_lora_path="lora_state.pth", load_lora_path=None):
    r_max = 32
    batch_size = 8
    learning_rate = 1e-4
    num_epochs = 2
    alpha = 64
    scale = 1 / 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    print(f"Loading model {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    dynamic_lora_manager = DynamicLoRAManager(model, r_max, keywords, alpha)
    dynamic_lora_manager.replace_with_dynamic_lora()
    model.to(device)

    if load_lora_path:
        dynamic_lora_manager.load_lora_layers(load_lora_path)

    train_dataset = MMLU(tokenizer).get_dataset(split="validation", tensor=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # train_dataset = Alpaca(tokenizer).get_dataset(split="train", tensor=True)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * num_epochs)

    # Training loop
    pre_loss = None
    for epoch in range(num_epochs):
        avg_rank = dynamic_lora_manager.update_lora_ranks(scale)
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss = train(model, train_loader, optimizer, scheduler, device, epoch, dynamic_lora_manager,
                           snapshot_path=snapshot_path)
        if pre_loss:
            scale = pre_loss / train_loss
        pre_loss = train_loss
        val_loss = evaluate(model, train_loader, device)
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Avg LoRA Rank: {avg_rank:.2f}")

    # Save the final LoRA layers
    print(f"saving LoRA layers to {save_lora_path}")
    dynamic_lora_manager.save_lora_layers(save_lora_path)


if __name__ == "__main__":
    model_path = "/root/autodl-tmp/models/qwen/Qwen2-0___5B"
    # run_times = 5
    # for i in range(run_times):
    #     snapshot_path = f"qwen-1_5-exp-{i + 1}.jsonl"
    #     main(model_path, snapshot_path)
    # model_path = "bert-base-uncased"
    keywords = ["q_proj", "k_proj", "v_proj"]
    # lora_path = "qwen_lora_state.pth"
    main(model_path, keywords)
