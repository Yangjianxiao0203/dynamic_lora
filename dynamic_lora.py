import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import math


class DynamicLoRALayer(nn.Module):
    def __init__(self, original_linear, r_max, name):
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
        self.scaling = 1 / r_max
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

class LoraWrapper:
    def __init__(self):
        pass