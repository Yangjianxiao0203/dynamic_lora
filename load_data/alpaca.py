from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torch


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




class Alpaca:
    def __init__(self, tokenizer,max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def process_func(self,example):
        MAX_LENGTH = self.max_length
        input_ids, attention_mask, labels = [], [], []
        instruction = self.tokenizer \
            (f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. \n{example['instruction'] + example['input']}\n",
             add_special_tokens=False)
        response = self.tokenizer(f"{example['output']}", add_special_tokens=False)
        input_ids = instruction["input_ids"] + response["input_ids"] + [self.tokenizer.pad_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [self.tokenizer.pad_token_id]

        if len(input_ids) > MAX_LENGTH:
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]
        else:
            padding_length = MAX_LENGTH - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            labels = labels + [-100] * padding_length

        assert len(input_ids) == MAX_LENGTH, "input_ids length not equal to MAX_LENGTH"
        assert len(attention_mask) == MAX_LENGTH, "input_ids length not equal to MAX_LENGTH"
        assert len(labels) == MAX_LENGTH, "input_ids length not equal to MAX_LENGTH"

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def get_dataset(self,split=None, tensor=True):
        if not split:
            raise Exception("Alpaca must have a split: train/validation")
        dataset = load_dataset("tatsu-lab/alpaca")[split]
        tokenized_dataset = dataset.map(lambda x: self.process_func(x), batched=False)
        if tensor:
            return AlpacaDataset(tokenized_dataset)
        return tokenized_dataset
