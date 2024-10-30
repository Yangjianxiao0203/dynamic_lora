from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torch


class HellaSwagDataset(Dataset):
    def __init__(self, tokenized_dataset):
        self.tokenized_dataset = tokenized_dataset

    def __len__(self):
        return len(self.tokenized_dataset)

    def __getitem__(self, idx):
        item = self.tokenized_dataset[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"]),
            "attention_mask": torch.tensor(item["attention_mask"]),
            "labels": torch.tensor(item["labels"])
        }


class HellaSwag:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def process_func(self, example):
        """
        Formats each HellaSwag example into a prompt with choices and the correct answer label.
        """
        def create_hellaswag_prompt(context, choices):
            prompt = """
            Given the context, choose the most plausible continuation among [A, B, C, D].\n\nContext: {}\n
            """
            index_labels = ["A", "B", "C", "D"]
            choice_text = "\n".join([f"{index}. {choice}" for index, choice in zip(index_labels, choices)])
            full_prompt = prompt.format(context) + choice_text
            return full_prompt

        MAX_LENGTH = self.max_length
        context = example["ctx_a"] + " " + example["ctx_b"]
        choices = example["endings"]
        label = int(example["label"])
        index_labels = ["A", "B", "C", "D"]

        # Create prompt and tokenize
        prompt = create_hellaswag_prompt(context, choices)
        instruction = self.tokenizer(prompt, add_special_tokens=False)
        response = self.tokenizer(f"Answer: {index_labels[label]}", add_special_tokens=False)

        # Concatenate input and response tokens
        input_ids = instruction["input_ids"] + response["input_ids"] + [self.tokenizer.pad_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [self.tokenizer.pad_token_id]

        # Ensure correct length with padding or truncation
        if len(input_ids) > MAX_LENGTH:
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]
        else:
            padding_length = MAX_LENGTH - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * padding_length
            attention_mask += [0] * padding_length
            labels += [-100] * padding_length

        assert len(input_ids) == MAX_LENGTH, "input_ids length not equal to MAX_LENGTH"
        assert len(attention_mask) == MAX_LENGTH, "attention_mask length not equal to MAX_LENGTH"
        assert len(labels) == MAX_LENGTH, "labels length not equal to MAX_LENGTH"

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def get_dataset(self, split=None, tensor=True):
        if not split:
            raise Exception("HellaSwag must have a split: train/validation/test")
        dataset = load_dataset("AlekseyKorshuk/hellaswag")[split]
        tokenized_dataset = dataset.map(lambda x: self.process_func(x), batched=False)
        if tensor:
            return HellaSwagDataset(tokenized_dataset)
        return tokenized_dataset
