from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torch


class MMLUDataset(Dataset):
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




class MMLU:
    def __init__(self, tokenizer,max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def process_func(self, example):
        def create_mmlu_prompt(context, choices):
            """
            构建成这种形式的格式 <|start_header_id|>user<|end_header_id|>\n\n{example['instruction_zh'] + example['input_zh']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """
            prompt = """
            You are an expert in the field of text classification. Please choose the most appropriate option from [A, B, C, D] based on the given context and output only one option. \nQuestion: {}\n
            """
            indexs = ["A", "B", "C", "D"]
            user_prompt = f"{context}\n" + "\n".join(
                [f"{index}. {choice}" for index, choice in zip(indexs, choices)])
            prompt = prompt.format(user_prompt)

            return prompt
        MAX_LENGTH = self.max_length
        context = example["question"]
        choices = example["choices"]
        label = int(example["answer"])
        indexs = ["A", "B", "C", "D"]
        prompt = create_mmlu_prompt(context, choices)
        instruction = self.tokenizer(prompt, add_special_tokens=False)
        response = self.tokenizer(f"Answer: {indexs[label]}", add_special_tokens=False)

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
            raise Exception("MMLU must have a split: train/validation")
        dataset = load_dataset("cais/mmlu", "all")[split]
        tokenized_dataset = dataset.map(lambda x: self.process_func(x), batched=False)
        if tensor:
            return MMLUDataset(tokenized_dataset)
        return tokenized_dataset
