import argparse
import json
import logging

import torch
from datasets import load_dataset, Dataset
from modelscope import snapshot_download
from swanlab.integration.huggingface import SwanLabCallback
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, \
    GenerationConfig, TrainerCallback
import os
from peft import LoraConfig, TaskType, get_peft_model

# 设置日志文件
logging.basicConfig(filename='training_log.txt', level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置wandb为离线模式
# wandb.init(mode="offline", project="llama_sft")

MAX_LENGTH = 256

# dataset = load_dataset("tatsu-lab/alpaca")
dataset = load_dataset("cais/mmlu","all")
model_path = "/root/autodl-tmp/models/qwen/Qwen2-0___5B"
output_dir_prefix = "./output/qwen15-alpaca"
print("start loading")
#all_dataset = dataset['train'].select(range(5000))
all_dataset = dataset["validation"]
# all_dataset = dataset['train']
columns_to_remove = ['output', 'input', 'instruction']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_path,
                                          use_fast=False, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token


def load_mmlu_dataset(tokenizer, max_length=512, subset="high_school_physics", split="validation"):
    """
    加载并预处理 MMLU 数据集
    :param tokenizer: 用于处理文本的tokenizer
    :param max_length: 最大长度
    :param subset: 选择 MMLU 数据集的某个子集，如 'high_school_physics'
    :param split: 选择数据集的划分，'validation' 或 'test'
    :return: 经过 tokenization 的数据集
    """
    # 加载指定子集的数据
    dataset = load_dataset("cais/mmlu", subset, split=split)

    def tokenize_function(examples):
        # MMLU 是一个多选题任务，每个问题有多种选择
        texts = []
        for question, choices in zip(examples["question"], examples["choices"]):
            # 生成多选任务提示
            prompt = f"Question: {question}\nChoices: "
            for idx, choice in enumerate(choices):
                prompt += f"({chr(65 + idx)}) {choice}  "
            texts.append(prompt)
        return tokenizer(texts, padding="max_length", truncation=True, max_length=max_length)

    tokenized_dataset = dataset.map(tokenize_function, batched=True,
                                    remove_columns=[col for col in dataset.column_names if col not in ['answer']])

    return tokenized_dataset


class MMLUDataset(Dataset):
    def __init__(self, tokenized_dataset, labels):
        self.tokenized_dataset = tokenized_dataset
        self.labels = labels  # 标签列代表每个问题的正确答案

    def __len__(self):
        return len(self.tokenized_dataset)

    def __getitem__(self, idx):
        item = self.tokenized_dataset[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"]),
            "attention_mask": torch.tensor(item["attention_mask"]),
            "labels": torch.tensor(self.labels[idx])  # 将标签转换为 Tensor
        }



# def process_func(example):
#     input_ids, attention_mask, labels = [], [], []
#     instruction = tokenizer \
#         (f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. \n{example['instruction'] + example['input']}\n",
#          add_special_tokens=False)
#     response = tokenizer(f"{example['output']}", add_special_tokens=False)
#     input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
#     attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
#     labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
#
#     if len(input_ids) > MAX_LENGTH:
#         input_ids = input_ids[:MAX_LENGTH]
#         attention_mask = attention_mask[:MAX_LENGTH]
#         labels = labels[:MAX_LENGTH]
#     else:
#         padding_length = MAX_LENGTH - len(input_ids)
#         input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
#         attention_mask = attention_mask + [0] * padding_length
#         labels = labels + [-100] * padding_length
#
#     assert len(input_ids) == MAX_LENGTH, "input_ids length not equal to MAX_LENGTH"
#     assert len(attention_mask) == MAX_LENGTH, "input_ids length not equal to MAX_LENGTH"
#     assert len(labels) == MAX_LENGTH, "input_ids length not equal to MAX_LENGTH"
#
#     return {
#         "input_ids": input_ids,
#         "attention_mask": attention_mask,
#         "labels": labels
#     }

def process_func(example):
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
    context = example["question"]
    choices = example["choices"]
    label = int(example["answer"])
    indexs = ["A", "B", "C", "D"]
    prompt = create_mmlu_prompt(context, choices)
    instruction = tokenizer(prompt, add_special_tokens=False)
    response = tokenizer(f"Answer: {indexs[label]}", add_special_tokens=False)

    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    else:
        padding_length = MAX_LENGTH - len(input_ids)
        input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
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


def train(lora_num):
    # model_path = "/root/autodl-tmp/models/qwen/Qwen2-1___5B"
    output_dir = f"{output_dir_prefix}/r_{lora_num}"
    model_save_path = f"{output_dir_prefix}/final_model_r_{lora_num}"

    # 预处理数据集
    #tokenized_dataset = all_dataset.map(process_func, batched=False, remove_columns=columns_to_remove)
    tokenized_dataset = all_dataset.map(process_func, batched=False)
    train_dataset = tokenized_dataset
    # train_dataset = load_mmlu_dataset(tokenizer, subset="all").select(range(1000))
    # train_dataset = MMLUDataset(train_dataset, train_dataset['answer'])
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    model.enable_input_require_grads()
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj"],
        inference_mode=False,
        r=lora_num,
        lora_alpha=1
        # lora_alpha=2*lora_num,
        # lora_dropout=0.1
    )

    model = get_peft_model(model, config)

    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        logging_steps=3,
        num_train_epochs=2,
        save_steps=800,
        learning_rate=1e-4,
        # weight_decay=0.01,  # 默认参数
        # warmup_steps=int(0.33 * (len(tokenized_dataset) // (16 * 4))),
        # save_on_each_node=True,
        # gradient_checkpointing=True,
        # report_to="wandb",
        report_to="none",
    )

    class CustomLoggingCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                try:
                    # 提取需要的日志信息
                    log_info = {key: logs[key] for key in ['loss', 'grad_norm', 'learning_rate', 'epoch'] if
                                key in logs}
                    # 记录到日志文件
                    logger.info(log_info)
                    # 保存到 JSON Lines 文件
                    with open(f'training_logs_r_{lora_num}.jsonl', 'a') as f:
                        json.dump(log_info, f)
                        f.write('\n')
                except Exception as e:
                    pass

    # swanlab_callback = SwanLabCallback(
    #     project="Qwen2-mmlu-fintune",
    #     experiment_name=f"Qwen2-1.5B-Instruct-lora-alpaca-{lora_num}",
    #     description="使用通义千问Qwen2-1.5B-Instruct模型在zh_cls_fudan-news数据集上微调。",
    #     config={
    #         "model": f"qwen/Qwen2-1.5B-Instruct-lora-{lora_num}",
    #         "dataset": "mmlu",
    #     }
    # )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        # eval_dataset=test_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        callbacks=[CustomLoggingCallback]
        # compute_metrics=compute_metrics
    )
    try:
        trainer.train(resume_from_checkpoint=True)
    except Exception as e:
        trainer.train()

    model_save_path = model_save_path
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)


def main():
    parser = argparse.ArgumentParser(description='Run LoRA training')
    parser.add_argument('--lora_num', type=int, required=True, help='LoRA parameter value')
    args = parser.parse_args()

    lora_num = args.lora_num
    print(f"Starting training with r={lora_num}")
    train(lora_num)
    print(f"Finished training with r={lora_num}")


if __name__ == '__main__':
    loras = [32]
    for lora_num in loras:
        print(f"current processing r={lora_num}")
        train(lora_num)

    # main()
