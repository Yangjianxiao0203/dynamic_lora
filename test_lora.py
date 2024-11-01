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

from load_data.hellaswag import HellaSwag

# 设置日志文件
logging.basicConfig(filename='training_log.txt', level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置wandb为离线模式
# wandb.init(mode="offline", project="llama_sft")

MAX_LENGTH = 512
model_path = "/root/autodl-tmp/models/qwen/Qwen2-0___5B"
output_dir_prefix = "./output/qwen15-alpaca"

tokenizer = AutoTokenizer.from_pretrained(model_path,
                                          use_fast=False, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token




def train(lora_num):
    # model_path = "/root/autodl-tmp/models/qwen/Qwen2-1___5B"
    output_dir = f"{output_dir_prefix}/r_{lora_num}"
    model_save_path = f"{output_dir_prefix}/final_model_r_{lora_num}"

    train_dataset = HellaSwag(tokenizer).get_dataset(split="validation", tensor=True)

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    model.enable_input_require_grads()
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj"],
        inference_mode=False,
        r=lora_num,
        lora_alpha=64
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
