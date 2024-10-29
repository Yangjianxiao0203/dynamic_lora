from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from string import Template
from abc import ABC, abstractmethod


class LogLikelihoodEvaluator(ABC):
    def __init__(self, model, tokenizer, device='cuda', num_few_shot=0):
        # self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        print(f"evaluator device: {device}")
        self.prompt_template = None
        self.num_few_shot = num_few_shot
        self.few_shot_examples = []

    def load_dataset(self, dataset_name, dataset_split='validation', subset=None, **kwargs):
        # 通用的数据集加载方法，允许传入不同的数据集名称
        if subset:
            self.dataset = load_dataset(dataset_name, subset, split=dataset_split, **kwargs)
        else:
            self.dataset = load_dataset(dataset_name, split=dataset_split, **kwargs)
        print(f"Loaded dataset {dataset_name} with {len(self.dataset)} samples.")

    def load_few_shot_examples(self, few_shot_data):
        """加载 few-shot 示例"""
        self.few_shot_examples = few_shot_data

    def encode(self, context, continuation):
        # 将问题和选项编码为输入张量
        context_enc = self.tokenizer.encode(context, add_special_tokens=False)
        continuation_enc = self.tokenizer.encode(continuation, add_special_tokens=False)
        return context_enc, continuation_enc

    def calculate_loglikelihood(self, context, continuations):
        '''
        看每个选项，谁的logllikelihood加起来最大，选哪个
        '''
        loglikelihoods = []
        max_tokens = []
        for continuation in continuations:
            context_enc, continuation_enc = self.encode(context, continuation)

            # 拼接上下文和续文本，并截断到模型的最大长度
            input_ids = torch.tensor(
                (context_enc + continuation_enc)[-(self.model.config.max_position_embeddings + 1):][:-1],
                dtype=torch.long).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = torch.nn.functional.log_softmax(outputs.logits, dim=-1)

            # 只取续文本部分的logits
            continuation_len = len(continuation_enc)
            logits_for_continuation = logits[:, -continuation_len:, :]

            continuation_tensor = torch.tensor(continuation_enc, dtype=torch.long).unsqueeze(0).unsqueeze(-1).to(
                self.device)

            """
            torch.gather(input,dim,index)
            input = torch.tensor([[1, 2, 3, 4],
                                  [5, 6, 7, 8],
                                  [9, 10, 11, 12]])
            index = torch.tensor([[0, 1, 2, 0],
                                  [2, 0, 1, 3],
                                  [3, 1, 2, 0]])

            output = torch.gather(input, dim=1, index=index)

            output = tensor([[ 1,  2,  3,  1],
                            [ 7,  5,  6,  8],
                            [12, 10, 11,  9]])

            """
            # 按照 continuation_tensor 提供的索引，从每个 token 对应的词汇表中选择特定词汇的 logits（即 log 概率）。 dim=2 是词汇表的维度
            selected_logits = torch.gather(logits_for_continuation, 2, continuation_tensor).squeeze(-1)

            # 找到概率最大的 token
            max_logit_indices = torch.argmax(logits_for_continuation, dim=-1)
            max_tokens.append(max_logit_indices.squeeze().tolist())

            log_likelihood = selected_logits.sum().item()
            avg_log_likelihood = log_likelihood / continuation_len  # 计算平均 log-likelihood
            loglikelihoods.append(avg_log_likelihood)

        # decoded_max_token = self.tokenizer.decode(max_tokens, skip_special_tokens=True)
        # print(f"Predicted Token with Highest Probability: {decoded_max_token}")

        return loglikelihoods, max_tokens

    @abstractmethod
    def format_prompt(self, sample):
        if self.prompt_template:
            # prompt_template 例如: "Question: $question\nAnswer: $answer"
            context = self.prompt_template.substitute(question=sample['question'], answer='')
        else:
            context = sample['question']

        continuations = sample['choices']
        return context, continuations

    @abstractmethod
    def get_correct_answer(self, sample) -> List[str]:
        pass

    def evaluate_sample(self, sample):
        # 评估单个样本
        context, continuations = self.format_prompt(sample)
        correct_answer_idxs = self.get_correct_answer(sample)

        # 计算每个选项的log-likelihood
        loglikelihoods, max_tokens = self.calculate_loglikelihood(context, continuations)

        # 选择log-likelihood最大的选项
        predicted_idx = torch.argmax(torch.tensor(loglikelihoods)).item()
        correct = False
        if predicted_idx in correct_answer_idxs:
            correct = True
        # correct = predicted_idx == correct_answer_idx

        return correct, predicted_idx, correct_answer_idxs, loglikelihoods, max_tokens

    def evaluate(self, num_samples=100):
        # 评估多个样本的准确率
        correct_count = 0
        for i, sample in enumerate(self.dataset.select(range(num_samples))):
            correct, predicted_idx, correct_answer_idx, loglikelihoods, max_tokens = self.evaluate_sample(sample)
            correct_count += int(correct)
            decoded_max_token = self.tokenizer.decode(max_tokens[0], skip_special_tokens=True)
            print(
                f"[{i}]: Predicted {predicted_idx}, Correct {correct_answer_idx}, Log-likelihoods: {loglikelihoods}, highest prob: {decoded_max_token}")

        accuracy = correct_count / num_samples
        print(f"Accuracy: {accuracy:.2f}")
        return accuracy


class HellaSwagEvaluator(LogLikelihoodEvaluator):

    def format_prompt(self, sample):
        """格式化 HellaSwag 的问题和选项"""

        context = sample['ctx_a'] + " " + sample['ctx_b']  # 上下文
        continuations = sample['endings']
        return context, continuations

    def get_correct_answer(self, sample):
        return [sample['label']]


class MMLUEvaluator(LogLikelihoodEvaluator):

    def format_prompt(self, sample):
        """
        构建问题提示，形式为:
        "You are an expert in the field of text classification. Please choose the most appropriate option from [A, B, C, D] based on the given context and output only one option, followed directly by '#Answer: '."
        """
        # 模板和示例之间的上下文处理    A B C D   1 2 3 4  -> A
        prompt = f"""You are an expert in the field of text classification. Please choose the most appropriate option from [A, B, C, D] based on the given context and output only one option. \n"""
        for example in self.few_shot_examples[:self.num_few_shot]:
            prompt += f"Question: {example['question']}\n"
            prompt += "\n".join([f"{choice}" for idx, choice in enumerate(example['choices'])])
            prompt += f"\nAnswer: {example['answer']}\n\n"

        # 构建 MMLU 问题提示
        context = sample['question']
        choices = sample['choices']
        prompt += f"""Question: {context}\n"""
        # 添加选项
        prompt += "\n".join([f"{chr(65 + idx)}. {choice}" for idx, choice in enumerate(choices)])
        prompt += "\nAnswer: "

        return prompt, ["A", "B", "C", "D", "1","2","3","4"]

    def get_correct_answer(self, sample):
        # 0: A, 1:B, 2:C, 3:D
        correct_alpha = sample['answer'] # 已经对应具体的字母的索引位置，因为0位就是A，还要再加4即可，对应1的位置
        return [correct_alpha,correct_alpha+4]

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model_name = "/root/autodl-tmp/models/qwen/Qwen2-0___5B"
    model_name = "bert-base-uncased"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    evaluator = MMLUEvaluator(model,tokenizer, device=device,num_few_shot=5)
    evaluator.load_dataset('cais/mmlu', subset='all')
    few_shot_data = [
        {
            "question": "What is the capital of France?",
            "choices": ["A. Paris", "B. London", "C. Berlin", "D. Madrid"],
            "answer": "A"
            # question + choice -> choice: likehood -> token likelihood / length(token)  2023 -> lm harness
            # question + choices + [A,B,C,D] -> mmlu Answer: A / Answer: B
        },
        {
            "question": "What is 2+2?",
            "choices": ["A. 3", "B. 4", "C. 5", "D. 6"],
            "answer": "B"
        },
        {
            "question": "Who wrote '1984'?",
            "choices": ["A. J.K. Rowling", "B. Ernest Hemingway", "C. George Orwell", "D. Mark Twain"],
            "answer": "C"
        },
        {
            "question": "What is the boiling point of water?",
            "choices": ["A. 90 degrees Celsius", "B. 100 degrees Celsius", "C. 110 degrees Celsius",
                        "D. 120 degrees Celsius"],
            "answer": "B"
        },
        {
            "question": "What is the largest planet in the Solar System?",
            "choices": ["A. Earth", "B. Mars", "C. Jupiter", "D. Saturn"],
            "answer": "C"
        }
    ]

    evaluator.load_few_shot_examples(few_shot_data)
    evaluator.evaluate(num_samples=1000)
    # evaluator = HellaSwagEvaluator('bert-base-uncased', device=device)
    # evaluator.load_dataset('AlekseyKorshuk/hellaswag', dataset_split='validation')
    # evaluator.evaluate(num_samples=30)

# A,B,C,D -> 1,2,3,4

if __name__ == '__main__':
    main()
