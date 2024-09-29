import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from string import Template
from abc import ABC, abstractmethod


class LogLikelihoodEvaluator(ABC):
    def __init__(self, model_name, device='cuda'):
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device
        self.prompt_template = None

    def load_dataset(self, dataset_name, dataset_split='validation', subset=None, **kwargs):
        # 通用的数据集加载方法，允许传入不同的数据集名称
        if subset:
            self.dataset = load_dataset(dataset_name, subset, split=dataset_split, **kwargs)
        else:
            self.dataset = load_dataset(dataset_name, split=dataset_split, **kwargs)
        print(f"Loaded dataset {dataset_name} with {len(self.dataset)} samples.")

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

            log_likelihood = selected_logits.sum().item()
            loglikelihoods.append(log_likelihood)

        return loglikelihoods

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
    def get_correct_answer(self, sample) -> str:
        pass

    def evaluate_sample(self, sample):
        # 评估单个样本
        context, continuations = self.format_prompt(sample)
        correct_answer_idx = self.get_correct_answer(sample)

        # 计算每个选项的log-likelihood
        loglikelihoods = self.calculate_loglikelihood(context, continuations)

        # 选择log-likelihood最大的选项
        predicted_idx = torch.argmax(torch.tensor(loglikelihoods)).item()
        correct = predicted_idx == correct_answer_idx

        return correct, predicted_idx, correct_answer_idx, loglikelihoods

    def evaluate(self, num_samples=100):
        # 评估多个样本的准确率
        correct_count = 0
        for i, sample in enumerate(self.dataset.select(range(num_samples))):
            correct, predicted_idx, correct_answer_idx, loglikelihoods = self.evaluate_sample(sample)
            correct_count += int(correct)
            print(
                f"[{i}]: Predicted {predicted_idx}, Correct {correct_answer_idx}, Log-likelihoods: {loglikelihoods}")

        accuracy = correct_count / num_samples
        print(f"Accuracy: {accuracy:.2f}")
        return accuracy


class HellaSwagEvaluator(LogLikelihoodEvaluator):

    def format_prompt(self, sample):
        """格式化 HellaSwag 的问题和选项"""

        context = sample['ctx_a'] + " " + sample['ctx_b']  # 上下文
        continuations = sample['endings']
        return context, continuations

    def get_correct_answer(self, sample) -> str:
        return sample['label']


class MMLUEvaluator(LogLikelihoodEvaluator):

    def format_prompt(self, sample):
        self.prompt_template = Template("Question: $question\nAnswer:")
        context = self.prompt_template.substitute(question=sample['question'], answer='')
        continuations = sample['choices']
        return context, continuations

    def get_correct_answer(self, sample) -> str:
        return sample['answer']



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # evaluator = HellaSwagEvaluator('bert-base-uncased', device=device)
    evaluator = MMLUEvaluator('bert-base-uncased', device=device)
    evaluator.load_dataset('cais/mmlu', subset='astronomy')
    evaluator.evaluate(num_samples=10)

    # evaluator.load_dataset('Rowan/hellaswag', dataset_split='validation')
    # evaluator.evaluate(num_samples=10)


if __name__ == '__main__':
    main()
