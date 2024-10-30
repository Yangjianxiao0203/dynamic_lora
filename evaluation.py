import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, StoppingCriteria, StoppingCriteriaList, \
    TextIteratorStreamer

from evaluations.evaluator import MMLUEvaluator, HellaSwagEvaluator
from dynamic_lora import DynamicLoRAManager

model_name = "/root/autodl-tmp/models/qwen/Qwen2-0___5B"

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters())

def mmlu_evaluate(model, tokenizer, device):
    evaluator = MMLUEvaluator(model, tokenizer, device=device, num_few_shot=5)
    evaluator.load_dataset('cais/mmlu', subset='all')
    few_shot_data = [
        {
            "question": "What is the capital of France?",
            "choices": ["A. Paris", "B. London", "C. Berlin", "D. Madrid"],
            "answer": "A"
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
    # evaluator.load_few_shot_examples(few_shot_data)
    evaluator.evaluate(num_samples=1000)


def hellaswag_evaluate(model, tokenizer, device):
    evaluator = HellaSwagEvaluator(model, tokenizer, device=device)
    evaluator.load_dataset('AlekseyKorshuk/hellaswag')
    evaluator.evaluate(num_samples=1000)

def evaluate_original(name= "mmlu"):
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(count_trainable_params(model)) # 494032768
    if name == "hellaswag":
        hellaswag_evaluate(model, tokenizer, device)
        return
    if name == "mmlu":
        mmlu_evaluate(model, tokenizer, device)
        return
    #mmlu_evaluate(model, tokenizer, device)


def evaluate_hf_peft(name= "mmlu"):
    adapters_name = "output/qwen15-alpaca/final_model_r_32"
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    model = PeftModel.from_pretrained(model, adapters_name)
    # print("loaded peft, start merging")
    # model = model.merge_and_unload()
    print(count_trainable_params(model)) #496981888
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # mmlu_evaluate(model, tokenizer, device)
    if name == "hellaswag":
        hellaswag_evaluate(model, tokenizer, device)
        return
    if name == "mmlu":
        mmlu_evaluate(model, tokenizer, device)
        return


def evaluate_torch_model(name= "mmlu"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lora_path = "lora_state.pth"
    model_path = model_name
    # Load model and tokenizer
    print(f"Loading model {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    r_max = 32
    alpha = 64
    keywords = ["q_proj", "k_proj", "v_proj"]
    dynamic_lora_manager = DynamicLoRAManager(model, r_max, keywords,alpha)
    dynamic_lora_manager.replace_with_dynamic_lora()
    dynamic_lora_manager.load_lora_layers(lora_path)
    model = dynamic_lora_manager.model
    # mmlu_evaluate(model, tokenizer, device)
    if name == "hellaswag":
        hellaswag_evaluate(model, tokenizer, device)
        return
    if name == "mmlu":
        mmlu_evaluate(model, tokenizer, device)
        return


if __name__ == '__main__':
    name = "hellaswag"
    # evaluate_hf_peft(name)
    # evaluate_original(name)
    evaluate_torch_model(name)
