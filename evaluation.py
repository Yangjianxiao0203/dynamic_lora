import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, StoppingCriteria, StoppingCriteriaList, \
    TextIteratorStreamer

from evaluations.evaluator import MMLUEvaluator
from dynamic_lora import DynamicLoRAManager

model_name = "/root/autodl-tmp/models/qwen/Qwen2-0___5B"


def mmlu_evaluate(model, tokenizer, device):
    evaluator = MMLUEvaluator(model, tokenizer, device=device, num_few_shot=5)
    evaluator.load_dataset('cais/mmlu', subset='all')
    few_shot_data = [
        {"question": "What is the capital of France?", "answer": "Paris"},
        {"question": "What is 2+2?", "answer": "4"},
        {"question": "Who wrote '1984'?", "answer": "George Orwell"},
        {"question": "What is the boiling point of water?", "answer": "100 degrees Celsius"},
        {"question": "What is the largest planet in the Solar System?", "answer": "Jupiter"},
    ]
    evaluator.load_few_shot_examples(few_shot_data)
    evaluator.evaluate(num_samples=1000)


def evaluate_original():
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mmlu_evaluate(model, tokenizer, device)


def evaluate_hf_peft():
    adapters_name = "output/qwen15-alpaca/final_model_r_32"
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    model = PeftModel.from_pretrained(model, adapters_name)
    print("loaded peft, start merging")
    model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mmlu_evaluate(model, tokenizer, device)


def evaluate_torch_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lora_path = "lora_state.pth"
    model_path = model_name
    # Load model and tokenizer
    print(f"Loading model {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    r_max = 32
    keywords = ["q_proj", "k_proj", "v_proj"]
    dynamic_lora_manager = DynamicLoRAManager(model, r_max, keywords)
    dynamic_lora_manager.replace_with_dynamic_lora()
    dynamic_lora_manager.load_lora_layers(lora_path)
    model = dynamic_lora_manager.model
    mmlu_evaluate(model, tokenizer, device)


if __name__ == '__main__':
    # evaluate_hf_peft()
    # evaluate_original()
    evaluate_torch_model()
