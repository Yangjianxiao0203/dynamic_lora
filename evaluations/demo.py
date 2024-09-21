import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

#MMLU: 获取4个输出的概率，然后看哪个最高
#https://github.com/modelscope/modelscope-classroom/blob/main/LLM-tutorial/K.%E5%A4%A7%E6%A8%A1%E5%9E%8B%E8%87%AA%E5%8A%A8%E8%AF%84%E4%BC%B0%E7%90%86%E8%AE%BA%E5%92%8C%E5%AE%9E%E6%88%98--LLM%20Automatic%20Evaluation.md
#TODO: 读这个https://github.com/modelscope/modelscope-classroom/tree/main


# 加载模型和tokenizer
model_name = "gpt2"  # 替换为你自己的模型路径
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 将模型设置为评估模式
model.eval()

# 输入文本
input_text = "The quick brown fox jumps over"

# 对输入文本进行tokenization
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# 获取模型输出的logits
with torch.no_grad():
    outputs = model(input_ids)
logits = outputs.logits

# 获取最后一个token的logits
last_token_logits = logits[0, -1, :]  # 只取最后一个位置的logits

# 使用softmax将logits转化为概率分布
probs = torch.softmax(last_token_logits, dim=-1)

# 定义你要查询的单词
words = ["A", "B", "C", "D"]

# 获取每个单词对应的token id
word_ids = [tokenizer.convert_tokens_to_ids(word) for word in words]

# 获取每个单词的概率
word_probs = {word: probs[wid].item() for word, wid in zip(words, word_ids)}

# 输出结果
print("Probabilities for each word:")
for word, prob in word_probs.items():
    print(f"{word}: {prob:.5f}")
