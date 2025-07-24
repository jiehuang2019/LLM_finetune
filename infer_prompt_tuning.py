from transformers import AutoTokenizer
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSequenceClassification

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained("./my_prompt_model")

# 获取 Prompt Tuning 的配置
peft_config = PeftConfig.from_pretrained("./my_prompt_model")

# 加载基础模型 + 加载 Prompt 模块
base_model = AutoModelForSequenceClassification.from_pretrained(peft_config.base_model_name_or_path)
model = PeftModel.from_pretrained(base_model, "./my_prompt_model")

import torch

def predict_sentiment(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()
    return "positive" if predicted_class == 0 else "negative"

# 示例推理
print(predict_sentiment("This movie is awesome!"))   # 输出：positive
print(predict_sentiment("The plot was very boring.")) # 输出：negative

