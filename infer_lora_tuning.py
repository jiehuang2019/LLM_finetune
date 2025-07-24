from transformers import AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig

peft_config = PeftConfig.from_pretrained("./lora-bert-sst2")
base_model = AutoModelForSequenceClassification.from_pretrained(peft_config.base_model_name_or_path)
model = PeftModel.from_pretrained(base_model, "./lora-bert-sst2")

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    pred = logits.argmax(dim=-1).item()
    return "positive" if pred == 1 else "negative"

print(predict("This movie is amazing."))
print(predict("I hated the acting."))

