from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("glue", "sst2")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(example):
    return tokenizer(example["sentence"], truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

from transformers import AutoModelWithHeads

# 使用 Adapter 增强的模型类
model = AutoModelWithHeads.from_pretrained("bert-base-uncased")

# 添加一个新的分类任务 adapter（不改动原模型参数）
model.add_adapter("sst2_adapter")
model.train_adapter("sst2_adapter")  # 仅训练 adapter
model.set_active_adapters("sst2_adapter")

# 添加分类头
model.add_classification_head("sst2_head", num_labels=2)
model.train_head("sst2_head")  # 训练头部

from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
import evaluate
import numpy as np

training_args = TrainingArguments(
    output_dir="./results_adapter",
    #evaluation_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

metric = evaluate.load("glue", "sst2")

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# 保存整个模型（包含 adapter 和 head）
model.save_adapter("./sst2_adapter", "sst2_adapter")

# 加载 adapter（在推理中也可只加载 adapter）
model.load_adapter("./sst2_adapter", load_as="sst2_adapter")
model.set_active_adapters("sst2_adapter")

def predict(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    pred = outputs.logits.argmax(dim=-1).item()
    return "positive" if pred == 1 else "negative"

print(predict("This movie is awesome."))
print(predict("I didn't like the acting."))

