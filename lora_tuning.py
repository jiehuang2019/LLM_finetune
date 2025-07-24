from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, TaskType, get_peft_model
import evaluate
import numpy as np

# 1. 加载 SST-2 数据集
dataset = load_dataset("glue", "sst2")

# 2. 加载 tokenizer，并对句子进行预处理
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess(example):
    return tokenizer(example["sentence"], truncation=True)

tokenized_datasets = dataset.map(preprocess, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 3. 加载 BERT 模型用于分类
base_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 4. 配置 LoRA（在注意力模块上注入低秩可训练参数）
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS,
    target_modules=["query", "value"],  # 针对 BERT 模型注意力结构
)

# 5. 应用 LoRA
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# 6. 指定训练参数
training_args = TrainingArguments(
    output_dir="./lora-bert-sst2",
    #evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=10,
    report_to="none"
)

# 7. 准备评估函数
metric = evaluate.load("glue", "sst2")

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels)

# 8. 创建 Trainer 并开始训练
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

model.save_pretrained("./lora-bert-sst2")
tokenizer.save_pretrained("./lora-bert-sst2")

