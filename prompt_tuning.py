from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import transformers
import inspect
print(transformers.__version__)
print(transformers.TrainingArguments.__module__)
import transformers
print("Transformers version:", transformers.__version__)
print("TrainingArguments module:", transformers.TrainingArguments.__module__)
print("TrainingArguments location:", inspect.getfile(transformers.TrainingArguments))

print("Trainer module:", transformers.Trainer.__module__)
print("Trainer location:", inspect.getfile(transformers.Trainer))


from datasets import load_dataset
from peft import PromptTuningConfig, PromptTuningInit, TaskType, get_peft_model
import evaluate
import numpy as np

# 1. 加载 SST-2 数据集
dataset = load_dataset("glue", "sst2")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess_function(example):
    return tokenizer(example["sentence"], truncation=True)

tokenized_datasets = dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 2. 加载 BERT 模型
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 3. 配置 Prompt Tuning（添加 20 个可训练 soft tokens）
peft_config = PromptTuningConfig(
    task_type=TaskType.SEQ_CLS,             # 任务类型为序列分类
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=20,
    prompt_tuning_init_text="Classify the sentiment of this sentence:",
    tokenizer_name_or_path="bert-base-uncased"
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # 仅 soft prompt 参数是可训练的

# 4. 评估函数
metric = evaluate.load("glue", "sst2")
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels)

# 5. 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    #evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_steps=10,
    report_to="none",
)

# 6. 创建 Trainer 并训练
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

# 保存模型和 tokenizer（包括 soft prompt 参数）
model.save_pretrained("./my_prompt_model")
tokenizer.save_pretrained("./my_prompt_model")

