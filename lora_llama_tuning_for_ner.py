from datasets import load_dataset

# 加载 CONLL-2003 命名实体识别数据集
dataset = load_dataset("conll2003")
labels = dataset["train"].features["ner_tags"].feature.names
id2label = {i: l for i, l in enumerate(labels)}
label2id = {l: i for i, l in enumerate(labels)}

from transformers import AutoTokenizer, AutoModelForTokenClassification, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

model_id = "meta-llama/Meta-Llama-3-8B"

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16")

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# 加载用于序列标注的 LLaMA
base_model = AutoModelForTokenClassification.from_pretrained(
    model_id,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    quantization_config=bnb_config,
    device_map="auto"
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    task_type=TaskType.TOKEN_CLS,
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

def tokenize_and_align_labels(example):
    tokenized = tokenizer(example["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    word_ids = tokenized.word_ids()
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100)
        elif word_idx != previous_word_idx:
            labels.append(example["ner_tags"][word_idx])
        else:
            labels.append(-100)  # 只标注第一个子词
        previous_word_idx = word_idx
    tokenized["labels"] = labels
    return tokenized

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification
import evaluate
import numpy as np

seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=-1)
    true_labels = [[labels[i][j] for j in range(len(labels[i])) if labels[i][j] != -100]
                   for i in range(len(labels))]
    true_predictions = [[predictions[i][j] for j in range(len(labels[i])) if labels[i][j] != -100]
                        for i in range(len(predictions))]
    true_labels = [[id2label[l] for l in seq] for seq in true_labels]
    true_predictions = [[id2label[p] for p in seq] for seq in true_predictions]
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {"f1": results["overall_f1"], "accuracy": results["overall_accuracy"]}

training_args = TrainingArguments(
    output_dir="./llama-ner-lora",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    #evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=20,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train()

def predict(text):
    tokens = text.split()
    inputs = tokenizer(tokens, return_tensors="pt", is_split_into_words=True).to(model.device)
    outputs = model(**inputs).logits
    preds = outputs.argmax(dim=-1)[0].tolist()
    pred_labels = [id2label[p] for i, p in enumerate(preds) if inputs.word_ids()[i] is not None]
    return list(zip(tokens, pred_labels))

print(predict("Hugging Face is based in New York City ."))

