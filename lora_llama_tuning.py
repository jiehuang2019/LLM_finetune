from datasets import load_dataset

# 使用 Alpaca 格式的数据集
dataset = load_dataset("tatsu-lab/alpaca", split="train[:5000]")  # 只取前5000条以加快训练

def format_alpaca(example):
    prompt = example["instruction"]
    if example["input"]:
        prompt += "\n" + example["input"]
    prompt += "\n\n### Response:\n"
    return {
        "prompt": prompt,
        "completion": example["output"]
    }

dataset = dataset.map(format_alpaca)

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "meta-llama/Meta-Llama-3-8B"  # 需要从 Hugging Face 下载权重

# 量化配置（4bit 减少内存）
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16")

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True
)

from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

def tokenize(example):
    full_prompt = example["prompt"] + example["completion"]
    tokens = tokenizer(full_prompt, truncation=True, padding="max_length", max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names, batched=True)

from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

training_args = TrainingArguments(
    output_dir="./llama3-lora-alpaca",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    #evaluation_strategy="no",
    save_strategy="epoch",
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True,
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()
model.save_pretrained("./llama3-lora-alpaca")
tokenizer.save_pretrained("./llama3-lora-alpaca")

from peft import PeftModel, PeftConfig

# 加载训练后模型
peft_config = PeftConfig.from_pretrained("./llama3-lora-alpaca")
base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, device_map="auto")
model = PeftModel.from_pretrained(base_model, "./llama3-lora-alpaca")
model.eval()

# 推理
input_text = "Explain what is HTTP protocol."
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

