import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

set_seed(42)

dataset = load_dataset("csv", delimiter="\t", data_files={
    "train": "data/train.tsv",
    "test_1": "data/test_1.tsv",
})
test_2 = load_dataset("csv", delimiter="\t", data_files={"test": "data/test_2.tsv"})["test"]
test_3 = load_dataset("csv", delimiter="\t", data_files={"test": "data/test_3.tsv"})["test"]

full_train = dataset["train"].train_test_split(test_size=0.1, seed=12345)
dataset_train = full_train["train"]
dataset_valid = full_train["test"]

model_name = "classla/bcms-bertic"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

def tokenize(batch):
    return tokenizer(batch["Sentence"], padding=True, truncation=True, max_length=128)

dataset_train = dataset_train.map(tokenize, batched=True)
dataset_valid = dataset_valid.map(tokenize, batched=True)
dataset["test_1"] = dataset["test_1"].map(tokenize, batched=True)
test_2 = test_2.map(tokenize, batched=True)
test_3 = test_3.map(tokenize, batched=True)

dataset_train = dataset_train.rename_column("Label", "labels")
dataset_valid = dataset_valid.rename_column("Label", "labels")
dataset["test_1"] = dataset["test_1"].rename_column("Label", "labels")
test_2 = test_2.rename_column("Label", "labels")
test_3 = test_3.rename_column("Label", "labels")

columns = ["input_ids", "attention_mask", "labels"]
dataset_train.set_format("torch", columns=columns)
dataset_valid.set_format("torch", columns=columns)
dataset["test_1"].set_format("torch", columns=columns)
test_2.set_format("torch", columns=columns)
test_3.set_format("torch", columns=columns)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

training_args = TrainingArguments(
    output_dir="./bertic-our-group",
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.03,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_valid,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

trainer.train()

print("\nTraining Evaluation:")
train_metrics = trainer.evaluate(dataset_train)
for k, v in train_metrics.items():
    print(f"{k}: {v:.4f}")

print("\nValidation Evaluation:")
val_metrics = trainer.evaluate(dataset_valid)
for k, v in val_metrics.items():
    print(f"{k}: {v:.4f}")

print("\nTest Set 1 Evaluation (Group 1):")
test_1_metrics = trainer.evaluate(dataset["test_1"])
for k, v in test_1_metrics.items():
    print(f"{k}: {v:.4f}")

print("\nTest Set 2 Evaluation (Group 2):")
test_2_metrics = trainer.evaluate(test_2)
for k, v in test_2_metrics.items():
    print(f"{k}: {v:.4f}")

print("\nTest Set 3 Evaluation (Group 3):")
test_3_metrics = trainer.evaluate(test_3)
for k, v in test_3_metrics.items():
    print(f"{k}: {v:.4f}")

trainer.model.save_pretrained("bertic")
tokenizer.save_pretrained("bertic")
