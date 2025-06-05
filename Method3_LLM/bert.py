import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset

train_path = "data/train.tsv"
test_paths = {
    "Test-1": "data/test_1.tsv",
    "Test-2": "data/test_2.tsv",
    "Test-3": "data/test_3.tsv"
}
label_names = ["0", "1", "2"]

model_name = "bert-base-multilingual-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

def tokenize(batch):
    return tokenizer(batch["Sentence"], truncation=True, padding="max_length", max_length=128)

train_df = pd.read_csv(train_path, sep="\t")
train_dataset = Dataset.from_pandas(train_df)
train_dataset = train_dataset.map(tokenize, batched=True)
train_dataset = train_dataset.rename_column("Label", "labels")
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="no",
    save_strategy="no",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    logging_dir="./logs",
    learning_rate=2e-5,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer)
)

trainer.train()

def save_classification_report_as_png(y_true, y_pred, labels, filename):
    report_dict = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose().round(2)

    plt.figure(figsize=(10, len(report_df) * 0.6))
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap="Blues", fmt=".2f", cbar=False)
    plt.title("Classification Report")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

os.makedirs("results", exist_ok=True)
output_md = ["# Evaluation Results\n"]

for name, path in test_paths.items():
    df = pd.read_csv(path, sep="\t")
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.rename_column("Label", "labels")
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    preds = trainer.predict(dataset)
    y_true = preds.label_ids
    y_pred = np.argmax(preds.predictions, axis=1)

    pd.DataFrame({"label": y_true}).to_csv(f"results/y_true_{name}.csv", index=False)
    pd.DataFrame({"label": y_pred}).to_csv(f"results/y_pred_{name}.csv", index=False)

    report_text = classification_report(y_true, y_pred, target_names=label_names)
    output_md.append(f"## {name}\n\n```\n{report_text}\n```")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {name}")
    plt.savefig(f"results/confusion_matrix_{name}.png")
    plt.close()

    save_classification_report_as_png(y_true, y_pred, label_names, filename=f"results/classification_report_{name}.png")

with open("results/metrics_results.md", "w", encoding="utf-8") as f:
    f.write("\n".join(output_md))

print("Training and evaluation complete. Results saved to 'results' folder.")
