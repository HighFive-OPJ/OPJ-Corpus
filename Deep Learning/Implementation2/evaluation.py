import pandas as pd 
import torch
import torch.nn as nn
from gensim.models.fasttext import load_facebook_model
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

FASTTEXT_PATH = "FastText.bin"
TEST_PATH = "Test-1.tsv"
LSTM_MODEL_PATH = "fasttext_lstm.pt"
CNN_MODEL_PATH = "fasttext_cnn.pt"

print("Učitavanje FastText modela...")
ft_model = load_facebook_model(FASTTEXT_PATH)
embedding_dim = ft_model.vector_size

def tokenize(text):
    return text.lower().split()

class FastTextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = [tokenize(text) for text in texts]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tokens = self.texts[idx]
        vectors = [ft_model.wv[token] if token in ft_model.wv else np.zeros(embedding_dim) for token in tokens]
        max_len = 50
        if len(vectors) > max_len:
            vectors = vectors[:max_len]
        else:
            vectors += [np.zeros(embedding_dim)] * (max_len - len(vectors))
        return torch.tensor(vectors, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

class CNNClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(embedding_dim, 100, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(100, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.pool(x).squeeze(2)
        x = self.dropout(x)
        return self.fc(x)

class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        hn = torch.cat((hn[-2], hn[-1]), dim=1)
        x = self.dropout(hn)
        return self.fc(x)

print("Učitavanje testnog skupa...")
test_df = pd.read_csv(TEST_PATH, sep="\t").rename(columns={"Sentence": "text", "Label": "label"})
test_df["label"] = test_df["label"].astype(int)
num_classes = test_df["label"].nunique()
label_names = sorted(test_df["label"].unique())

test_dataset = FastTextDataset(test_df["text"].tolist(), test_df["label"].tolist())
test_loader = DataLoader(test_dataset, batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(model, loader, model_name):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            preds = torch.argmax(output, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    print(f"\n=== Evaluacija: {model_name} ===")
    print("Distribucija predikcija:", np.bincount(all_preds))
    print("Stvarna distribucija:", np.bincount(all_labels))

    report = classification_report(all_labels, all_preds, digits=4, zero_division=0)
    print(report)

    cm = confusion_matrix(all_labels, all_preds, labels=label_names)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    os.makedirs("confusion_matrices", exist_ok=True)
    plt.savefig(f"confusion_matrices/confusion_matrix_{model_name.lower()}.png")
    plt.close()

print("\n=== EVALUACIJA: LSTM model ===")
lstm_model = LSTMClassifier(embedding_dim, hidden_dim=256, num_classes=num_classes)
lstm_model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=device))
lstm_model.to(device)
evaluate_model(lstm_model, test_loader, "LSTM")

print("\n=== EVALUACIJA: CNN model ===")
cnn_model = CNNClassifier(embedding_dim, num_classes=num_classes)
cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device))
cnn_model.to(device)
evaluate_model(cnn_model, test_loader, "CNN")
