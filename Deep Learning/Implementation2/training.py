import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from gensim.models.fasttext import load_facebook_model
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np

FASTTEXT_PATH = "FastText.bin"
TRAIN_PATH = "TRAIN.tsv"
VALIDATION_PATH = "Test-1.tsv"  

print("Učitavanje FastText modela...")
ft_model = load_facebook_model(FASTTEXT_PATH)
embedding_dim = ft_model.vector_size
MAX_LEN = 100  

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
        if len(vectors) > MAX_LEN:
            vectors = vectors[:MAX_LEN]
        else:
            vectors += [np.zeros(embedding_dim)] * (MAX_LEN - len(vectors))
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
        x = torch.cat((hn[-2], hn[-1]), dim=1)
        x = self.dropout(x)
        return self.fc(x)

print("Učitavanje podataka...")
train_df = pd.read_csv(TRAIN_PATH, sep="\t").rename(columns={"Sentence": "text", "Label": "label"})
val_df = pd.read_csv(VALIDATION_PATH, sep="\t").rename(columns={"Sentence": "text", "Label": "label"})

train_df["label"] = train_df["label"].astype(int)
val_df["label"] = val_df["label"].astype(int)

train_texts = train_df["text"].tolist()
train_labels = train_df["label"].tolist()

val_texts = val_df["text"].tolist()
val_labels = val_df["label"].tolist()

num_classes = train_df["label"].nunique()

train_dataset = FastTextDataset(train_texts, train_labels)
val_dataset = FastTextDataset(val_texts, val_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            preds = torch.argmax(output, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

for model_type in ["LSTM", "CNN"]:
    print(f"\n==============================")
    print(f"Treniramo model: {model_type}")
    print(f"==============================")

    if model_type == "LSTM":
        model = LSTMClassifier(embedding_dim, hidden_dim=256, num_classes=num_classes)
    else:
        model = CNNClassifier(embedding_dim, num_classes=num_classes)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 11):
        train_loss = train(model, train_loader, optimizer, criterion)
        val_acc = evaluate(model, val_loader)
        print(f"{model_type} | Epoch {epoch} | Loss: {train_loss:.4f} | Validation Accuracy: {val_acc:.4f}")

    model_path = f"fasttext_{model_type.lower()}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"{model_type} model spremljen kao: {model_path}")
