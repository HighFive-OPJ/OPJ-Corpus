import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from gensim.models.fasttext import load_facebook_vectors
from sklearn.metrics import classification_report
from collections import defaultdict

train_df = pd.read_csv("Train-1.tsv", sep="\t")
test_df = pd.read_csv("Test-1.tsv", sep="\t")

train_sentences, train_labels = train_df['Sentence'].values, train_df['Label'].values
test_sentences, test_labels = test_df['Sentence'].values, test_df['Label'].values

def tokenize(text):
    return text.lower().split()

word_to_idx = {}
idx = 2 
word_to_idx['<PAD>'] = 0
word_to_idx['<UNK>'] = 1

def build_vocab(sentences):
    global idx
    for sentence in sentences:
        for word in tokenize(sentence):
            if word not in word_to_idx:
                word_to_idx[word] = idx
                idx += 1

build_vocab(train_sentences)

fasttext_model = load_facebook_vectors("FastText.bin")  

embedding_dim = 300
embedding_matrix = np.zeros((len(word_to_idx), embedding_dim))

for word, i in word_to_idx.items():
    if word in fasttext_model:
        embedding_matrix[i] = fasttext_model[word]
    else:
        embedding_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))

def encode_sentence(sentence, max_len=100):
    tokens = tokenize(sentence)
    ids = [word_to_idx.get(w, word_to_idx['<UNK>']) for w in tokens[:max_len]]
    if len(ids) < max_len:
        ids += [word_to_idx['<PAD>']] * (max_len - len(ids))
    return ids

class ReviewDataset(Dataset):
    def __init__(self, sentences, labels):
        self.sentences = [encode_sentence(s) for s in sentences]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.sentences[idx]), self.labels[idx]

train_dataset = ReviewDataset(train_sentences, train_labels)
test_dataset = ReviewDataset(test_sentences, test_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

class SentimentLSTM(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=128, output_dim=3):
        super().__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=False)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        out = self.dropout(hidden[-1])
        return self.fc(out)

class SentimentGRU(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=128, output_dim=3):
        super().__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=False)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        _, hidden = self.gru(x)
        out = self.dropout(hidden[-1])
        return self.fc(out)
    
class SentimentCNN(nn.Module):
    def __init__(self, embedding_matrix, output_dim=3, filter_sizes=[3, 4, 5], num_filters=100):
        super().__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=False)
        
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embedding_dim)) for fs in filter_sizes
        ])
        
        self.fc = nn.Linear(num_filters * len(filter_sizes), output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # Add channel dimension
        x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [torch.max(i, dim=2)[0] for i in x]
        x = torch.cat(x, dim=1)
        x = self.dropout(x)
        return self.fc(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, train_loader, epochs=5, lr=1e-3):
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            preds = model(x_batch)
            pred_labels = torch.argmax(preds, dim=1).cpu().numpy()
            all_preds.extend(pred_labels)
            all_labels.extend(y_batch.numpy())
    
    print(classification_report(all_labels, all_preds))

print("\nTraining LSTM model...")
lstm_model = SentimentLSTM(embedding_matrix)
train_model(lstm_model, train_loader)
evaluate_model(lstm_model, test_loader)

print("\nTraining GRU model...")
gru_model = SentimentGRU(embedding_matrix)
train_model(gru_model, train_loader)
evaluate_model(gru_model, test_loader)

print("\nTraining CNN model...")
cnn_model = SentimentCNN(embedding_matrix)
train_model(cnn_model, train_loader)
evaluate_model(cnn_model, test_loader)
