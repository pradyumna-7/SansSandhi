import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter
import numpy as np
import pickle

# Constants
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
LABELS = ["NSP", "SP"]
MAX_LEN = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# Dataset Loader and Preprocessor
class SandhiDataset(Dataset):
    def __init__(self, data, char2idx, label2idx):
        self.samples = data
        self.char2idx = char2idx
        self.label2idx = label2idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chars, labels = self.samples[idx]
        x = [self.char2idx.get(c, self.char2idx[UNK_TOKEN]) for c in chars][:MAX_LEN]
        y = [self.label2idx[l] for l in labels][:MAX_LEN]
        pad_len = MAX_LEN - len(x)
        x += [self.char2idx[PAD_TOKEN]] * pad_len
        y += [self.label2idx["NSP"]] * pad_len
        return torch.tensor(x), torch.tensor(y)

# Model Definition (CNN + BiGRU)
class CNNGRUTagger(nn.Module):
    def __init__(self, vocab_size, label_size, embedding_dim=64, hidden_dim=128, cnn_out=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # CNN layers
        self.conv1 = nn.Conv1d(embedding_dim, cnn_out, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(cnn_out, cnn_out, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(cnn_out, cnn_out, kernel_size=7, padding=3)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        # BiGRU instead of BiLSTM
        self.gru = nn.GRU(cnn_out, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim, label_size)

    def forward(self, x):
        emb = self.embedding(x)  # (B, L, E)
        cnn_in = emb.permute(0, 2, 1)  # (B, E, L)

        out = self.relu(self.conv1(cnn_in))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.dropout(out)

        out = out.permute(0, 2, 1)  # (B, L, C)
        gru_out, _ = self.gru(out)  # (B, L, H)
        logits = self.fc(gru_out)  # (B, L, label_size)
        return logits


# Utility Functions
def read_dataset(path):
    samples = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                tokens, labels = line.strip().split("\t")
                chars = tokens.strip().split()
                lbls = labels.strip().split()
                if len(chars) == len(lbls):
                    samples.append((chars, lbls))
    return samples

def build_vocab(samples):
    chars = [c for x, _ in samples for c in x]
    char_counts = Counter(chars)
    vocab = [PAD_TOKEN, UNK_TOKEN] + sorted(char_counts.keys())
    char2idx = {c: i for i, c in enumerate(vocab)}
    label2idx = {l: i for i, l in enumerate(LABELS)}
    idx2label = {i: l for l, i in label2idx.items()}
    return char2idx, label2idx, idx2label

def train(model, dataloader, optimizer, loss_fn):
    model.train()
    for x, y in dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        logits = logits.view(-1, logits.shape[-1])
        y = y.view(-1)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

def evaluate(model, dataloader, idx2label):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            preds = torch.argmax(logits, dim=-1)
            for i in range(x.size(0)):
                length = (x[i] != 0).sum().item()
                all_preds.extend(preds[i][:length].tolist())
                all_labels.extend(y[i][:length].tolist())
    print(classification_report(all_labels, all_preds, target_names=[idx2label[i] for i in range(len(idx2label))]))

def predict_word(model, word, char2idx, idx2label):
    model.eval()
    chars = list(word)
    input_ids = [char2idx.get(c, char2idx[UNK_TOKEN]) for c in chars]
    x = torch.tensor(input_ids).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        preds = torch.argmax(logits, dim=-1).squeeze(0)
        pred_labels = [idx2label[idx.item()] for idx in preds[:len(chars)]]
    return list(zip(chars, pred_labels))

# Training Loop
if __name__ == "__main__":
    path = "dataset.txt"  # Input file
    raw_samples = read_dataset(path)
    char2idx, label2idx, idx2label = build_vocab(raw_samples)

    train_data, test_data = train_test_split(raw_samples, test_size=0.2, random_state=42)
    train_ds = SandhiDataset(train_data, char2idx, label2idx)
    test_ds = SandhiDataset(test_data, char2idx, label2idx)

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=32)

    model = CNNGRUTagger(len(char2idx), len(label2idx)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(100):  # Tweak based on dataset size
        print(f"\nEpoch {epoch+1}")
        train(model, train_dl, optimizer, loss_fn)
        evaluate(model, test_dl, idx2label)

    torch.save(model.state_dict(), "sandhi_model.pt")
    with open("sandhi_metadata.pkl", "wb") as f:
        pickle.dump({"char2idx": char2idx, "idx2label": idx2label}, f)

    print("\nPrediction for 'योयस्माज्जायते':")
    result = predict_word(model, "योयस्माज्जायते", char2idx, idx2label)
    print(" ".join([f"{c}({t})" for c, t in result]))


# Inference Script
def run_inference():
    with open("sandhi_metadata.pkl", "rb") as f:
        meta = pickle.load(f)
    char2idx, idx2label = meta["char2idx"], meta["idx2label"]

    model = CNNGRUTagger(len(char2idx), len(idx2label)).to(DEVICE)
    model.load_state_dict(torch.load("sandhi_model.pt", map_location=DEVICE))
    model.eval()

    while True:
        word = input("Enter Sanskrit compound word (or 'exit'): ").strip()
        if word.lower() == "exit":
            break
        result = predict_word(model, word, char2idx, idx2label)
        print("Prediction:", " ".join([f"{c}({t})" for c, t in result]))
        sandhi_split = ''.join([f"{c}|" if t == "SP" else c for c, t in result])
        print("Sandhi Split:", sandhi_split)
