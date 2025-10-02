import os
import argparse
import random
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

import librosa
import librosa.display

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -------------------------
# Args and reproducibility
# -------------------------
parser = argparse.ArgumentParser(description="Train CNN for music genre classification")
parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset folder (genrefolders)")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--sr", type=int, default=22050, help="Audio sample rate")
parser.add_argument("--duration", type=float, default=30.0, help="Clip duration in seconds")
parser.add_argument("--n_mels", type=int, default=128)
parser.add_argument("--hop_length", type=int, default=512)
parser.add_argument("--n_fft", type=int, default=2048)
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--save_path", type=str, default="genre_cnn.pth")
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.device.startswith("cuda"):
    torch.cuda.manual_seed_all(args.seed)

# -------------------------
# Dataset helper
# -------------------------
def find_genre_folders(data_dir: str) -> List[str]:
    genres = [d.name for d in Path(data_dir).iterdir() if d.is_dir()]
    genres.sort()
    return genres

def collect_files(data_dir: str, genres: List[str]) -> Tuple[List[str], List[int]]:
    filepaths = []
    labels = []
    for i, g in enumerate(genres):
        p = Path(data_dir) / g
        for ext in ("*.wav", "*.au", "*.mp3", "*.flac"):
            for f in p.glob(ext):
                filepaths.append(str(f))
                labels.append(i)
    return filepaths, labels

class GTZANDataset(Dataset):
    """
    On-the-fly mel-spectrogram dataset.
    Returns: (mel_tensor, label)
    mel_tensor shape: (1, n_mels, time_frames)
    """
    def __init__(self, filepaths: List[str], labels: List[int], sr=22050, duration=30.0,
                 n_mels=128, n_fft=2048, hop_length=512, augment=False):
        self.filepaths = filepaths
        self.labels = labels
        self.sr = sr
        self.duration = duration
        self.samples = int(sr * duration)
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.augment = augment

    def __len__(self):
        return len(self.filepaths)

    def load_audio(self, path: str):
        y, sr = librosa.load(path, sr=self.sr, mono=True, duration=self.duration, res_type='kaiser_fast')
        if len(y) < self.samples:
            padding = self.samples - len(y)
            y = np.pad(y, (0, padding), mode='constant')
        elif len(y) > self.samples:
            y = y[: self.samples]
        return y

    def compute_mel(self, y: np.ndarray):
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db_norm = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)
        return mel_db_norm.astype(np.float32)

    def __getitem__(self, idx: int):
        path = self.filepaths[idx]
        label = self.labels[idx]
        try:
            y = self.load_audio(path)
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")
            # Pick a random valid sample instead
            new_idx = random.randint(0, len(self.filepaths) - 1)
            return self.__getitem__(new_idx)
        if self.augment:
            shift = np.random.randint(low=-int(0.1*self.sr), high=int(0.1*self.sr))
            y = np.roll(y, shift)
            y = y + 0.001 * np.random.randn(len(y))
        mel = self.compute_mel(y)
        mel = np.expand_dims(mel, axis=0)  # (1, n_mels, time)
        return torch.tensor(mel), torch.tensor(label, dtype=torch.long)

# -------------------------
# Model
# -------------------------
class CNNGenre(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# -------------------------
# Training / Evaluation
# -------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []
    preds_all = []
    labels_all = []
    for xb, yb in tqdm(loader, desc="train batches", leave=True):
        xb = xb.to(device)
        yb = yb.to(device)
        out = model(xb)
        loss = criterion(out, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        preds = out.argmax(dim=1).detach().cpu().numpy()
        preds_all.extend(preds.tolist())
        labels_all.extend(yb.detach().cpu().numpy().tolist())
    avg_loss = float(np.mean(losses))
    acc = accuracy_score(labels_all, preds_all)
    return avg_loss, acc

def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    preds_all = []
    labels_all = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            losses.append(loss.item())
            preds = out.argmax(dim=1).cpu().numpy()
            preds_all.extend(preds.tolist())
            labels_all.extend(yb.cpu().numpy().tolist())
    avg_loss = float(np.mean(losses)) if losses else 0.0
    acc = accuracy_score(labels_all, preds_all) if labels_all else 0.0
    return avg_loss, acc, labels_all, preds_all

# -------------------------
# Main
# -------------------------
def main():
    data_dir = args.data_dir
    genres = find_genre_folders(data_dir)
    if not genres:
        raise ValueError(f"No genre folders found in {data_dir}. Expected folder/genre structure.")
    print(f"Found genres ({len(genres)}): {genres}")

    filepaths, labels = collect_files(data_dir, genres)
    if len(filepaths) == 0:
        raise ValueError("No audio files found under data_dir. Check extensions and folder layout.")

    train_files, test_files, train_labels, test_labels = train_test_split(
        filepaths, labels, test_size=0.2, stratify=labels, random_state=args.seed
    )

    train_dataset = GTZANDataset(train_files, train_labels,
                                 sr=args.sr, duration=args.duration,
                                 n_mels=args.n_mels, n_fft=args.n_fft, hop_length=args.hop_length,
                                 augment=True)
    test_dataset = GTZANDataset(test_files, test_labels,
                                sr=args.sr, duration=args.duration,
                                n_mels=args.n_mels, n_fft=args.n_fft, hop_length=args.hop_length,
                                augment=False)

    # Windows fix: num_workers=0
    num_workers = 0
    pin_memory = True if args.device.startswith("cuda") else False

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    device = torch.device(args.device)
    model = CNNGenre(n_classes=len(genres)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_labels, val_preds = evaluate(model, test_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Train loss: {train_loss:.4f} acc: {train_acc:.4f}")
        print(f"Val   loss: {val_loss:.4f} acc: {val_acc:.4f}")
        print(f"Epoch time: {time.time()-t0:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "genres": genres
            }, args.save_path)
            print(f"Saved best model to {args.save_path} with val_acc={best_val_acc:.4f}")

    val_loss, val_acc, val_labels, val_preds = evaluate(model, test_loader, criterion, device)
    print("\nFinal evaluation on test set:")
    print(f"Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")
    print("\nClassification report:")
    print(classification_report(val_labels, val_preds, target_names=genres, digits=4))

    cm = confusion_matrix(val_labels, val_preds)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(genres))
    plt.xticks(tick_marks, genres, rotation=45)
    plt.yticks(tick_marks, genres)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("Saved confusion matrix to confusion_matrix.png")

    np.savez("training_history.npz", **history)
    print("Saved training history to training_history.npz")

if __name__ == "__main__":
    main()
