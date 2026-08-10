"""
train.py
--------
Trains the CRNN Urdu OCR model used by backend.py.

Fixes applied vs. the original dataset that shipped in vocab.zip:
  1. vocab.json was missing 4 characters that actually appear in
     dataset/labels.csv: 'ۂ', 'ۃ', 'ژ', 'أ'  (present in ~6.6% of samples).
     -> use vocab_fixed.json (57 -> 61 chars, 62 classes incl. CTC blank).
  2. dataset/labels.csv referenced 6 image files that don't exist in
     dataset/images/ (synth_001925.png, synth_002282.png, synth_002852.png,
     synth_003421.png, synth_003446.png, synth_004344.png).
     -> use dataset/labels_clean.csv (4994 valid rows).

Because the vocab size changed (58 -> 62 output classes), the old
best_model.pth is NOT compatible anymore and must be retrained with this
script. Run this, then copy the newly produced best_model.pth (and
vocab_fixed.json -> vocab.json) next to backend.py.

Usage:
    python train.py --epochs 30 --batch-size 32
"""

import argparse
import csv
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter, ImageEnhance

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(APP_ROOT, "dataset", "images")
LABELS_CSV = os.path.join(APP_ROOT, "dataset", "labels_clean.csv")
VOCAB_PATH = os.path.join(APP_ROOT, "vocab_fixed.json")
OUT_MODEL = os.path.join(APP_ROOT, "best_model.pth")

IMG_HEIGHT = 32
MAX_WIDTH = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- Vocab ----
class Vocab:
    def __init__(self, chars):
        self.chars = chars
        self.char_to_idx = {ch: idx + 1 for idx, ch in enumerate(self.chars)}
        self.idx_to_char = {idx + 1: ch for idx, ch in enumerate(self.chars)}
        self.idx_to_char[0] = "<blank>"

    @property
    def size(self):
        return len(self.chars) + 1

    def encode(self, text):
        return [self.char_to_idx[ch] for ch in text if ch in self.char_to_idx]

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            chars = json.load(f)
        return cls(chars)


# -------------------------------------------------------------- Dataset ----
def preprocess_image(pil_image, img_height=IMG_HEIGHT, max_width=MAX_WIDTH):
    img = pil_image.convert("L")
    w, h = img.size
    new_w = max(1, int(w * (img_height / h)))
    img = img.resize((new_w, img_height), Image.BILINEAR)

    arr = np.array(img, dtype=np.float32) / 255.0
    if new_w < max_width:
        pad = np.ones((img_height, max_width - new_w), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=1)
    else:
        arr = arr[:, :max_width]

    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)


def augment_image(img):
    """Randomly perturb a training image so the model generalizes beyond
    the clean synthetic renders (real photos/scans have blur, noise,
    rotation, uneven lighting, compression artifacts, etc.)."""
    if random.random() < 0.5:
        angle = random.uniform(-3, 3)
        img = img.rotate(angle, expand=True, fillcolor=255)

    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.2)))

    if random.random() < 0.5:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
    if random.random() < 0.5:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))

    if random.random() < 0.4:
        arr = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, random.uniform(3, 12), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    return img


class OCRDataset(Dataset):
    def __init__(self, rows, vocab, augment=False):
        self.rows = rows
        self.vocab = vocab
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        fname, text = self.rows[idx]
        img = Image.open(os.path.join(IMG_DIR, fname)).convert("L")
        if self.augment:
            img = augment_image(img)
        tensor = preprocess_image(img)
        target = torch.tensor(self.vocab.encode(text), dtype=torch.long)
        return tensor, target, len(target)


def collate_fn(batch):
    imgs, targets, lengths = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    targets_cat = torch.cat(targets)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return imgs, targets_cat, lengths


# ------------------------------------------------------------------ Model --
class CRNN(nn.Module):
    def __init__(self, img_height=32, num_channels=1, num_classes=58, rnn_hidden=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 64, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.rnn = nn.LSTM(512, rnn_hidden, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(rnn_hidden * 2, num_classes)

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        conv = conv.squeeze(2).permute(0, 2, 1)
        rnn_out, _ = self.rnn(conv)
        logits = self.fc(rnn_out)
        log_probs = torch.log_softmax(logits, dim=2).permute(1, 0, 2)  # (T, B, C)
        return log_probs


# ---------------------------------------------------------------- Train ----
def load_rows():
    with open(LABELS_CSV, encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)  # header
        return [row for row in r if len(row) >= 2 and row[1].strip() != ""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-split", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    vocab = Vocab.load(VOCAB_PATH)
    print(f"Vocab size (incl. blank): {vocab.size}")

    rows = load_rows()
    random.shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_split))
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    print(f"Train samples: {len(train_rows)}  Val samples: {len(val_rows)}")

    train_ds = OCRDataset(train_rows, vocab, augment=True)
    val_ds = OCRDataset(val_rows, vocab, augment=False)
    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=num_workers)

    model = CRNN(img_height=IMG_HEIGHT, num_channels=1, num_classes=vocab.size, rnn_hidden=256).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, targets, target_lengths in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            log_probs = model(imgs)  # (T, B, C)
            input_lengths = torch.full((imgs.size(0),), log_probs.size(0), dtype=torch.long, device=device)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)

        train_loss = total_loss / len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets, target_lengths in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                target_lengths = target_lengths.to(device)
                log_probs = model(imgs)
                input_lengths = torch.full((imgs.size(0),), log_probs.size(0), dtype=torch.long, device=device)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(val_ds)

        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "img_height": IMG_HEIGHT,
                "max_width": MAX_WIDTH,
                "rnn_hidden": 256,
                "num_classes": vocab.size,
            }, OUT_MODEL)
            print(f"  -> saved new best model (val_loss={val_loss:.4f}) to {OUT_MODEL}")

    print("Done. Remember to also copy vocab_fixed.json over vocab.json next to backend.py.")


if __name__ == "__main__":
    main()
