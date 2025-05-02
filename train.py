import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from collections import Counter
import numpy as np

from data.audio_dataloader import AudioDataloader
from model import LSTMClassifier, GRUClassifier
from metrics import FocalLoss, framewise_accuracy, weighted_chord_symbol_recall_full

# === 参数配置 ===
NUM_CLASSES = 25
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 80
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(model_type: str):
    if model_type == "lstm":
        return LSTMClassifier(
            input_size=168,
            hidden_dim=128,
            output_size=NUM_CLASSES,
            num_layers=2,
            use_gpu=torch.cuda.is_available(),
            bidirectional=True,
            dropout=(0.4, 0.3, 0.3)
        )
    elif model_type == "gru":
        return GRUClassifier(
            input_size=168,
            hidden_dim=128,
            output_size=NUM_CLASSES,
            num_layers=2,
            use_gpu=torch.cuda.is_available(),
            bidirectional=True,
            dropout=(0.4, 0.3, 0.3)
        )
    else:
        raise ValueError("Unsupported model type. Use 'lstm' or 'gru'.")

def train(model_type: str):
    print(f"🚀 Starting training with model: {model_type.upper()}")
    
    # === 数据加载 ===
    config = {
        "mp3": {"song_hz": 22050, "inst_len": 10.0, "skip_interval": 5.0},
        "feature": {"n_bins": 168, "bins_per_octave": 24, "hop_length": 512}
    }

    dataloader = AudioDataloader(
        config=config,
        data_root_dir="./data",
        batch_size=BATCH_SIZE,
        shuffle=True,
        augment=False,
        featuretype="cqt",
        num_workers=4
    )

    train_loader, val_loader, _, _ = dataloader.get_loaders()

    # === 模型初始化 ===
    model = get_model(model_type).to(DEVICE)

    # === 类别权重计算 ===
    all_labels = []
    for _, y, _ in train_loader:
        y = y.view(-1)
        y = y[y != -100]
        all_labels.extend(y.cpu().numpy())
    label_counts = Counter(all_labels)
    total_labels = sum(label_counts.values())
    weights = [total_labels / (NUM_CLASSES * label_counts.get(i, 1)) for i in range(NUM_CLASSES)]
    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    # === 损失函数和优化器 ===
    criterion = FocalLoss(gamma=2.0, weight=class_weights, ignore_index=-100, reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_wcsr = 0.0
    model_save_path = f"best_{model_type}_model.pt"

    # === 训练主循环 ===
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, total_acc, total_steps = 0, 0, 0
        for X, y, lengths in tqdm(train_loader, desc=f"Epoch {epoch}"):
            X, y, lengths = X.to(DEVICE), y.to(DEVICE), lengths.to(DEVICE)
            logits = model(X, lengths)
            loss = criterion(logits.reshape(-1, NUM_CLASSES), y.reshape(-1))
            acc = framewise_accuracy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += acc
            total_steps += 1

        avg_loss = total_loss / total_steps
        avg_acc = total_acc / total_steps
        val_wcsr = weighted_chord_symbol_recall_full(model, val_loader, DEVICE)

        print(f"\n[Epoch {epoch}] Loss: {avg_loss:.4f}, Frame Acc: {avg_acc:.4f}, Val WCSR: {val_wcsr:.4f}")

        if val_wcsr > best_wcsr:
            best_wcsr = val_wcsr
            torch.save(model.state_dict(), model_save_path)
            print(f"✅ Saved best model to: {model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="lstm", choices=["lstm", "gru"], help="Model type to train")
    args = parser.parse_args()

    train(args.model)
