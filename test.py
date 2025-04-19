import torch
import torch.nn as nn
from data.audio_dataloader import AudioDataloader
import os

from data.preprocess import FeatureTypes

# === 参数配置 ===
EPOCHS = 20
BATCH_SIZE = 16
LR = 1e-3
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_CLASSES = 25  # 12maj + 12min + N
DEVICE = torch.device("cpu")


# === 模型定义 ===
class BiLSTMChordRecognizer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=0.3,
                            bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)  # out: (B, T, 2H)
        logits = self.classifier(out)  # (B, T, C)
        return logits


# === 准确率计算函数 ===
def accuracy(logits, labels):
    preds = torch.argmax(logits, dim=-1)
    mask = (labels != -100)
    correct = (preds == labels) & mask
    return correct.sum().item() / mask.sum().item()


# === 主训练函数 ===
def train():
    # 模拟 config（需与你 preprocess 使用的一致）
    config = {
        "mp3": {
            "song_hz": 22050,
            "inst_len": 10.0,
            "skip_interval": 5.0
        },
        "feature": {
            "n_bins": 168,
            "bins_per_octave": 24,
            "hop_length": 512
        }
    }

    # === 数据加载器 ===
    dataloader = AudioDataloader(
        config=config,
        data_root_dir="./data/Test_full",  # 所有 .pt 数据
        batch_size=BATCH_SIZE,
        val_split=0.15,
        shuffle=True,
        augment=False,
        featuretype=FeatureTypes.cqt,

    )
    train_loader, val_loader, test_loader = dataloader.get_loaders()

    # === 初始化模型 ===
    model = BiLSTMChordRecognizer(input_dim=168,
                                   hidden_dim=HIDDEN_SIZE,
                                   num_classes=NUM_CLASSES,
                                   num_layers=NUM_LAYERS).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)  # 忽略 padding 位

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        total_acc = 0
        total_steps = 0

        for X, y in train_loader:
            X = X.to(DEVICE)      # (B, T, F)
            y = y.to(DEVICE)      # (B, T)

            logits = model(X)     # (B, T, C)
            loss = criterion(logits.view(-1, NUM_CLASSES), y.view(-1))
            acc = accuracy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += acc
            total_steps += 1

        avg_loss = total_loss / total_steps
        avg_acc = total_acc / total_steps
        print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")

        # === 验证 ===
        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_acc = 0
            steps = 0
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                logits = model(X)
                loss = criterion(logits.view(-1, NUM_CLASSES), y.view(-1))
                acc = accuracy(logits, y)
                val_loss += loss.item()
                val_acc += acc
                steps += 1
            val_loss /= steps
            val_acc /= steps

            print(f"           Val  Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

            # === 模型保存 ===
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "best_bilstm_model.pt")
                print(f"           ✅ Saved best model (acc={val_acc:.4f})")

    print("✅ Training complete!")




if __name__ == "__main__":
    train()
