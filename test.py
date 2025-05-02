import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

from data.audio_dataloader import AudioDataloader
from model import LSTMClassifier, GRUClassifier
from metrics import (
    framewise_accuracy_full,
    weighted_chord_symbol_recall_full,
    FocalLoss
)

NUM_CLASSES = 25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def classwise_accuracy_full(model, loader, num_classes):
    total_per_class = [0] * num_classes
    correct_per_class = [0] * num_classes
    model.eval()
    with torch.no_grad():
        for X, y, lengths in loader:
            X, y, lengths = X.to(DEVICE), y.to(DEVICE), lengths.to(DEVICE)
            logits = model(X, lengths)
            preds = torch.argmax(logits, dim=-1)
            mask = (y != -100)
            for i in range(num_classes):
                class_mask = (y == i) & mask
                total_per_class[i] += class_mask.sum().item()
                correct_per_class[i] += ((preds == y) & class_mask).sum().item()
    acc_per_class = [c / t if t > 0 else None for c, t in zip(correct_per_class, total_per_class)]
    return acc_per_class, correct_per_class, total_per_class


def print_classwise_accuracy(acc_per_class, correct_per_class, total_per_class):
    print("\nClass-wise Accuracy:")
    print("Class |     Acc |  Correct / Total")
    print("-----------------------------------")
    for i, (acc, c, t) in enumerate(zip(acc_per_class, correct_per_class, total_per_class)):
        acc_str = f"{acc:.2%}" if acc is not None else "N/A"
        print(f"{i:5} | {acc_str:>7} | {c:7} / {t:7}")


def get_model(model_type):
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
        raise ValueError("Unsupported model type: use 'lstm' or 'gru'")


def evaluate(model_type="lstm", split="test"):
    print(f"📦 Evaluating {model_type.upper()} model on {split.upper()} set...")

    config = {
        "mp3": {"song_hz": 22050, "inst_len": 10.0, "skip_interval": 5.0},
        "feature": {"n_bins": 168, "bins_per_octave": 24, "hop_length": 512}
    }

    dataloader = AudioDataloader(
        config=config,
        data_root_dir="./data",
        batch_size=32,
        shuffle=False,
        augment=False,
        featuretype="cqt",
        num_workers=4
    )

    _, _, test_loader, test_jay_loader = dataloader.get_loaders()
    loader = test_loader if split == "test" else test_jay_loader

    model = get_model(model_type).to(DEVICE)
    model_path = f"best_{model_type}_model.pt"

    print(f"📥 Loading weights from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    total_loss, total_steps = 0, 0

    with torch.no_grad():
        for X, y, lengths in tqdm(loader):
            X, y, lengths = X.to(DEVICE), y.to(DEVICE), lengths.to(DEVICE)
            logits = model(X, lengths)
            loss = criterion(logits.reshape(-1, NUM_CLASSES), y.reshape(-1))
            total_loss += loss.item()
            total_steps += 1

    test_loss = total_loss / total_steps
    frame_acc = framewise_accuracy_full(model, loader, DEVICE)
    wcsr = weighted_chord_symbol_recall_full(model, loader, DEVICE)
    acc_per_class, correct, total = classwise_accuracy_full(model, loader, NUM_CLASSES)

    print(f"\n📊 Test Loss: {test_loss:.4f}")
    print(f"📊 Frame Accuracy: {frame_acc:.4f}")
    print(f"📊 WCSR: {wcsr:.4f}")
    print_classwise_accuracy(acc_per_class, correct, total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "gru"], help="Model type to evaluate")
    args = parser.parse_args()

    evaluate(model_type=args.model, split="test")
    evaluate(model_type=args.model, split="jay")
