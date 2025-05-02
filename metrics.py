import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, ignore_index=-100, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        N, C = input.size()
        valid_mask = (target != self.ignore_index)
        valid_targets = target[valid_mask]

        if valid_targets.numel() == 0:
            return torch.tensor(0.0, device=input.device, requires_grad=True)

        logpt = nn.functional.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        safe_target = target.clone()
        safe_target[~valid_mask] = 0
        target_one_hot = torch.zeros_like(input).scatter_(1, safe_target.unsqueeze(1), 1)

        focal_term = (1 - pt) ** self.gamma
        loss = -focal_term * logpt * target_one_hot
        if self.weight is not None:
            loss *= self.weight.view(1, -1)

        loss = loss.sum(dim=1)[valid_mask]

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def framewise_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=-1)
    mask = (labels != -100)
    correct = (preds == labels) & mask
    return correct.sum().item() / mask.sum().item()


def framewise_accuracy_full(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y, lengths in loader:
            X, y, lengths = X.to(device), y.to(device), lengths.to(device)
            logits = model(X, lengths)
            preds = torch.argmax(logits, dim=-1)
            mask = (y != -100)
            correct += ((preds == y) & mask).sum().item()
            total += mask.sum().item()
    return correct / total if total > 0 else 0.0


def weighted_chord_symbol_recall_full(model, loader, device, hop_length=512, sr=22050):
    correct_frames = 0
    total_frames = 0
    model.eval()
    with torch.no_grad():
        for X, y, lengths in loader:
            X, y, lengths = X.to(device), y.to(device), lengths.to(device)
            logits = model(X, lengths)
            preds = torch.argmax(logits, dim=-1)
            mask = (y != -100)
            correct_frames += (preds[mask] == y[mask]).sum().item()
            total_frames += mask.sum().item()
    return correct_frames / total_frames if total_frames > 0 else 0.0
