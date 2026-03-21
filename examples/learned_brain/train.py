#!/usr/bin/env python3
"""Train the Learned Brain model on EmpatheticDialogues.

Usage:
    python -m learned_brain.train          # default settings
    python -m learned_brain.train --epochs 30 --batch-size 64
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .model import LearnedBrainModel
from .data_prep import EMOTION_TO_IDX, DATA_DIR

MODEL_DIR = Path(__file__).parent / "checkpoints"


class ConversationDataset(Dataset):
    """Dataset of conversations with pre-computed embeddings."""

    def __init__(self, split: str = "train"):
        conv_path = DATA_DIR / f"{split}.json"
        emb_path = DATA_DIR / f"{split}_embeddings.pkl"

        with open(conv_path) as f:
            self.conversations = json.load(f)
        with open(emb_path, "rb") as f:
            self.embeddings = pickle.load(f)

        assert len(self.conversations) == len(self.embeddings)

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        conv = self.conversations[idx]
        embs = self.embeddings[idx]

        # Situation embedding
        situation_emb = np.array(embs["situation_emb"], dtype=np.float32)

        # Utterance embeddings
        utt_embs = np.array(embs["utterance_embs"], dtype=np.float32)

        # Emotion label
        emotion = conv["emotion"]
        emotion_idx = EMOTION_TO_IDX.get(emotion, 0)

        return {
            "situation_emb": torch.from_numpy(situation_emb),
            "utterance_embs": torch.from_numpy(utt_embs),
            "emotion_idx": emotion_idx,
            "n_utterances": len(utt_embs),
        }


def collate_fn(batch):
    """Pad utterance sequences to max length in batch."""
    max_len = max(item["n_utterances"] for item in batch)
    emb_dim = batch[0]["utterance_embs"].shape[1]

    padded_utts = torch.zeros(len(batch), max_len, emb_dim)
    lengths = torch.zeros(len(batch), dtype=torch.long)
    situation_embs = torch.stack([item["situation_emb"] for item in batch])
    emotion_idxs = torch.tensor([item["emotion_idx"] for item in batch], dtype=torch.long)

    for i, item in enumerate(batch):
        n = item["n_utterances"]
        padded_utts[i, :n] = item["utterance_embs"]
        lengths[i] = n

    return {
        "utterance_embs": padded_utts,
        "situation_emb": situation_embs,
        "emotion_idx": emotion_idxs,
        "lengths": lengths,
    }


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_ce = 0
    total_aux = 0
    correct = 0
    total = 0

    for batch in loader:
        utts = batch["utterance_embs"].to(device)
        sits = batch["situation_emb"].to(device)
        labels = batch["emotion_idx"].to(device)
        lengths = batch["lengths"].to(device)

        outputs = model(utts, sits, lengths)
        losses = model.compute_loss(outputs, labels, utts, lengths)

        optimizer.zero_grad()
        losses["total_loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += losses["total_loss"].item() * len(labels)
        total_ce += losses["ce_loss"].item() * len(labels)
        total_aux += losses["aux_loss"].item() * len(labels)

        preds = outputs["emotion_logits"].argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += len(labels)

    n = total
    return {
        "loss": total_loss / n,
        "ce_loss": total_ce / n,
        "aux_loss": total_aux / n,
        "accuracy": correct / n,
    }


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0
    total_ce = 0
    correct = 0
    total = 0
    top5_correct = 0

    for batch in loader:
        utts = batch["utterance_embs"].to(device)
        sits = batch["situation_emb"].to(device)
        labels = batch["emotion_idx"].to(device)
        lengths = batch["lengths"].to(device)

        outputs = model(utts, sits, lengths)
        losses = model.compute_loss(outputs, labels, utts, lengths)

        total_loss += losses["total_loss"].item() * len(labels)
        total_ce += losses["ce_loss"].item() * len(labels)

        preds = outputs["emotion_logits"].argmax(dim=-1)
        correct += (preds == labels).sum().item()

        # Top-5 accuracy
        top5 = outputs["emotion_logits"].topk(5, dim=-1).indices
        top5_correct += sum(labels[i] in top5[i] for i in range(len(labels)))

        total += len(labels)

    n = total
    return {
        "loss": total_loss / n,
        "ce_loss": total_ce / n,
        "accuracy": correct / n,
        "top5_accuracy": top5_correct / n,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Learned Brain")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--aux-weight", type=float, default=0.3)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load data
    print("Loading datasets...")
    train_ds = ConversationDataset("train")
    val_ds = ConversationDataset("validation")
    print(f"Train: {len(train_ds)} conversations, Val: {len(val_ds)} conversations")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # Model
    model = LearnedBrainModel(
        input_dim=384,
        proj_dim=args.proj_dim,
        hidden_dim=args.hidden_dim,
        n_emotions=32,
        aux_weight=args.aux_weight,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_acc = 0.0
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining for {args.epochs} epochs...")
    print(f"{'Epoch':>5} {'Train Loss':>10} {'Train Acc':>9} {'Val Loss':>10} {'Val Acc':>8} {'Val Top5':>9} {'Time':>6}")
    print("-" * 65)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_epoch(model, train_loader, optimizer, device)
        val_metrics = eval_epoch(model, val_loader, device)
        scheduler.step()

        dt = time.time() - t0

        print(f"{epoch:>5d} {train_metrics['loss']:>10.4f} {train_metrics['accuracy']:>8.1%} "
              f"{val_metrics['loss']:>10.4f} {val_metrics['accuracy']:>7.1%} {val_metrics['top5_accuracy']:>8.1%} "
              f"{dt:>5.1f}s")

        # Save best model
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_accuracy": best_val_acc,
                "val_top5_accuracy": val_metrics["top5_accuracy"],
                "args": vars(args),
            }, MODEL_DIR / "best_model.pt")

    # Save final model too
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs,
        "val_accuracy": val_metrics["accuracy"],
        "args": vars(args),
    }, MODEL_DIR / "final_model.pt")

    print(f"\nBest validation accuracy: {best_val_acc:.1%}")
    print(f"Models saved to {MODEL_DIR}")

    # Quick test set evaluation
    print("\nEvaluating on test set...")
    test_ds = ConversationDataset("test")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_metrics = eval_epoch(model, test_loader, device)
    print(f"Test accuracy: {test_metrics['accuracy']:.1%} (top-5: {test_metrics['top5_accuracy']:.1%})")


if __name__ == "__main__":
    main()
