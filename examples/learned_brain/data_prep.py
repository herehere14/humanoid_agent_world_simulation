#!/usr/bin/env python3
"""Download and prepare EmpatheticDialogues for training the learned brain.

Produces:
  - Conversations grouped by conv_id with emotion labels
  - Pre-computed SentenceBERT embeddings for each utterance
  - Train/val/test splits (already in the dataset)

Downloads raw CSV files directly since datasets>=4.x dropped script support
for the facebook/empathetic_dialogues repo.
"""

import csv
import io
import json
import os
import pickle
import urllib.request
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent / "data"

# The 32 emotion labels in EmpatheticDialogues
EMOTION_LABELS = [
    "afraid", "angry", "annoyed", "anticipating", "anxious",
    "apprehensive", "ashamed", "caring", "confident", "content",
    "devastated", "disappointed", "disgusted", "embarrassed", "excited",
    "faithful", "furious", "grateful", "guilty", "hopeful",
    "impressed", "jealous", "joyful", "lonely", "nostalgic",
    "prepared", "proud", "sad", "sentimental", "surprised",
    "terrified", "trusting",
]
EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTION_LABELS)}

# Raw CSV URLs (same source the HF dataset script uses)
_BASE_URL = "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz"


def _download_and_extract():
    """Download the tar.gz and extract CSV files."""
    import tarfile

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = DATA_DIR / "empatheticdialogues.tar.gz"

    # Check if already extracted
    if (DATA_DIR / "empatheticdialogues" / "train.csv").exists():
        print("Raw data already downloaded.")
        return

    print(f"Downloading EmpatheticDialogues from {_BASE_URL}...")
    urllib.request.urlretrieve(_BASE_URL, tar_path)
    print(f"Downloaded to {tar_path} ({tar_path.stat().st_size / 1e6:.1f} MB)")

    print("Extracting...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(DATA_DIR)
    print("Extracted.")

    # Clean up tar
    tar_path.unlink()


def _parse_csv(csv_path: Path) -> list[dict]:
    """Parse an EmpatheticDialogues CSV into rows."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def download_and_process():
    """Download EmpatheticDialogues and group into conversations."""
    _download_and_extract()

    raw_dir = DATA_DIR / "empatheticdialogues"
    split_map = {
        "train": raw_dir / "train.csv",
        "validation": raw_dir / "valid.csv",
        "test": raw_dir / "test.csv",
    }

    conversations = {}
    for split, csv_path in split_map.items():
        print(f"Processing {split} from {csv_path}...")
        rows = _parse_csv(csv_path)

        convs = {}
        for row in rows:
            cid = row["conv_id"]
            if cid not in convs:
                convs[cid] = {
                    "conv_id": cid,
                    "emotion": row["context"],  # emotion label
                    "situation": row["prompt"].replace("_comma_", ","),
                    "utterances": [],
                }
            convs[cid]["utterances"].append({
                "speaker": int(row["speaker_idx"]),
                "text": row["utterance"].replace("_comma_", ","),
                "idx": int(row["utterance_idx"]),
            })

        # Sort utterances by idx within each conversation
        for conv in convs.values():
            conv["utterances"].sort(key=lambda u: u["idx"])

        conversations[split] = list(convs.values())

    print(f"Conversations: train={len(conversations['train'])}, "
          f"val={len(conversations['validation'])}, "
          f"test={len(conversations['test'])}")

    # Filter out emotions not in our label set and short conversations
    for split in conversations:
        conversations[split] = [
            c for c in conversations[split]
            if c["emotion"] in EMOTION_TO_IDX and len(c["utterances"]) >= 2
        ]

    print(f"After filtering: train={len(conversations['train'])}, "
          f"val={len(conversations['validation'])}, "
          f"test={len(conversations['test'])}")

    # Save processed conversations
    for split, convs in conversations.items():
        path = DATA_DIR / f"{split}.json"
        with open(path, "w") as f:
            json.dump(convs, f)
        print(f"Saved {len(convs)} conversations to {path}")

    return conversations


def compute_embeddings(conversations: dict):
    """Pre-compute SentenceBERT embeddings for all utterances."""
    print("Loading SentenceBERT model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    for split, convs in conversations.items():
        print(f"Computing embeddings for {split} ({len(convs)} conversations)...")
        all_texts = []
        index_map = []  # (conv_idx, utt_idx) for each text

        for ci, conv in enumerate(convs):
            # Also embed the situation description as turn -1
            all_texts.append(conv["situation"])
            index_map.append((ci, -1))
            for ui, utt in enumerate(conv["utterances"]):
                all_texts.append(utt["text"])
                index_map.append((ci, ui))

        # Batch encode
        embeddings = model.encode(all_texts, batch_size=256, show_progress_bar=True)

        # Organize back into conversations
        conv_embeddings = [{"situation_emb": None, "utterance_embs": []} for _ in convs]
        for (ci, ui), emb in zip(index_map, embeddings):
            if ui == -1:
                conv_embeddings[ci]["situation_emb"] = emb
            else:
                conv_embeddings[ci]["utterance_embs"].append(emb)

        # Save as numpy arrays packed into a pickle
        path = DATA_DIR / f"{split}_embeddings.pkl"
        with open(path, "wb") as f:
            pickle.dump(conv_embeddings, f)
        print(f"Saved embeddings to {path}")


def load_processed_data(split: str = "train"):
    """Load pre-processed conversations and embeddings."""
    conv_path = DATA_DIR / f"{split}.json"
    emb_path = DATA_DIR / f"{split}_embeddings.pkl"

    with open(conv_path) as f:
        conversations = json.load(f)
    with open(emb_path, "rb") as f:
        embeddings = pickle.load(f)

    return conversations, embeddings


if __name__ == "__main__":
    conversations = download_and_process()
    compute_embeddings(conversations)
    print("\nDone! Data ready for training.")
