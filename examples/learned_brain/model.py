#!/usr/bin/env python3
"""Learned Brain Model — GRU temporal encoder over conversation embeddings.

Architecture:
  Frozen SentenceBERT (384-dim) → Linear(384, 128) → 2-layer GRU (hidden=32)
  → emotion classifier (32 labels) + next-embedding predictor (384-dim)

Loss:
  L = CE(emotion_logits, label) + 0.3 * MSE(predicted_next_emb, actual_next_emb)
"""

import torch
import torch.nn as nn


class LearnedBrainModel(nn.Module):
    """Temporal encoder that learns latent emotional state from conversation flow."""

    def __init__(
        self,
        input_dim: int = 384,       # SentenceBERT embedding dim
        proj_dim: int = 128,        # projection before GRU
        hidden_dim: int = 32,       # GRU hidden = latent state dim
        n_emotions: int = 32,       # EmpatheticDialogues labels
        n_gru_layers: int = 2,
        dropout: float = 0.1,
        aux_weight: float = 0.3,    # weight for next-embedding prediction loss
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_gru_layers = n_gru_layers
        self.aux_weight = aux_weight

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Temporal encoder
        self.gru = nn.GRU(
            input_size=proj_dim,
            hidden_size=hidden_dim,
            num_layers=n_gru_layers,
            batch_first=True,
            dropout=dropout if n_gru_layers > 1 else 0.0,
        )

        # Emotion classifier head (from final hidden state)
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_emotions),
        )

        # Next-embedding predictor head (auxiliary task)
        self.next_emb_head = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, input_dim),
        )

        # Scenario encoder for cold-start z_0 initialization
        self.z0_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        )

    def init_hidden(self, situation_emb: torch.Tensor) -> torch.Tensor:
        """Initialize GRU hidden state from scenario/situation embedding.

        Args:
            situation_emb: (batch, input_dim) situation description embedding

        Returns:
            h_0: (n_layers, batch, hidden_dim) initial hidden state
        """
        z0 = self.z0_proj(situation_emb)  # (batch, hidden_dim)
        # Expand to all GRU layers
        return z0.unsqueeze(0).expand(self.n_gru_layers, -1, -1).contiguous()

    def forward(
        self,
        utterance_embs: torch.Tensor,
        situation_emb: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> dict:
        """Forward pass over a conversation.

        Args:
            utterance_embs: (batch, max_seq_len, input_dim) padded utterance embeddings
            situation_emb: (batch, input_dim) situation/scenario embedding
            lengths: (batch,) actual sequence lengths (for packing)

        Returns:
            dict with: emotion_logits, next_emb_pred, latent_states, final_hidden
        """
        batch_size, max_len, _ = utterance_embs.shape

        # Project inputs
        projected = self.input_proj(utterance_embs)  # (batch, max_len, proj_dim)

        # Initialize hidden from situation
        h_0 = self.init_hidden(situation_emb)

        # Pack if lengths provided
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                projected, lengths.cpu(), batch_first=True, enforce_sorted=False,
            )
            gru_out, h_n = self.gru(packed, h_0)
            gru_out, _ = nn.utils.rnn.pad_packed_sequence(
                gru_out, batch_first=True, total_length=max_len,
            )
        else:
            gru_out, h_n = self.gru(projected, h_0)

        # Get final hidden state per sequence (use lengths to index)
        if lengths is not None:
            # Gather the hidden state at the last valid timestep
            idx = (lengths - 1).long().unsqueeze(1).unsqueeze(2).expand(-1, 1, self.hidden_dim)
            final_hidden = gru_out.gather(1, idx).squeeze(1)  # (batch, hidden_dim)
        else:
            final_hidden = gru_out[:, -1, :]

        # Emotion classification from final hidden
        emotion_logits = self.emotion_head(final_hidden)  # (batch, n_emotions)

        # Next-embedding prediction from each timestep
        next_emb_pred = self.next_emb_head(gru_out)  # (batch, max_len, input_dim)

        return {
            "emotion_logits": emotion_logits,
            "next_emb_pred": next_emb_pred,
            "latent_states": gru_out,       # (batch, max_len, hidden_dim)
            "final_hidden": final_hidden,    # (batch, hidden_dim)
        }

    def compute_loss(
        self,
        outputs: dict,
        emotion_label: torch.Tensor,
        utterance_embs: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> dict:
        """Compute combined loss.

        Args:
            outputs: from forward()
            emotion_label: (batch,) integer emotion labels
            utterance_embs: (batch, max_len, input_dim) original embeddings
            lengths: (batch,) sequence lengths

        Returns:
            dict with: total_loss, ce_loss, aux_loss
        """
        # Cross-entropy on emotion prediction
        ce_loss = nn.functional.cross_entropy(outputs["emotion_logits"], emotion_label)

        # Auxiliary: predict next utterance embedding from current hidden state
        # For timestep t, predict embedding at t+1
        pred = outputs["next_emb_pred"][:, :-1, :]   # (batch, max_len-1, input_dim)
        target = utterance_embs[:, 1:, :]              # (batch, max_len-1, input_dim)

        if lengths is not None:
            # Mask out padding
            mask = torch.arange(pred.size(1), device=pred.device).unsqueeze(0) < (lengths - 1).unsqueeze(1)
            mask = mask.unsqueeze(2).float()
            aux_loss = ((pred - target) ** 2 * mask).sum() / mask.sum().clamp(min=1) / pred.size(2)
        else:
            aux_loss = nn.functional.mse_loss(pred, target)

        total_loss = ce_loss + self.aux_weight * aux_loss

        return {
            "total_loss": total_loss,
            "ce_loss": ce_loss,
            "aux_loss": aux_loss,
        }

    def get_latent_state(
        self,
        utterance_embs: torch.Tensor,
        situation_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Get the current latent state for inference (no batching needed).

        Args:
            utterance_embs: (seq_len, input_dim) unbatched
            situation_emb: (input_dim,) unbatched

        Returns:
            latent: (hidden_dim,) current latent emotional state
        """
        with torch.no_grad():
            # Add batch dim
            utts = utterance_embs.unsqueeze(0)
            sit = situation_emb.unsqueeze(0)
            outputs = self.forward(utts, sit)
            return outputs["final_hidden"].squeeze(0)

    def predict_emotion(
        self,
        utterance_embs: torch.Tensor,
        situation_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict emotion distribution for inference.

        Returns:
            (emotion_probs, latent_state) — both unbatched
        """
        with torch.no_grad():
            utts = utterance_embs.unsqueeze(0)
            sit = situation_emb.unsqueeze(0)
            outputs = self.forward(utts, sit)
            probs = torch.softmax(outputs["emotion_logits"], dim=-1).squeeze(0)
            latent = outputs["final_hidden"].squeeze(0)
            return probs, latent
