"""
Cross-Modal Attention modules for multimodal time-series forecasting.

Architecture: dual LSTM encoders (numeric + news) → directional cross-attention
(numeric queries news) → scalar modality gate → additive fusion → quantile output.

Ablation toggles:
  use_cross_attention=False, use_gate=False  →  direct additive fusion
  use_cross_attention=True,  use_gate=False  →  attention, no gate
  use_cross_attention=True,  use_gate=True   →  full model
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl


# ──────────────────────────────────────────────────────────────
# Standalone Quantile (Pinball) Loss
# ──────────────────────────────────────────────────────────────
class QuantileLoss(nn.Module):
    """Pinball loss for quantile regression."""

    def __init__(self, quantiles):
        super().__init__()
        self.register_buffer(
            "quantiles", torch.tensor(quantiles, dtype=torch.float32)
        )

    def forward(self, preds, target):
        """
        preds:  [B, n_q]
        target: [B]
        """
        errors = target.unsqueeze(-1) - preds          # [B, n_q]
        q = self.quantiles.unsqueeze(0)                 # [1, n_q]
        loss = torch.max(q * errors, (q - 1.0) * errors)  # [B, n_q]
        return loss.mean()


# ──────────────────────────────────────────────────────────────
# Sub-modules
# ──────────────────────────────────────────────────────────────
class InputProjection(nn.Module):
    """Project raw features into shared hidden dimension."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, input_dim] → [B, T, hidden_dim]
        return self.drop(self.norm(self.linear(x)))


class ModalityEncoder(nn.Module):
    """LSTM encoder for a single modality."""

    def __init__(self, hidden_dim: int, n_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: [B, T, d] → [B, T, d]
        out, _ = self.lstm(x)
        return self.norm(out)


class CrossModalAttention(nn.Module):
    """Directional multi-head cross-attention: numeric queries news."""

    def __init__(self, hidden_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, query, key_value):
        """
        query:     H_num  [B, T, d]
        key_value: H_news [B, T, d]

        Returns:
            out:          [B, T, d]   (residual + layernorm applied)
            attn_weights: [B, n_heads, T, T]  (per-head, for interpretability)
        """
        attn_out, attn_weights = self.mha(
            query, key_value, key_value,
            need_weights=True,
            average_attn_weights=False,   # keep per-head weights
        )
        out = self.norm(query + self.drop(attn_out))  # residual connection
        return out, attn_weights


class ModalityGate(nn.Module):
    """Scalar sigmoid gate controlling attended news contribution.

    Gate input: [H_num ; attn_out]  (not raw H_news).
    Output: g ∈ (0,1) per timestep, shape [B, T, 1].
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim * 2, 1)
        # Init bias=0 → sigmoid(0)=0.5, start uncertain
        nn.init.zeros_(self.gate_proj.bias)

    def forward(self, h_numeric, attn_out):
        # h_numeric: [B, T, d], attn_out: [B, T, d]
        combined = torch.cat([h_numeric, attn_out], dim=-1)  # [B, T, 2d]
        return torch.sigmoid(self.gate_proj(combined))        # [B, T, 1]


class QuantileOutputHead(nn.Module):
    """Map final hidden state to quantile predictions."""

    def __init__(self, hidden_dim: int, n_quantiles: int = 5):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, n_quantiles)

    def forward(self, h):
        # h: [B, d] → [B, n_quantiles]
        return self.linear(h)


# ──────────────────────────────────────────────────────────────
# Top-level LightningModule
# ──────────────────────────────────────────────────────────────
class CrossModalAttentionForecaster(pl.LightningModule):
    """
    Dual-encoder cross-modal attention forecaster with ablation toggles.

    Ablation configurations:
      use_cross_attention=False, use_gate=False  → additive fusion only
      use_cross_attention=True,  use_gate=False  → attention, no gate
      use_cross_attention=True,  use_gate=True   → full model
    """

    def __init__(
        self,
        n_numeric_features: int,
        n_news_features: int,
        numeric_offset: int = 0,
        hidden_dim: int = 64,
        n_heads: int = 4,
        n_lstm_layers_numeric: int = 2,
        n_lstm_layers_news: int = 1,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        quantiles: tuple = (0.05, 0.25, 0.5, 0.75, 0.95),
        gradient_clip_val: float = 0.5,
        gate_lambda: float = 0.0,
        use_cross_attention: bool = True,
        use_gate: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ── Encoders ──
        self.numeric_proj = InputProjection(n_numeric_features, hidden_dim, dropout)
        self.news_proj = InputProjection(n_news_features, hidden_dim, dropout)
        self.numeric_encoder = ModalityEncoder(hidden_dim, n_lstm_layers_numeric, dropout)
        self.news_encoder = ModalityEncoder(hidden_dim, n_lstm_layers_news, dropout)

        # ── Cross-attention (optional) ──
        if use_cross_attention:
            self.cross_attn = CrossModalAttention(hidden_dim, n_heads, dropout)
        else:
            self.cross_attn = None

        # ── Modality gate (optional) ──
        if use_gate and use_cross_attention:
            self.gate = ModalityGate(hidden_dim)
        else:
            self.gate = None

        # ── Fusion projection for ablation #3 (no attention) ──
        if not use_cross_attention:
            self.news_fusion_proj = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.news_fusion_proj = None

        # ── Fusion + output ──
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.output_head = QuantileOutputHead(hidden_dim, len(quantiles))

        # ── Loss ──
        self.loss_fn = QuantileLoss(list(quantiles))

        # ── Interpretability storage (populated during forward) ──
        self._last_attn_weights = None    # [B, n_heads, T, T]
        self._last_gate_values = None     # [B, T, 1]

        # ── For gate stats accumulation during validation ──
        self._val_gate_sum = 0.0
        self._val_gate_count = 0
        self._val_gate_sparse_count = 0

    # ──────────────────────────────────────────────────────────
    # Feature splitting
    # ──────────────────────────────────────────────────────────
    def _split_features(self, x):
        """Split encoder_cont into numeric and news sub-tensors.

        encoder_cont contains internal TFT columns (encoder_length,
        price_center, price_scale, time_idx, relative_time_idx) before
        the user features.  ``numeric_offset`` skips past them.
        """
        encoder_cont = x["encoder_cont"]  # [B, T, N_total]
        off = self.hparams.numeric_offset
        n_num = self.hparams.n_numeric_features
        n_news = self.hparams.n_news_features
        x_numeric = encoder_cont[:, :, off:off + n_num]
        x_news = encoder_cont[:, :, off + n_num:off + n_num + n_news]
        return x_numeric, x_news

    # ──────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────
    def forward(self, x_numeric, x_news):
        """
        x_numeric: [B, T, n_num]
        x_news:    [B, T, n_news]
        Returns:   [B, n_q]
        """
        # 1. Project
        h_num = self.numeric_proj(x_numeric)     # [B, T, d]
        h_news = self.news_proj(x_news)          # [B, T, d]

        # 2. Encode
        h_num_enc = self.numeric_encoder(h_num)   # [B, T, d]
        h_news_enc = self.news_encoder(h_news)    # [B, T, d]

        # 3. Fusion (depends on ablation config)
        if self.cross_attn is not None:
            # Cross-modal attention: numeric queries news
            attn_out, attn_weights = self.cross_attn(
                query=h_num_enc, key_value=h_news_enc
            )
            self._last_attn_weights = attn_weights.detach()

            if self.gate is not None:
                # Gated fusion
                g = self.gate(h_num_enc, attn_out)    # [B, T, 1]
                self._last_gate_values = g.detach()
                news_contribution = g * attn_out       # [B, T, d]
            else:
                # Attention without gate
                self._last_gate_values = None
                news_contribution = attn_out           # [B, T, d]

            fused = self.fusion_norm(h_num_enc + news_contribution)
        else:
            # No attention: direct additive fusion with projection
            self._last_attn_weights = None
            self._last_gate_values = None
            news_proj = self.news_fusion_proj(h_news_enc)  # [B, T, d]
            fused = self.fusion_norm(h_num_enc + news_proj)

        # 4. Last timestep → quantile output
        h_final = fused[:, -1, :]                  # [B, d]
        return self.output_head(h_final)           # [B, n_q]

    # ──────────────────────────────────────────────────────────
    # Training / validation steps
    # ──────────────────────────────────────────────────────────
    def training_step(self, batch, batch_idx):
        x, y = batch
        x_numeric, x_news = self._split_features(x)
        y_true = y[0] if isinstance(y, (tuple, list)) else y
        y_true = y_true.reshape(-1)  # [B]

        preds = self(x_numeric, x_news)  # [B, n_q]
        loss = self.loss_fn(preds, y_true)

        # Gate sparsity regularisation
        if self._last_gate_values is not None and self.hparams.gate_lambda > 0:
            gate_l1 = self._last_gate_values.mean()
            loss = loss + self.hparams.gate_lambda * gate_l1

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_numeric, x_news = self._split_features(x)
        y_true = y[0] if isinstance(y, (tuple, list)) else y
        y_true = y_true.reshape(-1)

        preds = self(x_numeric, x_news)
        loss = self.loss_fn(preds, y_true)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Accumulate gate statistics
        if self._last_gate_values is not None:
            g = self._last_gate_values
            self._val_gate_sum += g.sum().item()
            self._val_gate_count += g.numel()
            self._val_gate_sparse_count += (g < 0.1).sum().item()

        return loss

    def on_validation_epoch_end(self):
        if self._val_gate_count > 0:
            mean_gate = self._val_gate_sum / self._val_gate_count
            sparsity = self._val_gate_sparse_count / self._val_gate_count
            self.log("gate_mean", mean_gate, prog_bar=True)
            self.log("gate_sparsity", sparsity)
        # Reset accumulators
        self._val_gate_sum = 0.0
        self._val_gate_count = 0
        self._val_gate_sparse_count = 0

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
