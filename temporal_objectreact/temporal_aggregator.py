"""
Temporal aggregation modules for WayObject Costmap sequences.

Given a sliding window of K per-frame goal embeddings e(c_{t-K+1..t}),
produce a single aggregated embedding to feed into the downstream
ObjectReact controller head.

Implements four trainable / training-free variants:
  - EMATemporalAggregator: training-free baseline, exponential moving average.
  - GRUTemporalAggregator: single-layer GRU over the embedding sequence.
  - TemporalCostmapAggregator: GRU wrapped with a learned confidence gate that
    down-weights anomalous frames *before* they enter the GRU.
  - ReliabilityGatedGRUTemporalAggregator: GRU wrapped with a learned
    reliability gate over current/history feature relations.

All aggregators expect a tensor of shape (B, K, D) where K is the window
length and D is the embedding dimension produced by the upstream
``GoalEncoder``.  They return a tensor of shape (B, D).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# EMA baseline (training-free)
# ---------------------------------------------------------------------------
class EMATemporalAggregator(nn.Module):
    """Exponentially-weighted moving average over the time dimension.

    Weights are normalised so that they sum to 1 along the K axis. The most
    recent frame receives weight ``1``, the previous one ``lam``, ``lam**2``,
    and so on.  When ``lam=1.0`` this reduces to a simple mean.
    """

    def __init__(self, lam: float = 0.7):
        super().__init__()
        if not (0.0 < lam <= 1.0):
            raise ValueError(f"lam must be in (0, 1], got {lam}")
        self.lam = lam

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, K, D) -> (B, D)
        K = x.shape[1]
        # weights[k] = lam ** (K - 1 - k); index K-1 (most recent) has weight 1.
        exps = torch.arange(K - 1, -1, -1, device=x.device, dtype=x.dtype)
        w = self.lam**exps
        w = w / w.sum()
        return (x * w.view(1, K, 1)).sum(dim=1)


# ---------------------------------------------------------------------------
# Plain GRU aggregator
# ---------------------------------------------------------------------------
class GRUTemporalAggregator(nn.Module):
    """Single-layer GRU that reads the K embeddings in temporal order and
    returns its last hidden state."""

    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.gru = nn.GRU(
            input_size=dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        if hidden_dim != dim:
            self.proj = nn.Linear(hidden_dim, dim)
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, K, D) -> (B, D)
        _, h_n = self.gru(x)  # h_n: (1, B, H)
        return self.proj(h_n.squeeze(0))


# ---------------------------------------------------------------------------
# Confidence gate
# ---------------------------------------------------------------------------
class ConfidenceGate(nn.Module):
    """Per-frame confidence gate.

    For each frame t in the window, compute a confidence scalar
    ``alpha_t = sigmoid(w * cos(e_t, mean_history) + b)`` where the running
    mean is an exponential moving average of past embeddings in the same
    window.  Anomalous frames (low cosine similarity to recent history)
    receive a small ``alpha_t`` and are convex-combined with the running mean
    before being passed to the downstream aggregator.

    The very first frame in the window is trusted unconditionally
    (``alpha = 1``) since there is no history yet.
    """

    def __init__(self, ema_lambda: float = 0.7):
        super().__init__()
        if not (0.0 < ema_lambda < 1.0):
            raise ValueError(f"ema_lambda must be in (0, 1), got {ema_lambda}")
        self.ema_lambda = ema_lambda
        # Two learnable scalars: w (slope) and b (bias).
        self.w = nn.Parameter(torch.tensor(5.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(
        self, x: torch.Tensor, return_alpha: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply the confidence gate to a window of embeddings.

        Args:
            x: (B, K, D) sequence of per-frame embeddings, in temporal order.
            return_alpha: if True, also return per-frame confidences (B, K).

        Returns:
            gated: (B, K, D) tensor where each frame has been mixed with the
                running mean according to its confidence.
            alpha: (B, K) confidence scalars, or None.
        """
        B, K, D = x.shape
        gated = []
        alphas = []
        # Initialise the running mean with the first frame; that frame gets
        # alpha = 1 by convention so it is passed through unchanged.
        mean = x[:, 0]  # (B, D)
        gated.append(x[:, 0])
        alphas.append(torch.ones(B, device=x.device, dtype=x.dtype))

        for t in range(1, K):
            e_t = x[:, t]  # (B, D)
            cos = F.cosine_similarity(e_t, mean, dim=-1).clamp(-1.0, 1.0)
            alpha = torch.sigmoid(self.w * cos + self.b)  # (B,)
            alphas.append(alpha)
            mixed = alpha.unsqueeze(-1) * e_t + (1.0 - alpha).unsqueeze(-1) * mean
            gated.append(mixed)
            # Update the running mean with the (raw) frame so the EMA tracks
            # incoming data even when the gate is mostly closed.
            mean = self.ema_lambda * mean + (1.0 - self.ema_lambda) * e_t

        gated_t = torch.stack(gated, dim=1)  # (B, K, D)
        alpha_t = torch.stack(alphas, dim=1) if return_alpha else None
        return gated_t, alpha_t


# ---------------------------------------------------------------------------
# Confidence-gated GRU aggregator (our full method)
# ---------------------------------------------------------------------------
class TemporalCostmapAggregator(nn.Module):
    """GRU temporal aggregator with a learned confidence gate.

    This is the full method described in Section III of the proposal.

    Args:
        dim: dimension of the per-frame goal embedding.
        hidden_dim: optional hidden size for the internal GRU.
        ema_lambda: EMA decay for the running mean used inside the gate.
        use_gate: if False, behaves like ``GRUTemporalAggregator``.  Useful
            for ablations.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        ema_lambda: float = 0.7,
        use_gate: bool = True,
    ):
        super().__init__()
        self.use_gate = use_gate
        if use_gate:
            self.gate = ConfidenceGate(ema_lambda=ema_lambda)
        else:
            self.gate = None
        self.gru = GRUTemporalAggregator(dim=dim, hidden_dim=hidden_dim)

    def forward(
        self, x: torch.Tensor, return_alpha: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        alpha = None
        if self.gate is not None:
            x, alpha = self.gate(x, return_alpha=return_alpha)
        out = self.gru(x)
        return out, alpha


# ---------------------------------------------------------------------------
# Reliability-gated GRU aggregator
# ---------------------------------------------------------------------------
class ReliabilityGate(nn.Module):
    """Per-frame learned reliability gate using current/history relations."""

    def __init__(
        self,
        dim: int,
        ema_lambda: float = 0.7,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        if not (0.0 < ema_lambda < 1.0):
            raise ValueError(f"ema_lambda must be in (0, 1), got {ema_lambda}")
        self.ema_lambda = ema_lambda
        hidden_dim = hidden_dim or max(32, dim // 8)
        self.scorer = nn.Sequential(
            nn.Linear(dim * 4 + 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, x: torch.Tensor, return_alpha: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, K, _D = x.shape
        gated = []
        alphas = []

        mean = x[:, 0]
        gated.append(x[:, 0])
        alphas.append(torch.ones(B, device=x.device, dtype=x.dtype))

        for t in range(1, K):
            e_t = x[:, t]
            diff = torch.abs(e_t - mean)
            prod = e_t * mean
            cos = F.cosine_similarity(e_t, mean, dim=-1).clamp(-1.0, 1.0)
            delta = diff.norm(dim=-1) / (mean.norm(dim=-1) + e_t.norm(dim=-1) + 1e-6)
            gate_input = torch.cat(
                [e_t, mean, diff, prod, cos.unsqueeze(-1), delta.unsqueeze(-1)],
                dim=-1,
            )
            alpha = torch.sigmoid(self.scorer(gate_input).squeeze(-1))
            alphas.append(alpha)

            mixed = alpha.unsqueeze(-1) * e_t + (1.0 - alpha).unsqueeze(-1) * mean
            gated.append(mixed)
            mean = self.ema_lambda * mean + (1.0 - self.ema_lambda) * e_t

        gated_t = torch.stack(gated, dim=1)
        alpha_t = torch.stack(alphas, dim=1) if return_alpha else None
        return gated_t, alpha_t


class ReliabilityGatedGRUTemporalAggregator(nn.Module):
    """GRU aggregator with a learned reliability gate before recurrent update."""

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        ema_lambda: float = 0.7,
        gate_hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.gate = ReliabilityGate(
            dim=dim,
            ema_lambda=ema_lambda,
            hidden_dim=gate_hidden_dim,
        )
        self.gru = GRUTemporalAggregator(dim=dim, hidden_dim=hidden_dim)

    def forward(
        self, x: torch.Tensor, return_alpha: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x, alpha = self.gate(x, return_alpha=return_alpha)
        out = self.gru(x)
        return out, alpha


# ---------------------------------------------------------------------------
# Small sanity test (only executed when running the file directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, K, D = 4, 5, 1024
    x = torch.randn(B, K, D)

    for name, mod in [
        ("EMA", EMATemporalAggregator(lam=0.7)),
        ("GRU", GRUTemporalAggregator(dim=D)),
        ("GatedGRU", TemporalCostmapAggregator(dim=D)),
        ("RelGated", ReliabilityGatedGRUTemporalAggregator(dim=D)),
    ]:
        out = mod(x)
        if isinstance(out, tuple):
            out = out[0]
        print(
            f"{name:9s}  in={tuple(x.shape)}  out={tuple(out.shape)}  "
            f"params={sum(p.numel() for p in mod.parameters())}"
        )
