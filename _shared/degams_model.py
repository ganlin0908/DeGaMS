"""DeGaMS: Decomposed Gated Multi-Scale multimodal fusion.

Turn-aligned T/A/V encoders, shared||specific decomposition, presence-aware
modality gating, multi-scale attention fusion, and PHQ-structured heads
(soft item KL + total regression + supervised contrastive / SupCon).

Official method name for this codebase / paper: **DeGaMS** (DeGaMS-Net).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Must match daic_degams_dataset
VIS_KIND_MISSING = 0
VIS_KIND_CLNF = 1
VIS_KIND_OPENFACE = 2
OPENFACE_CORE_DIM = 29


@dataclass
class DeGaMSConfig:
    text_dim: int = 768
    mel_dim: int = 768  # wav2vec2-base turn embeds (clean_v2 VAD-pooled)
    visual_dim: int = 210  # OpenFace core 29 + pad; DualVisual OF head uses first 29
    openface_dim: int = OPENFACE_CORE_DIM
    # DenseNet/CLIP-style high-order branch; 0 → legacy DualVisual / single Linear
    hi_visual_dim: int = 0
    hidden: int = 256
    shared_dim: int = 256
    specific_dim: int = 256
    lstm_layers: int = 2
    dropout: float = 0.3  # paper uses 0.7 in CNN path; keep milder for small N
    contrast_temp: float = 0.07
    cmcl_temp: float = 0.07
    # loss λ — extra SupCon on the shared subspace is unused (λ≡0).
    # lambda_sscl kept only so old checkpoints/args still load.
    lambda_sscl: float = 0.0
    # Paper name: ℒ_SupCon on fused session vector R. Config key kept as lambda_lbcl.
    lambda_lbcl: float = 0.004
    lambda_mse: float = 0.01
    lambda_kl: float = 2.0  # soft-PHQ KL (raised so soft-PHQ ablation is visible)
    lambda_cmcl: float = 0.05  # cross-modal contrastive; 0 disables
    lambda_gate_entropy: float = 0.02  # encourage multi-modal gate mass (TAV)
    cmcl_include_av: bool = True  # also contrast A–V (in addition to T–A, T–V)
    max_len: int = 128
    soft_temp: float = 0.25  # sharper soft PHQ targets (was 0.35)
    # mild class rebalance (not extreme FP punishment)
    class_weight_power: float = 0.5  # 0=uniform, 1=inverse-freq; 0.5 = mild
    neg_emphasis: float = 1.0  # multiply negative-class loss weight ( >1 → fewer FPs)
    use_dual_visual: bool = True
    # Modality ablations (group A): when True, zero out that modality's fusion contrib
    drop_text: bool = False
    drop_audio: bool = False
    drop_visual: bool = False
    # Model/loss ablations (group C)
    ablate_soft_phq: bool = False  # remove soft-PHQ KL (λ_kl→0; keep PHQ-MSE)
    ablate_lbcl: bool = False  # force λ_SupCon→0 (config key kept as ablate_lbcl)
    ablate_msaf: bool = False  # masked mean instead of MultiScaleAttentionFusion
    ablate_decomp: bool = False  # skip shared/specific; fuse encoder H directly
    # Upgrades: CMCL / Gate / TurnAttn (ablate_* → previous behavior)
    ablate_turn_attn: bool = False  # True → masked mean pool over turns
    ablate_gate: bool = False  # True → equal sum fusion (no learned gates)
    ablate_cmcl: bool = False  # True → lambda_cmcl forced to 0
    gate_temp: float = 1.5  # >1 softens softmax (less collapse); unused if ablate_gate
    gate_min_mass: float = 0.08  # floor mass per *present* modality (anti A-collapse)
    train_mod_dropout: float = 0.12  # stochastic T/V drop in Full TAV training
    # Fusion strategy: "gate" | "equal" | "residual"
    #   gate     — ModalityGate (ignored if ablate_gate)
    #   equal    — ft+fa+fv (same as ablate_gate)
    #   residual — ft + α·fa + β·fv (learnable positive α,β; text is main)
    fusion_mode: str = "gate"
    # Bias audio gate logit by per-turn audio_quality / speech_frac (needs batch key)
    gate_quality_prior: bool = False
    gate_quality_scale: float = 2.0  # logit bias scale for (q - 0.5)


class TemporalBackbone(nn.Module):
    """Conv1d + BiLSTM on already-projected [B,T,H]."""

    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            hidden,
            hidden // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if 2 > 1 else 0.0,
        )
        self.out = nn.Linear(hidden, hidden)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.conv(h.transpose(1, 2)).transpose(1, 2)
        lengths = mask.sum(dim=1).clamp_min(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            h, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=h.size(1)
        )
        out = self.out(out)
        return out * mask.unsqueeze(-1).float()


class ModalityEncoder(nn.Module):
    """Conv1d + BiLSTM → [B,T,H] (paper-style light front-end)."""

    def __init__(self, in_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.backbone = TemporalBackbone(hidden, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        return self.backbone(h, mask)


class DualVisualEncoder(nn.Module):
    """Separate projection heads for CLNF (210) vs OpenFace core (29), shared temporal backbone.

    Stops treating zero-padded OpenFace as identical to CLNF in one Linear.
    """

    def __init__(
        self,
        clnf_dim: int,
        openface_dim: int,
        hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.openface_dim = openface_dim
        self.proj_clnf = nn.Sequential(
            nn.Linear(clnf_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.proj_of = nn.Sequential(
            nn.Linear(openface_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.backbone = TemporalBackbone(hidden, dropout)

    def forward(
        self,
        visual: torch.Tensor,
        mask: torch.Tensor,
        visual_kind: torch.Tensor,
        visual_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # visual [B,T,210], visual_kind [B]
        h_clnf = self.proj_clnf(visual)
        of_in = visual[..., : self.openface_dim]
        h_of = self.proj_of(of_in)
        kind = visual_kind.view(-1, 1, 1)
        is_clnf = (kind == VIS_KIND_CLNF).float()
        is_of = (kind == VIS_KIND_OPENFACE).float()
        # Missing → zeros (present_v also gates later)
        h = h_clnf * is_clnf + h_of * is_of
        if visual_valid is not None:
            h = h * visual_valid.unsqueeze(-1).float()
        return self.backbone(h, mask)


class DenseVisualEncoder(nn.Module):
    """OpenFace core + high-order (DenseNet/CLIP) MLP projections, fused then temporal.

    Input layout: visual[..., :of_dim] = OF, visual[..., of_dim:of_dim+hi_dim] = hi.
    Per-turn ``visual_valid`` zeros invalid turns before the temporal backbone.
    """

    def __init__(
        self,
        openface_dim: int,
        hi_dim: int,
        hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.openface_dim = openface_dim
        self.hi_dim = hi_dim
        self.proj_of = nn.Sequential(
            nn.Linear(openface_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.proj_hi = nn.Sequential(
            nn.Linear(hi_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.backbone = TemporalBackbone(hidden, dropout)

    def forward(
        self,
        visual: torch.Tensor,
        mask: torch.Tensor,
        visual_kind: Optional[torch.Tensor] = None,
        visual_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        of_in = visual[..., : self.openface_dim]
        hi_in = visual[..., self.openface_dim : self.openface_dim + self.hi_dim]
        h = self.fuse(torch.cat([self.proj_of(of_in), self.proj_hi(hi_in)], dim=-1))
        if visual_valid is not None:
            h = h * visual_valid.unsqueeze(-1).float()
        return self.backbone(h, mask)


class Decomposition(nn.Module):
    """Shared FC (weight-tied across modalities) + modality-specific FC."""

    def __init__(self, hidden: int, shared_dim: int, specific_dim: int) -> None:
        super().__init__()
        self.shared = nn.Linear(hidden, shared_dim)  # tied
        self.spec_t = nn.Linear(hidden, specific_dim)
        self.spec_a = nn.Linear(hidden, specific_dim)
        self.spec_v = nn.Linear(hidden, specific_dim)

    def forward_mod(
        self, h: torch.Tensor, which: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        c = F.normalize(self.shared(h), dim=-1)
        if which == "t":
            s = F.normalize(self.spec_t(h), dim=-1)
        elif which == "a":
            s = F.normalize(self.spec_a(h), dim=-1)
        else:
            s = F.normalize(self.spec_v(h), dim=-1)
        return c, s


class TurnAttentionPool(nn.Module):
    """Learned mask-aware attention pooling over turns → [B,D]."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x [B,T,D], mask [B,T]
        logits = self.score(x).squeeze(-1)
        logits = logits.masked_fill(~mask.bool(), float("-inf"))
        w = torch.softmax(logits, dim=-1)
        w = torch.nan_to_num(w, nan=0.0)
        return (w.unsqueeze(-1) * x).sum(dim=1)


class ModalityGate(nn.Module):
    """Per-sample soft gates over T/A/V before fusion (presence-aware).

    Softmax alone yields a convex combination (avg scale). Ablation equal-sum
    uses ft+fa+fv (sum scale). Without rescaling, Gate≈1/3 of ablate magnitude
    when weights are uniform — MSAF sees a silent scale mismatch. Fix: multiply
    by n_present so uniform gates recover equal-sum of present modalities.
    Temperature >1 softens gates (less winner-take-all collapse on small N).
    ``min_mass`` floors each *present* modality so TAV cannot silently collapse
    to audio-only (which previously made A≈TAV under thr=10 saturation).
    """

    def __init__(
        self,
        dim: int,
        dropout: float = 0.1,
        temperature: float = 1.5,
        min_mass: float = 0.08,
    ) -> None:
        super().__init__()
        self.temperature = max(float(temperature), 1e-4)
        self.min_mass = float(max(0.0, min_mass))
        self.net = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 3),
        )
        # Zero-init final layer → uniform softmax at start (identity vs equal-sum)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        ft: torch.Tensor,
        fa: torch.Tensor,
        fv: torch.Tensor,
        mask: torch.Tensor,
        *,
        present_t: torch.Tensor,
        present_a: torch.Tensor,
        present_v: torch.Tensor,
        pool_fn,
        audio_quality: Optional[torch.Tensor] = None,
        quality_scale: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns gated sum [B,T,D] and gate weights [B,3] (softmax over present)."""
        pt = pool_fn(ft, mask)
        pa = pool_fn(fa, mask)
        pv = pool_fn(fv, mask)
        logits = self.net(torch.cat([pt, pa, pv], dim=-1))  # [B,3]
        # Optional quality prior: raise/lower audio logit by mean turn quality
        if audio_quality is not None and float(quality_scale) != 0.0:
            q = audio_quality.float()
            if q.dim() == 2:
                m = mask.float()
                q = (q * m).sum(dim=-1) / m.sum(dim=-1).clamp_min(1.0)
            logits = logits.clone()
            logits[:, 1] = logits[:, 1] + float(quality_scale) * (q.view(-1) - 0.5)
        # Presence mask: dropped / missing modalities get -inf before softmax
        present = torch.stack(
            [
                present_t.view(-1).float(),
                present_a.view(-1).float(),
                present_v.view(-1).float(),
            ],
            dim=-1,
        )  # [B,3]
        logits = logits.masked_fill(present < 0.5, float("-inf"))
        gates = torch.softmax(logits / self.temperature, dim=-1)
        gates = torch.nan_to_num(gates, nan=0.0)
        # Floor mass on present modalities, then re-normalize (anti audio-collapse)
        n_pres = present.sum(dim=-1).clamp_min(1.0)  # [B]
        eps = self.min_mass
        if eps > 0.0:
            # residual mass after floors; clamp eps so n_pres*eps < 1
            eps_eff = torch.minimum(
                gates.new_full((), eps),
                (0.95 / n_pres).clamp(max=eps),
            )
            # broadcast: each present mod gets ≥ eps_eff
            floor = present * eps_eff.unsqueeze(-1)
            remain = (1.0 - floor.sum(dim=-1, keepdim=True)).clamp_min(0.0)
            gates = floor + remain * gates
            gates = gates / gates.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            gates = gates * present  # keep absent at 0
            gates = gates / gates.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        # Scale to equal-sum magnitude: n_pres * Σ g_m f_m  (== Σ f_m when g uniform)
        scale = n_pres.view(-1, 1, 1)
        g_t = gates[:, 0:1].unsqueeze(-1)  # [B,1,1]
        g_a = gates[:, 1:2].unsqueeze(-1)
        g_v = gates[:, 2:3].unsqueeze(-1)
        fused = scale * (g_t * ft + g_a * fa + g_v * fv)
        return fused, gates


class ResidualTextMainFusion(nn.Module):
    """Text-main residual fusion: f = ft + α·fa + β·fv (α,β > 0 via softplus)."""

    def __init__(self) -> None:
        super().__init__()
        # softplus(0)≈0.693 → mild A/V contribution at init
        self.raw_alpha = nn.Parameter(torch.zeros(()))
        self.raw_beta = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        ft: torch.Tensor,
        fa: torch.Tensor,
        fv: torch.Tensor,
        *,
        present_a: torch.Tensor,
        present_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha = F.softplus(self.raw_alpha)
        beta = F.softplus(self.raw_beta)
        pa = present_a.view(-1, 1, 1).float()
        pv = present_v.view(-1, 1, 1).float()
        fused = ft + alpha * fa * pa + beta * fv * pv
        # Diagnostic "gates" as relative scale weights (not softmax)
        scales = fused.new_zeros(ft.size(0), 3)
        scales[:, 0] = 1.0
        scales[:, 1] = (alpha * pa.view(-1)).detach()
        scales[:, 2] = (beta * pv.view(-1)).detach()
        return fused, scales


class MultiScaleAttentionFusion(nn.Module):
    """Multi-scale conv fusion; returns turn sequence [B,T,D] (pool externally)."""

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [nn.Conv1d(dim, dim, k, padding=k // 2) for k in (1, 3, 5)]
        )
        self.attn = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.Tanh(),
            nn.Linear(dim, 3),
        )
        self.out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        xt = x.transpose(1, 2)
        feats = [b(xt).transpose(1, 2) for b in self.branches]
        cat = torch.cat(feats, dim=-1)
        w = torch.softmax(self.attn(cat), dim=-1)
        fused = sum(w[..., i : i + 1] * feats[i] for i in range(3))
        fused = self.out(fused)
        return fused * mask.unsqueeze(-1).float()


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).float()
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


class DeGaMSModel(nn.Module):
    def __init__(self, cfg: Optional[DeGaMSConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or DeGaMSConfig()
        c = self.cfg
        self.enc_t = ModalityEncoder(c.text_dim, c.hidden, c.dropout)
        self.enc_a = ModalityEncoder(c.mel_dim, c.hidden, c.dropout)
        hi_dim = int(getattr(c, "hi_visual_dim", 0) or 0)
        if hi_dim > 0 and int(c.visual_dim) >= int(c.openface_dim) + hi_dim:
            self.enc_v = DenseVisualEncoder(
                c.openface_dim, hi_dim, c.hidden, c.dropout
            )
        elif c.use_dual_visual:
            self.enc_v = DualVisualEncoder(
                c.visual_dim, c.openface_dim, c.hidden, c.dropout
            )
        else:
            self.enc_v = ModalityEncoder(c.visual_dim, c.hidden, c.dropout)
        self.decomp = Decomposition(c.hidden, c.shared_dim, c.specific_dim)
        # w/o decomp: fuse encoder H (hidden); full: shared∥specific
        fuse_dim = c.hidden if c.ablate_decomp else (c.shared_dim + c.specific_dim)
        self.fuse_dim = fuse_dim
        self.msaf = MultiScaleAttentionFusion(fuse_dim, c.dropout)
        # TurnAttn pools over turns for fusion R and modality vectors (shared/specific dims)
        pool_dim_fuse = fuse_dim
        pool_dim_mod = c.hidden if c.ablate_decomp else c.shared_dim
        self.turn_attn_fuse = TurnAttentionPool(pool_dim_fuse)
        self.turn_attn_mod = TurnAttentionPool(pool_dim_mod)
        self.modality_gate = ModalityGate(
            fuse_dim,
            c.dropout,
            temperature=float(getattr(c, "gate_temp", 1.5)),
            min_mass=float(getattr(c, "gate_min_mass", 0.08)),
        )
        self.residual_fuse = ResidualTextMainFusion()
        self.item_heads = nn.ModuleList([nn.Linear(fuse_dim, 4) for _ in range(8)])

    def _pool_turns(self, x: torch.Tensor, mask: torch.Tensor, which: str = "fuse") -> torch.Tensor:
        if self.cfg.ablate_turn_attn:
            return masked_mean(x, mask)
        if which == "mod":
            return self.turn_attn_mod(x, mask)
        return self.turn_attn_fuse(x, mask)

    def encode(
        self,
        text: torch.Tensor,
        mel: torch.Tensor,
        visual: torch.Tensor,
        mask: torch.Tensor,
        present_v: torch.Tensor,
        visual_kind: Optional[torch.Tensor] = None,
        audio_quality: Optional[torch.Tensor] = None,
        visual_valid: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        c = self.cfg
        use_t = not c.drop_text
        use_a = not c.drop_audio
        use_v = not c.drop_visual
        if not (use_t or use_a or use_v):
            raise RuntimeError("All modalities dropped — invalid ablation")
        skip_decomp = bool(c.ablate_decomp)

        B, T, _ = text.shape
        # Defense-in-depth: physically isolate dropped modalities at the tensor
        # boundary BEFORE any encoder / gate / quality prior can touch them.
        # (Skip-encoder alone is correct; zero-out prevents accidental reuse.)
        if not use_t:
            text = text.new_zeros(text.shape)
        if not use_a:
            mel = mel.new_zeros(mel.shape)
            audio_quality = None
        if not use_v:
            visual = visual.new_zeros(visual.shape)
            present_v = present_v.new_zeros(present_v.shape)
            if visual_valid is not None:
                visual_valid = visual_valid.new_zeros(visual_valid.shape)

        # Train-time stochastic T/V dropout (Full TAV only): force fusion to rely on
        # more than audio, preventing silent A-collapse that made A≈TAV on CMDC.
        p_drop = float(getattr(c, "train_mod_dropout", 0.0) or 0.0)
        if (
            self.training
            and p_drop > 0.0
            and use_t
            and use_a
            and use_v
            and not (c.drop_text or c.drop_audio or c.drop_visual)
        ):
            # Independent Bernoulli; never drop all of T+V (keep at least one of T/V)
            drop_t = torch.rand((), device=text.device) < p_drop
            drop_v = torch.rand((), device=text.device) < p_drop
            if drop_t and drop_v:
                # keep text with 50% else visual
                if torch.rand((), device=text.device) < 0.5:
                    drop_t = text.new_zeros((), dtype=torch.bool)
                else:
                    drop_v = text.new_zeros((), dtype=torch.bool)
            if bool(drop_t):
                use_t = False
                text = text.new_zeros(text.shape)
            if bool(drop_v):
                use_v = False
                visual = visual.new_zeros(visual.shape)
                present_v = present_v.new_zeros(present_v.shape)
                if visual_valid is not None:
                    visual_valid = visual_valid.new_zeros(visual_valid.shape)

        # Per-turn visual key-padding (AND with sequence mask)
        if visual_valid is None:
            visual_valid = mask
        else:
            visual_valid = visual_valid.bool() & mask.bool()

        # Skip dropped encoders: zeros (no fusion / CMCL contribution)
        if use_t:
            ht = self.enc_t(text, mask)
            if skip_decomp:
                ct, st = ht, None
            else:
                ct, st = self.decomp.forward_mod(ht, "t")
        else:
            zdim = c.hidden if skip_decomp else c.shared_dim
            ct = text.new_zeros(B, T, zdim)
            st = None if skip_decomp else text.new_zeros(B, T, c.specific_dim)

        if use_a:
            ha = self.enc_a(mel, mask)
            if skip_decomp:
                ca, sa = ha, None
            else:
                ca, sa = self.decomp.forward_mod(ha, "a")
        else:
            zdim = c.hidden if skip_decomp else c.shared_dim
            ca = mel.new_zeros(B, T, zdim)
            sa = None if skip_decomp else mel.new_zeros(B, T, c.specific_dim)

        if use_v:
            if isinstance(self.enc_v, (DualVisualEncoder, DenseVisualEncoder)):
                if visual_kind is None and isinstance(self.enc_v, DualVisualEncoder):
                    visual_kind = torch.where(
                        present_v,
                        torch.full_like(present_v, VIS_KIND_CLNF, dtype=torch.long),
                        torch.zeros_like(present_v, dtype=torch.long),
                    )
                hv = self.enc_v(visual, mask, visual_kind, visual_valid)
            else:
                vis_in = visual * visual_valid.unsqueeze(-1).float()
                hv = self.enc_v(vis_in, mask)
            if skip_decomp:
                cv, sv = hv, None
            else:
                cv, sv = self.decomp.forward_mod(hv, "v")
        else:
            zdim = c.hidden if skip_decomp else c.shared_dim
            cv = visual.new_zeros(B, T, zdim)
            sv = None if skip_decomp else visual.new_zeros(B, T, c.specific_dim)

        # Presence gates (visual may still be missing per-sample even when modality kept)
        pt = float(use_t)
        pa = float(use_a)
        # Session present AND at least one valid visual turn
        turn_has_v = visual_valid.any(dim=1).float()
        pv = present_v.view(-1, 1, 1).float() * float(use_v) * turn_has_v.view(-1, 1, 1)
        present_t = text.new_full((B,), float(use_t))
        present_a = mel.new_full((B,), float(use_a))
        present_v1 = present_v.float() * float(use_v) * turn_has_v

        vv = visual_valid.unsqueeze(-1).float()
        if skip_decomp:
            cv = cv * pv * vv
            ft = ct * pt
            fa = ca * pa
            fv = cv
        else:
            cv, sv = cv * pv * vv, sv * pv * vv
            ft = torch.cat([ct, st], dim=-1) * pt
            fa = torch.cat([ca, sa], dim=-1) * pa
            fv = torch.cat([cv, sv], dim=-1)

        # Fusion strategy: residual / equal / gate
        mode = str(getattr(c, "fusion_mode", "gate") or "gate").lower()
        if bool(c.ablate_gate):
            mode = "equal"
        if mode == "residual":
            f_sum, gates = self.residual_fuse(
                ft, fa, fv, present_a=present_a, present_v=present_v1
            )
        elif mode == "equal":
            f_sum = ft + fa + fv
            gates = None
        else:
            q_scale = (
                float(getattr(c, "gate_quality_scale", 2.0))
                if bool(getattr(c, "gate_quality_prior", False))
                else 0.0
            )
            # Never feed audio_quality when audio modality is ablated
            aq = audio_quality if (q_scale != 0.0 and use_a) else None
            f_sum, gates = self.modality_gate(
                ft,
                fa,
                fv,
                mask,
                present_t=present_t,
                present_a=present_a,
                present_v=present_v1,
                pool_fn=lambda x, m: self._pool_turns(x, m, "fuse"),
                audio_quality=aq,
                quality_scale=q_scale if use_a else 0.0,
            )

        if c.ablate_msaf:
            r_seq = f_sum
        else:
            r_seq = self.msaf(f_sum, mask)
        r = self._pool_turns(r_seq, mask, "fuse")

        # Modality pooled vectors (for cross-modal InfoNCE / diagnostics).
        # Shared maps still feed fusion; no extra SupCon is applied on z_shared.
        # Visual pool uses key-padding (valid turns only).
        zt = self._pool_turns(ct, mask, "mod") * pt
        za = self._pool_turns(ca, mask, "mod") * pa
        zv = self._pool_turns(cv, visual_valid, "mod")
        pv1 = present_v1.unsqueeze(-1)
        n_mod = pt + pa + pv1  # [B,1]; at least one modality active by config
        z_raw = (zt + za + zv * pv1) / n_mod.clamp_min(1.0)
        z_shared = F.normalize(z_raw, dim=-1)
        z_shared = torch.nan_to_num(z_shared, nan=0.0)

        item_logits = [head(r) for head in self.item_heads]
        item_probs = [F.softmax(lg, dim=-1) for lg in item_logits]
        scores = torch.arange(4, device=r.device, dtype=r.dtype)
        item_pred = torch.stack([(p * scores).sum(dim=-1) for p in item_probs], dim=-1)
        phq_pred = item_pred.sum(dim=-1)
        out = {
            "R": r,
            "z_shared": z_shared,
            "z_t": zt,
            "z_a": za,
            "z_v": zv,
            "present_t": present_t,
            "present_a": present_a,
            "present_v": present_v1,
            "visual_valid": visual_valid,
            "item_logits": torch.stack(item_logits, dim=1),
            "item_probs": torch.stack(item_probs, dim=1),
            "item_pred": item_pred,
            "phq_pred": phq_pred,
            "drop_text": c.drop_text,
            "drop_audio": c.drop_audio,
            "drop_visual": c.drop_visual,
        }
        if gates is not None:
            out["modality_gates"] = gates
        return out

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.encode(
            batch["text"],
            batch["mel"],
            batch["visual"],
            batch["mask"],
            batch["present_v"],
            batch.get("visual_kind"),
            batch.get("audio_quality"),
            batch.get("visual_valid"),
        )


def phq_item_soft_targets(items: torch.Tensor, temp: float = 0.35) -> torch.Tensor:
    """items [B,8] int 0..3 → soft [B,8,4]."""
    device = items.device
    xs = torch.arange(4, device=device).float().view(1, 1, 4)
    y = items.float().unsqueeze(-1)
    logits = -((xs - y) ** 2) / max(temp, 1e-4)
    return torch.softmax(logits, dim=-1)


def _supcon_loss(z: torch.Tensor, y: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """Sample-to-sample supervised contrastive (SupCon) on [B,D], labels [B].

    Used on the fused session embedding ``R`` (paper: ℒ_SupCon).
    An extra copy of this term on ``z_shared`` is not added to the total loss.
    """
    b = z.size(0)
    if b < 2:
        return z.new_zeros(())
    z = F.normalize(z, dim=-1)
    sim = z @ z.t() / tau
    y = y.view(-1, 1)
    mask_pos = (y == y.t()).float()
    mask_pos.fill_diagonal_(0)
    logits = sim - sim.max(dim=1, keepdim=True).values.detach()
    exp = logits.exp()
    losses = []
    for i in range(b):
        pos = mask_pos[i]
        if pos.sum() < 1:
            continue
        denom = exp[i].sum() - exp[i, i]
        losses.append(-(torch.log((exp[i] * pos).sum() / denom.clamp_min(1e-8))))
    if not losses:
        return z.new_zeros(())
    return torch.stack(losses).mean()


def sscl_loss(z: torch.Tensor, y: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """Deprecated alias. Kept so old imports do not break."""
    return _supcon_loss(z, y, tau=tau)


def supcon_loss(r: torch.Tensor, y: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """Supervised contrastive loss on fused session vector R (same-class attraction)."""
    return _supcon_loss(r, y, tau=tau)


def lbcl_loss(r: torch.Tensor, y: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """Alias of ``supcon_loss`` (legacy name kept for callers)."""
    return supcon_loss(r, y, tau=tau)


def info_nce_paired(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Asymmetric InfoNCE; diagonal is the positive pair. Both [N,D] L2-normalized."""
    if anchor.size(0) < 2:
        return anchor.new_zeros(())
    logits = anchor @ positive.t() / temperature
    labels = torch.arange(anchor.size(0), device=anchor.device)
    return F.cross_entropy(logits, labels)


def bidirectional_infonce(
    a: torch.Tensor,
    b: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return 0.5 * (
        info_nce_paired(a, b, temperature) + info_nce_paired(b, a, temperature)
    )


def cmcl_loss(
    z_t: torch.Tensor,
    z_a: torch.Tensor,
    z_v: torch.Tensor,
    *,
    present_t: torch.Tensor,
    present_a: torch.Tensor,
    present_v: torch.Tensor,
    tau: float = 0.07,
    include_av: bool = True,
) -> torch.Tensor:
    """Cross-modal contrastive on pooled modality vectors (T–A, T–V, optional A–V).

    Only uses samples where both modalities in a pair are present.
    """
    losses = []
    pt = present_t.view(-1) > 0.5
    pa = present_a.view(-1) > 0.5
    pv = present_v.view(-1) > 0.5

    m_ta = pt & pa
    if int(m_ta.sum()) >= 2:
        losses.append(bidirectional_infonce(z_t[m_ta], z_a[m_ta], tau))

    m_tv = pt & pv
    if int(m_tv.sum()) >= 2:
        losses.append(bidirectional_infonce(z_t[m_tv], z_v[m_tv], tau))

    if include_av:
        m_av = pa & pv
        if int(m_av.sum()) >= 2:
            losses.append(bidirectional_infonce(z_a[m_av], z_v[m_av], tau))

    if not losses:
        return z_t.new_zeros(())
    return torch.stack(losses).mean()


def mild_class_weights(
    y_bin: torch.Tensor,
    *,
    power: float = 0.5,
    n0: Optional[float] = None,
    n1: Optional[float] = None,
    neg_emphasis: float = 1.0,
) -> torch.Tensor:
    """Per-sample mild inverse-frequency weights (normalized to mean 1).

    power=0 → uniform; power=1 → full balanced; default 0.5 = mild rebalance.
    neg_emphasis>1 upweights non-depressed samples (precision-oriented).
    """
    y = y_bin.long()
    if n0 is None or n1 is None:
        # batch estimate fallback
        n0_b = float((y == 0).sum().clamp_min(1).item())
        n1_b = float((y == 1).sum().clamp_min(1).item())
        n0 = n0 if n0 is not None else n0_b
        n1 = n1 if n1 is not None else n1_b
    n0 = max(float(n0), 1.0)
    n1 = max(float(n1), 1.0)
    # inverse-freq^power, then mean-normalize
    w0 = ((1.0 / n0) ** power) * float(neg_emphasis)
    w1 = (1.0 / n1) ** power
    w = torch.where(y == 1, torch.full_like(y, w1, dtype=torch.float32), torch.full_like(y, w0, dtype=torch.float32))
    w = w.float()
    return w / w.mean().clamp_min(1e-8)


def degams_total_loss(
    out: Dict[str, torch.Tensor],
    *,
    phq_items: torch.Tensor,
    phq_total: torch.Tensor,
    y_bin: torch.Tensor,
    cfg: DeGaMSConfig,
    class_counts: Optional[Tuple[float, float]] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    mse_per = (out["phq_pred"] - phq_total.float()) ** 2  # [B]

    n0 = n1 = None
    if class_counts is not None:
        n0, n1 = class_counts
    w = mild_class_weights(
        y_bin,
        power=cfg.class_weight_power,
        n0=n0,
        n1=n1,
        neg_emphasis=getattr(cfg, "neg_emphasis", 1.0),
    )

    if getattr(cfg, "ablate_soft_phq", False):
        # True soft-PHQ ablation: remove item soft-KL entirely (keep PHQ-MSE only).
        # Previous hard-CE@same λ_kl ≈ soft KL on CMDC's near-hard labels → ΔF1≈0.
        item_per = out["phq_pred"].new_zeros(out["phq_pred"].shape)
        lam_kl = 0.0
    else:
        soft = phq_item_soft_targets(phq_items, cfg.soft_temp)
        log_p = F.log_softmax(out["item_logits"], dim=-1)
        item_per = F.kl_div(log_p, soft, reduction="none").sum(dim=-1).mean(dim=-1)  # [B]
        lam_kl = float(cfg.lambda_kl)

    l_kl = (item_per * w).mean() if lam_kl > 0 else out["phq_pred"].new_zeros(())
    l_mse = (mse_per * w).mean()
    # Extra SupCon on the shared subspace is unused (ignore lambda_sscl).
    l_sscl = out["R"].new_zeros(())
    l_lbcl = supcon_loss(out["R"], y_bin, cfg.contrast_temp)
    # Defense-in-depth: honor ablate_lbcl even if cfg.lambda_lbcl was not zeroed upstream
    lam_lbcl = 0.0 if getattr(cfg, "ablate_lbcl", False) else float(cfg.lambda_lbcl)

    lam_cmcl = 0.0 if getattr(cfg, "ablate_cmcl", False) else float(getattr(cfg, "lambda_cmcl", 0.0))
    if lam_cmcl > 0 and "z_t" in out:
        l_cmcl = cmcl_loss(
            out["z_t"],
            out["z_a"],
            out["z_v"],
            present_t=out.get("present_t", y_bin.new_ones(y_bin.shape)),
            present_a=out.get("present_a", y_bin.new_ones(y_bin.shape)),
            present_v=out.get("present_v", y_bin.new_ones(y_bin.shape)),
            tau=float(getattr(cfg, "cmcl_temp", cfg.contrast_temp)),
            include_av=bool(getattr(cfg, "cmcl_include_av", True)),
        )
    else:
        l_cmcl = out["R"].new_zeros(())

    # Gate entropy: maximize entropy over present mods (minimize −H)
    l_gate = out["R"].new_zeros(())
    lam_ge = float(getattr(cfg, "lambda_gate_entropy", 0.0) or 0.0)
    gates = out.get("modality_gates")
    if lam_ge > 0 and gates is not None and not bool(getattr(cfg, "ablate_gate", False)):
        present = torch.stack(
            [
                out.get("present_t", gates.new_ones(gates.size(0))),
                out.get("present_a", gates.new_ones(gates.size(0))),
                out.get("present_v", gates.new_ones(gates.size(0))),
            ],
            dim=-1,
        ).float()
        if present.dim() == 1:
            present = present.unsqueeze(0)
        g = (gates * present).clamp_min(1e-8)
        g = g / g.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        # only regularize when ≥2 modalities present
        n_pres = present.sum(dim=-1)
        ent = -(g * g.log()).sum(dim=-1)
        # maximize entropy ↔ minimize negative entropy
        l_gate = (-ent * (n_pres >= 2).float()).mean()

    total = (
        lam_lbcl * l_lbcl
        + cfg.lambda_mse * l_mse
        + lam_kl * l_kl
        + lam_cmcl * l_cmcl
        + lam_ge * l_gate
    )
    return total, {
        "sscl": l_sscl.detach(),  # always 0; key kept for logging compat
        "lbcl": l_lbcl.detach(),  # ℒ_SupCon; key kept as lbcl for logs/plots
        "mse": l_mse.detach(),
        "kl": l_kl.detach(),
        "cmcl": l_cmcl.detach(),
        "gate_ent": l_gate.detach(),
        "lambda_lbcl_eff": out["R"].new_tensor(lam_lbcl).detach(),
        "lambda_kl_eff": out["R"].new_tensor(lam_kl).detach(),
    }
