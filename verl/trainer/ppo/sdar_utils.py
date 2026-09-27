"""
Confidence-Gated Teacher Distillation (SDAR) utilities.

Token-level gated distillation loss where the gate is derived from
the teacher-student log-probability gap, so tokens where the teacher
is more confident receive stronger distillation signal.
"""

from typing import Optional

import torch

from verl.trainer.ppo.core_algos import agg_loss


def compute_sdar_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gate_beta: float = 5.0,
    loss_agg_mode: str = "token-mean",
    gate_mode: str = "sigmoid",
    gate_topq: float = 0.10,
    gate_tau: Optional[float] = None,
) -> tuple[torch.Tensor, dict]:
    """
    Confidence-Gated Teacher Distillation loss.

    L_SDAR = agg( g_t * (log pi_teacher - log pi_student) )

    Three gate modes are supported:
      sigmoid  g_t = sigmoid(beta * delta_t)     (upstream SDAR)
      topq     g_t = 1 on the top-fraction of positive-Delta tokens within the
               batch, 0 elsewhere. Fraction = gate_topq. Only tokens where the
               teacher is MORE confident than the student (delta_t > 0) compete;
               the threshold is set batch-globally so gate_active_ratio is
               bounded by gate_topq regardless of sequence length or delta scale.
               This is the "sparse prior" gate: it avoids pulling every token
               slightly toward the skill text (the length-positive-feedback
               channel), and only distills on the few tokens where the skill
               strongly endorses a specific choice.
      const    g_t = 1 everywhere (ungated KL, diagnostic).

    The gate g_t is detached so gradients only flow through the student log-probs.

    Args:
        student_log_probs: (bs, response_length) - log pi_theta(y_t | x, y_<t).
        teacher_log_probs: (bs, response_length) - log pi_teacher(y_t | x, r, y_<t).
        response_mask: (bs, response_length) - mask for valid response tokens.
        gate_beta: temperature for the sigmoid gate. Higher = sharper gating.
        loss_agg_mode: aggregation mode passed to agg_loss.
        gate_mode: 'sigmoid' | 'topq' | 'const'.
        gate_topq: fraction of tokens to gate-ON under 'topq' (default 0.10).
        gate_tau: optional hard threshold; if set (and gate_mode='topq'), tokens
            with delta_t > gate_tau are selected directly instead of by quantile.

    Returns:
        sdar_loss: scalar loss.
        metrics: dict with gating statistics.
    """
    teacher_log_probs = teacher_log_probs.detach()

    delta_t = teacher_log_probs - student_log_probs.detach()

    if gate_mode == "sigmoid":
        gate = torch.sigmoid(gate_beta * delta_t).detach()
    elif gate_mode == "const":
        gate = torch.ones_like(delta_t)
    elif gate_mode == "topq":
        # Only tokens that (a) are valid, (b) have delta_t > 0 (teacher more
        # confident, not less) compete for the gate.
        eligible = response_mask.to(torch.bool) & (delta_t > 0)
        gate = torch.zeros_like(delta_t)
        if float(eligible.any()):
            pos_d = delta_t[eligible]
            if gate_tau is not None:
                selected = eligible & (delta_t > float(gate_tau))
            else:
                q = max(0.0, min(1.0, float(gate_topq)))
                k = max(1, int(pos_d.numel() * q))
                # torch.kthvalue on a 1-D view returns the k-th smallest; threshold
                # is the value at the (1-q) quantile so that k tokens are above it.
                # Use a quantile-based threshold that works on GPU without sorting
                # the entire eligible set (torch.quantile is available on CUDA).
                thr = torch.quantile(pos_d.float(), 1.0 - q).item()
                selected = eligible & (delta_t >= thr)
            gate = selected.to(delta_t.dtype)
        gate = gate.detach()
    else:
        raise ValueError(f"unknown sdar gate_mode {gate_mode!r}; expected sigmoid|topq|const")

    kl_per_token = teacher_log_probs - student_log_probs

    gated_kl = gate * kl_per_token

    loss = agg_loss(loss_mat=gated_kl, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    with torch.no_grad():
        mask_sum = response_mask.sum().clamp(min=1)
        gate_mean = (gate * response_mask).sum() / mask_sum
        gate_active = ((gate > 0.5).float() * response_mask).sum() / mask_sum
        gap_mean = (delta_t * response_mask).sum() / mask_sum
        # on gated tokens only: how big is the gap that actually pushes the grad?
        gated_mask = (gate > 0.5) & response_mask.to(torch.bool)
        if bool(gated_mask.any()):
            gap_gated = (delta_t * gated_mask.to(delta_t.dtype)).sum() / gated_mask.to(delta_t.dtype).sum().clamp(min=1)
        else:
            gap_gated = torch.zeros(())

    metrics = {
        "sdar/gate_mean": gate_mean.item(),
        "sdar/gate_active_ratio": gate_active.item(),
        "sdar/teacher_gap_mean": gap_mean.item(),
        "sdar/teacher_gap_gated_mean": float(gap_gated),
        "sdar/loss": loss.detach().item(),
    }

    return loss, metrics
