"""
RLSD (Reinforcement Learning with Self-Distillation) utilities.

This module provides skill-based privileged information loading and
token-level advantage computation for the RLSD algorithm.
All task types and keyword mappings are loaded from skill_mapping.json,
making this module environment-agnostic.
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch


def load_skill_mapping(skills_dir: str) -> dict:
    mapping_path = os.path.join(skills_dir, "skill_mapping.json")
    with open(mapping_path, "r") as f:
        return json.load(f)


def load_skill_content(skills_dir: str, skill_mapping: dict) -> Dict[str, str]:
    """Load all skill markdown files into a dict keyed by skill name."""
    contents = {}
    for skill_name, filename in skill_mapping["skill_files"].items():
        filepath = os.path.join(skills_dir, filename)
        with open(filepath, "r") as f:
            contents[skill_name] = f.read().strip()
    return contents


class SkillProvider:
    """Loads and caches skill files, provides privileged info per task type.

    Everything is driven by skill_mapping.json:
      - skill_files: maps skill names to markdown filenames
      - task_to_skill: maps task type strings to skill names
      - task_keywords (optional): ordered list of (task_type -> keywords) for
        inferring task type from prompt text. Keywords are matched in order;
        the first task type whose ALL keywords appear in the text wins.
    """

    def __init__(self, skills_dir: str, skill_all: bool = False, include_general: bool = True):
        self.skills_dir = skills_dir
        self.skill_all = skill_all
        self.include_general = include_general
        self.skill_mapping = load_skill_mapping(skills_dir)
        self.skill_contents = load_skill_content(skills_dir, self.skill_mapping)
        self.task_to_skill = self.skill_mapping["task_to_skill"]
        # task_keywords: ordered dict of task_type -> list of keywords (all must match)
        self.task_keywords: Dict[str, List[str]] = self.skill_mapping.get("task_keywords", {})

        if self.skill_all:
            self._all_skills_text = self._build_all_skills_text()

    def _build_all_skills_text(self) -> str:
        """Concatenate general_skills + all task-specific skills."""
        parts = []
        if self.include_general:
            general = self.skill_contents.get("general_skills", "")
            parts.append(general)
        for skill_name, content in self.skill_contents.items():
            if skill_name != "general_skills":
                parts.append(content)
        return "\n\n".join(parts)

    def _get_skill_text(self, task_type: Optional[str]) -> str:
        """Assemble general_skills + task-specific skill text (general controllable)."""
        parts = []
        if self.include_general:
            general = self.skill_contents.get("general_skills", "")
            parts.append(general)
        if task_type:
            mapped_name = self.task_to_skill.get(task_type)
            if mapped_name and mapped_name in self.skill_contents:
                parts.append(self.skill_contents[mapped_name])
        return "\n\n".join(parts)

    def get_privileged_info(self, gamefile: str) -> str:
        """Return skills for a gamefile path (task type appears as substring)."""
        if self.skill_all:
            return self._all_skills_text
        matched_task = None
        for task_type in self.task_to_skill:
            if task_type in gamefile:
                matched_task = task_type
                break
        return self._get_skill_text(matched_task)

    def get_privileged_info_from_prompt(self, prompt_text: str) -> str:
        """Infer task type from prompt text using keyword rules from skill_mapping.json.

        Uses ``any`` matching: a task type is matched if ANY of its keywords
        appear in the prompt.  When multiple task types match, all of their
        skill texts are concatenated (general_skills is included only once).
        """
        if self.skill_all:
            return self._all_skills_text
        text_lower = prompt_text.lower()
        matched_tasks = []
        for task_type, keywords in self.task_keywords.items():
            if keywords and any(kw in text_lower for kw in keywords):
                matched_tasks.append(task_type)

        if not matched_tasks:
            return self._get_skill_text(None)

        general = self.skill_contents.get("general_skills", "")
        parts = [general]
        for task_type in matched_tasks:
            mapped_name = self.task_to_skill.get(task_type)
            if mapped_name and mapped_name in self.skill_contents:
                parts.append(self.skill_contents[mapped_name])
        return "\n\n".join(parts)

    def get_privileged_info_from_data_source(self, data_source: str, prompt_text: str) -> str:
        """Infer task type using data_source field and prompt content.

        Matching rules:
          - data_source == 'popqa' -> entity_attribute_lookup
          - data_source in ('nq', 'triviaqa') -> direct_retrieval
          - prompt contains 'which' + 'or' (without 'for') -> compare
          - data_source == 'hotpotqa' -> multi_hop_reasoning
          - data_source == 'text' (AlfWorld/Webshop parquet) -> prompt keyword matching
          - otherwise -> general skills only (unknown)
        """
        if self.skill_all:
            return self._all_skills_text

        task_type = None
        if data_source == "popqa":
            task_type = "entity_attribute_lookup"
        elif data_source in ("nq", "triviaqa"):
            task_type = "direct_retrieval"
        elif data_source == "hotpotqa":
            task_type = "multi_hop_reasoning"
        else:
            text_lower = prompt_text.lower()
            if "which" in text_lower and "or" in text_lower and "for" not in text_lower:
                task_type = "compare"
            elif data_source in ("2wikimultihopqa", "musique", "bamboogle"):
                task_type = "multi_hop_reasoning"

        if task_type is None and data_source == "text":
            return self.get_privileged_info_from_prompt(prompt_text)
        return self._get_skill_text(task_type)


def centered_delta(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    add_clip: float = 2.0,
    center: bool = True,
) -> torch.Tensor:
    """Δ_t = log π_teacher − log π_student, de-biased and outlier-clipped.

    Only used by the additive forms. Two deliberate choices:

    * The batch-level mean is subtracted. Δ has a measured global offset
      (teacher_student_gap_mean ≈ −0.13 on ALFWorld/Qwen2.5-3B), and an offset
      added to every advantage is a uniform push on the likelihood of every
      token, carrying no credit information at all. The multiplicative form is
      immune to this (a constant offset only rescales w), the additive form is
      not.
    * The result is NOT divided by the within-row std. Δ is sharply peaked at 0
      with heavy tails -- ~64% of tokens land inside ±0.2 where a Gaussian of
      the same std would put 22% -- because most tokens are template/syntax and
      only a few decision tokens carry the gap. Standardising within a step
      would normalise away exactly the sparsity that makes the token tier
      informative. ``add_clip`` is a rail against outliers, not a rescaling: at
      the default 2.0 it is ~3σ and touches a fraction of a percent.
    """
    delta_t = (teacher_log_probs - student_log_probs) * response_mask
    if center:
        mask_sum = response_mask.sum().clamp(min=1.0)
        delta_t = delta_t - (delta_t.sum() / mask_sum)
    return torch.clamp(delta_t, -add_clip, add_clip) * response_mask


def compute_opsd_step_tier(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    step_group_uids,
    step_rewards: torch.Tensor,
    step_adv_row: torch.Tensor,
    c_prior: float = 1.0,
    prior_beta: float = 1.0,
    gate: float = 1.0,
    rho_hat: float = None,
    a_reliability: float = 0.5,
    eps: float = 1e-6,
    prior_teacher: bool = False,
):
    """A SECOND estimate of the same latent quantity GiGPO's step advantage estimates.

    GiGPO's A^S answers "was step k better than its siblings at the same anchor
    state?" using the environment return. The privileged teacher answers the very
    same question from an unrelated direction: ``delta_bar_k = mean_j Delta_{k,j}``
    is how much more likely the student's own action becomes once the teacher is
    allowed to see the hindsight. delta_bar_k > 0 means the privileged view endorses
    what the student did at step k; < 0 means that, knowing how the episode should
    have gone, the action looks unlikely. Same question, different evidence.

    To be fusable with A^S the teacher estimate has to be made commensurate:

    * **Centred in the SAME anchor group.** Both are relative-to-siblings
      quantities, never absolute ones. Centring A^OPSD anywhere else would fuse a
      relative number with an absolute one.
    * **Scaled to A^S's spread.** Delta is in nats, the step reward is not. The
      rescale is estimated on rows where both tiers exist -- including the untaught
      rows (whose A^OPSD is identically 0) would deflate the teacher tier's std and
      so silently inflate every surviving teacher row.

    After both, A^OPSD is a drop-in replacement for A^S: same normalisation, same
    units, different evidence source. That is the whole point -- it makes
    ``c*A^S + (1-c)*A^OPSD`` an interpolation between two estimators rather than a
    sum of two unrelated things.

    The reliability weight is inverse-variance -- the optimal linear combination of
    two unbiased estimators of one quantity, and it needs no learned parameters:

        prec_S = n_anchor / Var(R_anchor)      the anchor group's baseline precision
        prec_O = L_k / Var_within(Delta_k)     the teacher estimate's precision

    but it must be compared RELATIVELY, not absolutely. The two precisions live in
    incomparable units (inverse returns vs inverse nats, the latter further scaled by
    ``scale**2`` from the commensurability rescale), so ``prec_S/(prec_S+prec_O)``
    collapses to whichever endpoint the unit mismatch happens to favour -- measured
    c = 0.911 with the scale factor in, c ~ 0.015 with it out. Each precision is
    therefore normalised by its OWN median OVER THE ARBITRATING ROWS (those carrying
    both kinds of evidence), and a single explicit prior sets the global trust level::

        z_S = prec_S / median_both(prec_S)     "unusually sharp anchor group?"
        z_O = prec_O / median_both(prec_O)     "unusually sharp teacher?"
        c_k = c_prior*z_S / (c_prior*z_S + z_O)

    ``c_prior = 1`` puts the median arbitrating row at c = 1/2 exactly and leaves the
    rest to the per-row evidence; it is the one interpretable, ablatable knob of the method. The
    naive absolute form is kept reachable as ``c_mode='invvar_raw'`` (returned in
    ``diag['c_raw']``) because it is the obvious thing to try and its collapse is
    worth reporting. Medians rather than means because a precision is 1/variance and
    one near-degenerate row would send a mean to infinity.

    Absence is handled as zero precision, NOT as large variance, so the weight
    degrades gracefully at both ends instead of dividing by zero:

    * A singleton anchor group, or one whose siblings all scored identically, has
      A^S == 0 exactly -- no discriminating information. prec_S = 0 => c = 0 => the
      teacher takes over. This is exactly the hole ``decoupled`` leaves open
      (measured step_adv_zero_row_frac 0.56 at step 60), so ``fusion`` should shrink
      rlsd/zero_adv_row_frac relative to it. That is a falsifiable prediction.
    * An untaught row has Delta identically 0. prec_O = 0 => c = 1 => the fused tier
      falls back to plain A^S, i.e. to ``decoupled``.
    * Both absent => both terms are 0 anyway; c is set to 1 for determinism.

    Returns (all shaped ``(bs,)``):
        opsd_row   A^OPSD_k, anchor-centred and scale-matched to omega*A^S
        c_eff      the EFFECTIVE reliability after role separation: inverse-variance
                   arbitration (gated) on both-evidence rows, 1-prior_beta on vacuum
                   rows, 1 where there is no teacher. (prior_beta=1, gate=1) == the
                   pure inverse-variance weight.
        c_invvar   the median-normalised reliability in [0, 1]
        diag       dict of tensors for metrics: n_anchor, has_step, has_teacher,
                   scale (scalar), prec_S, prec_O, z_S, z_O, c_raw
    """
    device = response_mask.device
    dtype = student_log_probs.dtype
    bs = response_mask.size(0)

    delta = (teacher_log_probs - student_log_probs) * response_mask
    L = response_mask.sum(dim=-1).to(dtype).clamp(min=1.0)              # (bs,)
    delta_bar = delta.sum(dim=-1) / L                                   # (bs,)
    d_center = (delta - delta_bar.unsqueeze(-1)) * response_mask
    delta_var = (d_center ** 2).sum(dim=-1) / L                         # (bs,)
    has_teacher = delta.abs().sum(dim=-1) > 0                           # (bs,) bool

    # ---- anchor-group reductions ----
    _, inv = np.unique(np.asarray(step_group_uids), return_inverse=True)
    inv_t = torch.as_tensor(inv, device=device, dtype=torch.long)
    n_grp = int(inv.max()) + 1 if bs > 0 else 0
    zeros_g = torch.zeros(n_grp, device=device, dtype=dtype)
    cnt = zeros_g.clone().index_add_(0, inv_t, torch.ones(bs, device=device, dtype=dtype))

    g_sum_d = zeros_g.clone().index_add_(0, inv_t, delta_bar)
    opsd_raw = delta_bar - (g_sum_d / cnt)[inv_t]

    r = step_rewards.to(device=device, dtype=dtype).reshape(-1)
    g_sum_r = zeros_g.clone().index_add_(0, inv_t, r)
    r_dev = r - (g_sum_r / cnt)[inv_t]
    # (n-1) denominator to match the torch.std GiGPO itself uses on these groups.
    r_var = (zeros_g.clone().index_add_(0, inv_t, r_dev ** 2) / (cnt - 1.0).clamp(min=1.0))[inv_t]
    n_anchor = cnt[inv_t]

    # ---- scale match, estimated on taught rows only (see docstring) ----
    scale = torch.zeros((), device=device, dtype=dtype)
    if int(has_teacher.sum()) > 1:
        s_op = opsd_raw[has_teacher].std()
        if float(s_op) > eps:
            scale = step_adv_row.to(device=device, dtype=dtype)[has_teacher].std() / (s_op + eps)
    opsd_row = opsd_raw * scale

    # ---- precisions ----
    has_step = (n_anchor >= 2) & (r_var > 0)
    prec_S = torch.where(has_step, n_anchor / r_var.clamp(min=eps), torch.zeros_like(r_var))
    var_O = (delta_var / L) * (scale ** 2)
    prec_O = torch.where(has_teacher, 1.0 / var_O.clamp(min=eps), torch.zeros_like(var_O))

    # ABSOLUTE inverse-variance weighting does not work here, and the reason is
    # structural rather than a tuning miss. prec_S is in inverse-return units and
    # prec_O in inverse-nats; the scale match that makes A^OPSD commensurate with A^S
    # also multiplies its variance by scale^2, and scale measured 25.6 on the first
    # real batch, so scale^2 = 657. That put prec_S/prec_O ~ 10 and collapsed c to
    # 0.911 (86% of rows at c >= 0.9, only 9% arbitrating). Dropping the scale^2 does
    # not repair it, it inverts it: prec_O would be ~1028 against prec_S ~16, i.e.
    # c ~ 0.015. The absolute ratio is dominated by the unit mismatch in whichever
    # direction the arithmetic happens to land, so NEITHER endpoint is evidence about
    # which source to trust. Kept as c_mode='invvar_raw' because that failure is worth
    # one appendix paragraph -- it is the obvious thing to try and it does not work.
    #
    # What IS comparable is each source's precision RELATIVE TO ITS OWN typical value:
    # z_S asks "is this anchor group better or worse than usual at pinning down the
    # baseline", z_O asks the same of the teacher. Medians rather than means, because
    # a precision is 1/variance and one near-degenerate group sends a mean to infinity.
    # The global trust level then becomes a single explicit, ablatable prior c_prior
    # instead of an artefact of nats-vs-returns: c_prior=1 makes the median row a coin
    # flip and leaves the arbitrating to the per-row evidence.
    #
    # This makes c_k batch-relative, like every other quantity in this family (GRPO
    # and GiGPO advantages are both group-normalised); it is not a new dependence.
    #
    # BOTH medians are taken over the rows that ACTUALLY ARBITRATE (has_step &
    # has_teacher), not over each source's own population. Normalising prec_S over all
    # has_step rows was measurably wrong: prec_S = n/Var(R_anchor), and an anchor group
    # containing a success has a large return spread (a few rollouts score 10, the rest
    # 0) hence a SMALL prec_S, while an all-fail group has near-zero spread hence a huge
    # one. Under teacher='peer' the taught rows are exactly the rows whose group
    # contains a success, so the median over all has_step rows is set by the all-fail
    # groups -- a population that can never arbitrate, having no teacher -- and every
    # arbitrating row lands at z_S << 1. Measured on step 1 of the smoke:
    # prec_ratio_median 1.6e4, and 93% of the both-evidence rows shoved to c <= 0.1.
    # Normalising on the arbitrating population removes that by construction and makes
    # c_prior exact: the median arbitrating row sits at c_prior/(1+c_prior).
    both_ev = has_step & has_teacher

    def _median_of(x, primary, fallback):
        m = primary if int(primary.sum()) > 0 else fallback
        return x[m].median() if int(m.sum()) > 0 else torch.ones((), device=device, dtype=dtype)

    z_S = torch.where(has_step, prec_S / _median_of(prec_S, both_ev, has_step).clamp(min=eps),
                      torch.zeros_like(prec_S))
    z_O = torch.where(has_teacher, prec_O / _median_of(prec_O, both_ev, has_teacher).clamp(min=eps),
                      torch.zeros_like(prec_O))
    num = c_prior * z_S
    denom = num + z_O
    c_invvar = torch.where(denom > 0, num / denom.clamp(min=eps), torch.ones_like(denom))
    c_invvar = c_invvar.clamp(0.0, 1.0)

    # ---- LOADING CORRECTION (c_mode='corr'; rho_hat is not None) ----
    # The weight above is inverse-variance, which is the optimal combination of two
    # unbiased estimators OF THE SAME QUANTITY. A^OPSD is not that in general:
    #     A^S = T + eps_S            A^OPSD = a*T + eps_O
    # and `a` -- how much of the teacher gap is actually about the latent step
    # advantage -- never appears above. Setting a == 1 by omission is what let a
    # skill teacher (rho ~ 0.002, measured 20 steps, sign rate 0.55) take 58-89% of
    # the step tier: its gap is nearly CONSTANT across steps, because the ~900-token
    # prefix is the same every step, so its variance is tiny and inverse-variance
    # reads tiny variance as high reliability. Low-variance noise is the worst case
    # for this rule, and it is exactly the case a skill teacher produces.
    #
    # Recovering `a` needs one number the median normalisation threw away. Because
    # opsd_row was already rescaled to A^S's spread above, sigma_O == sigma_S == s,
    # so with u := Var(T)/s^2 (the RELIABILITY of A^S itself):
    #     rho = corr(A^OPSD, A^S) = a*Var(T)/s^2 = a*u      =>   a = rho/u
    #     Var(eps_S) = s^2*(1-u)        Var(eps_O) = s^2*(1 - rho^2/u)
    #     c* = prec_S / (prec_S + a^2*prec_O) = u*(u - rho^2) / (u^2 - 2*u*rho^2 + rho^2)
    # Endpoints, all exact:
    #     rho -> 0  =>  c* -> 1  =>  fused == A^S, i.e. `decoupled`, WITHOUT being told
    #     u   -> 1  =>  c* -> 1  =>  a perfect environment estimate never yields to a teacher
    #     u   -> rho^2 => c* -> 0 =>  the teacher explains all of A^S's signal, take it
    #
    # c* is a GLOBAL trust level, so it enters exactly where the hand-set one did:
    # c_prior. That is the point -- c_prior stops being a hyperparameter.
    #   both-evidence rows: per-row arbitration is kept, only its centre of mass moves
    #                       (at the median row z_S == z_O == 1, so c == c*).
    #   vacuum rows (A^S == 0): no second estimate to arbitrate against, so c* IS the
    #                       posterior shrinkage: fused = (1 - c*)*A^OPSD. This is
    #                       `prior_beta`, now derived rather than set -- and it is the
    #                       branch the beta sweep proved harmful at every hand-set dose.
    #   the sign test on rho is deleted: c* is continuous in rho, so there is no
    #                       threshold to random-walk across and no repeated test.
    # `u` is the one surviving knob, and unlike c_prior it is a property of the
    # BACKBONE (rollout count, env stochasticity), not of the teacher, so it does not
    # silently change dose when the teacher type changes.
    if rho_hat is not None:
        _u = float(min(max(a_reliability, 1e-3), 1.0))
        _r2 = float(min(max(rho_hat, 0.0), 1.0)) ** 2
        _r2 = min(_r2, _u * (1.0 - 1e-6))          # keep u - rho^2 > 0
        _den = _u * _u - 2.0 * _u * _r2 + _r2
        _cstar = 1.0 if _den <= 0 else min(max(_u * (_u - _r2) / _den, 0.0), 1.0)
        if _cstar >= 1.0 - 1e-6:
            # teacher carries no information about T: fall through to A^S everywhere,
            # including the vacuum rows. This is `decoupled`, reached from data.
            c_invvar = torch.ones_like(c_invvar)
        else:
            _num2 = (_cstar / (1.0 - _cstar)) * z_S
            _den2 = _num2 + z_O
            c_invvar = torch.where(_den2 > 0, _num2 / _den2.clamp(min=eps), torch.ones_like(_den2))
            c_invvar = torch.where(has_step & has_teacher, c_invvar,
                                   torch.where(has_teacher,
                                               torch.full_like(c_invvar, _cstar),   # vacuum
                                               torch.ones_like(c_invvar)))          # no teacher
        c_invvar = c_invvar.clamp(0.0, 1.0)

    # ---- ROLE SEPARATION: A^OPSD is not one kind of evidence, it is two ----
    # The form above silently assumes A^O is always a RIVAL ESTIMATE of the latent
    # A^S estimates. Measured across five runs, that holds for a hindsight/peer
    # teacher and fails for a static skill-library teacher:
    #
    #   teacher   corr(A^O, A^S)   steps with rho>0   sign-test z   signal/noise
    #   peer          +0.30..0.39        100%            +12.0        3.4 .. 5.8
    #   skill         -0.05..+0.00      20..40%           -3.8        1.4 .. 2.4
    #
    # When `prior_teacher=True` the caller has identified this teacher as a
    # STATIC PRIOR rather than a second estimate (see rlsd_ray_trainer's
    # teacher_role routing). In that case two hard safety rules apply ON TOP OF
    # any rho_hat/gate/prior_beta arithmetic:
    #
    #   * vacuum rows (teacher present, A^S == 0): c_eff is forced to 1, i.e.
    #     fused_S = A^S exactly. A static prior must NOT inject advantage where
    #     the environment has no signal -- that is the channel that caused the
    #     length-positive-feedback collapse (response_length +12 tok/step,
    #     val 24% -> 0.8% in 50 steps). Regularisation belongs in the KL term,
    #     not in the advantage on information-free rows.
    #   * arbitration rows (both evidence): left to the existing gate/rho_hat
    #     machinery. If rho ~ 0 the corr path will already push c* -> 1; keeping
    #     the arbitration branch alive lets a future genuinely-correlated static
    #     signal still earn weight without code changes.
    #
    # prior_teacher=False reproduces every published arm bit-for-bit.
    vacuum = has_teacher & ~has_step
    c_eff = c_invvar
    if prior_teacher:
        # Hard vacuum-closer for prior teachers. Overrides prior_beta/c* on
        # vacuum rows; arbitration rows untouched.
        c_eff = torch.where(vacuum, torch.ones_like(c_eff), c_eff)
    else:
        if gate < 1.0:
            # arbitration rows only: shrink A^O's share toward 0; gate=0 => pure A^S.
            c_eff = torch.where(both_ev, 1.0 - (1.0 - c_eff) * gate, c_eff)
        if prior_beta != 1.0:
            # vacuum rows only: A_S == 0 there, so c = 1 - beta gives fused = beta * A^O.
            c_eff = torch.where(vacuum, torch.full_like(c_eff, 1.0 - prior_beta), c_eff)
    if gate < 1.0 or prior_beta != 1.0 or prior_teacher:
        c_eff = c_eff.clamp(0.0, 1.0)

    # the naive absolute version, retained for the ablation
    d_raw = prec_S + prec_O
    c_raw = torch.where(d_raw > 0, prec_S / d_raw.clamp(min=eps), torch.ones_like(d_raw))
    c_raw = c_raw.clamp(0.0, 1.0)

    diag = {
        "n_anchor": n_anchor,
        "has_step": has_step,
        "has_teacher": has_teacher,
        "scale": scale,
        "prec_S": prec_S,
        "prec_O": prec_O,
        "z_S": z_S,
        "z_O": z_O,
        "c_raw": c_raw,
        # the pure inverse-variance weight, BEFORE role separation. Kept so the
        # trainer can log what arbitration alone would have chosen and so the
        # (prior_beta, gate) ablation has its own baseline column.
        "c_invvar": c_invvar,
        "vacuum": vacuum,
        "both_ev": both_ev,
    }
    return opsd_row, c_eff, diag


def compute_rlsd_token_advantage(
    seq_advantages: torch.Tensor,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    rlsd_lambda: float = 0.5,
    rlsd_clip_eps: float = 0.2,
    form: str = "mult",
    add_clip: float = 2.0,
    center_delta: bool = True,
    episode_advantages: torch.Tensor = None,
    step_advantages: torch.Tensor = None,
    opsd_advantages: torch.Tensor = None,
    reliability: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute token-level RLSD advantages.

    Args:
        seq_advantages: (bs, response_length) — sequence-level advantage (GRPO or
            GiGPO) broadcast to all tokens (same value per sequence).
        student_log_probs: (bs, response_length) — log π_θ(y_t | x, y_<t).
        teacher_log_probs: (bs, response_length) — log π_θ(y_t | x, r, y_<t).
        response_mask: (bs, response_length) — mask for valid response tokens.
        rlsd_lambda: mixing coefficient λ. When 0, every form degrades to the
            plain sequence-level advantage.
        rlsd_clip_eps: clipping bound ε_w on the token weight w_t (``mult`` only).
        form: how the teacher gap enters the advantage.
            ``mult``   Â = A·[(1−λ) + λ·clip(exp(sign(A)Δ))]  — upstream RLSD.
                       sign(Â) = sign(A) always, since w > 0: the teacher can
                       only modulate the magnitude of outcome credit, never
                       contradict it. Price: A ≡ 0 ⇒ Â ≡ 0, so the token tier is
                       inert on groups with no outcome variance (measured: 33%
                       of rows on ALFWorld, where all 8 rollouts of a task fail).
            ``add``    Â = A + λ·Δ_c                        — additive.
                       Survives A ≡ 0, so it is the only form that puts gradient
                       on all-fail groups. Price: it CAN flip the sign of the
                       outcome credit; watch rlsd/sign_flip_frac.
            ``hybrid`` multiplicative where A ≠ 0, additive where A = 0.
                       Keeps mult's sign guarantee exactly where outcome credit
                       exists, and falls back to the teacher only where the
                       group carries no information.
            ``decoupled`` Â = A^E + ω·A^S·[(1−λ) + λ·w]  — the teacher is confined
                       to the STEP tier and never touches the episode tier.
                       Three tiers with three distinct jobs: A^E says whether the
                       trajectory succeeded, A^S says whether this step was good,
                       w says which tokens inside the step carry that step credit.
                       Two consequences, both measurable:
                         + The sign(A^E) vs sign(A^E + ω·A^S) ambiguity disappears
                           rather than being resolved: w keys off sign(A^S), the
                           only tier it modulates. (Measured on ALFWorld with
                           ω=1.0, the ambiguity is moot anyway -- the step tier
                           never overturns the episode tier's sign, 0/5 steps.)
                         − The teacher goes inert wherever A^S == 0, which is
                           where the anchor group has no return spread. Measured
                           at 39-81% of rows (mean 70%), against mult's 0-34%.
                           So this trades teacher REACH for teacher PRECISION,
                           and rlsd/teacher_active_frac is the price tag.
            ``fusion``  Â = A^E + [c_k·A^S + (1−c_k)·A^OPSD]·[(1−λ) + λ·w]. Same
                       three tiers as ``decoupled``, but the step tier is no longer
                       the environment's estimate alone: A^OPSD is a SECOND estimate
                       of the same per-step quantity, derived from the teacher gap
                       and made commensurate (see compute_opsd_step_tier), and c_k
                       arbitrates between them by inverse variance.
                       Motivated by a measurement, not by symmetry: the two evidence
                       sources are CO-LOCATED 1.7-2.2x above the independence
                       prediction yet directionally uncorrelated (corr_taught ≈ 0.08)
                       with a ~0.56 sign-conflict rate. Co-occurrence without
                       redundancy is exactly the regime where arbitration can pay --
                       no co-occurrence and there is nothing to arbitrate, correlated
                       and the second source is noise.
                       ``c_k ≡ 1`` reproduces ``decoupled`` bit for bit; that is the
                       positive control, and the ablation ladder mult -> decoupled ->
                       fusion runs through it.
        add_clip: outlier rail on Δ_c (``add``/``hybrid`` only). See centered_delta.
        center_delta: subtract Δ's batch mean (``add``/``hybrid`` only).
        episode_advantages: (bs, response_length) — A^E alone. ``decoupled``/``fusion``.
        step_advantages: (bs, response_length) — ω·A^S, the step tier WITH the
            GiGPO weight already applied. ``decoupled``/``fusion``.
        opsd_advantages: (bs,) — A^OPSD, the teacher-derived step tier, already
            anchor-centred and scale-matched. ``fusion`` only. See
            compute_opsd_step_tier.
        reliability: (bs,) — c_k ∈ [0,1], how much of the step tier to take from the
            environment rather than from the teacher. ``fusion`` only.

    Returns:
        token_advantages: (bs, response_length) — token-level advantage Â_t.
    """
    if form not in ("mult", "add", "hybrid", "decoupled", "fusion"):
        raise ValueError(f"unknown rlsd form {form!r}; expected mult|add|hybrid|decoupled|fusion")
    if form in ("decoupled", "fusion") and (episode_advantages is None or step_advantages is None):
        raise ValueError(f"form={form!r} needs episode_advantages and step_advantages "
                         "(the two GiGPO tiers), not just their sum")
    if form == "fusion" and (opsd_advantages is None or reliability is None):
        raise ValueError("form='fusion' needs opsd_advantages and reliability "
                         "(see compute_opsd_step_tier); without them it is just 'decoupled'")

    with torch.no_grad():
        first_valid = response_mask.int().argmax(dim=-1)  # (bs,)
        batch_indices = torch.arange(seq_advantages.size(0), device=seq_advantages.device)
        A_seq = seq_advantages[batch_indices, first_valid]  # (bs,)
        A_seq_expanded = A_seq.unsqueeze(-1)  # (bs, 1)

        if form in ("mult", "hybrid"):
            delta_t = teacher_log_probs - student_log_probs  # (bs, response_length)
            sign_A = torch.sign(A_seq).unsqueeze(-1)  # (bs, 1)
            w_t = torch.exp(sign_A * delta_t)  # (bs, response_length)
            w_t = torch.clamp(w_t, 1.0 - rlsd_clip_eps, 1.0 + rlsd_clip_eps)
            mult_adv = A_seq_expanded * ((1.0 - rlsd_lambda) + rlsd_lambda * w_t)

        if form == "decoupled":
            delta_t = teacher_log_probs - student_log_probs
            A_E = episode_advantages[batch_indices, first_valid].unsqueeze(-1)  # (bs, 1)
            A_S = step_advantages[batch_indices, first_valid].unsqueeze(-1)     # (bs, 1), already x omega
            # The weight keys off the STEP tier's sign, not the joint advantage's.
            # That is the whole point: w redistributes credit inside a step, so the
            # direction it should preserve is the one the step tier asserts. It also
            # sidesteps the sign(A^E) vs sign(A^E + w*A^S) ambiguity entirely --
            # the question does not arise once the teacher never touches A^E.
            sign_S = torch.sign(A_S)
            w_t = torch.exp(sign_S * delta_t)
            w_t = torch.clamp(w_t, 1.0 - rlsd_clip_eps, 1.0 + rlsd_clip_eps)
            token_advantages = A_E + A_S * ((1.0 - rlsd_lambda) + rlsd_lambda * w_t)

        if form == "fusion":
            delta_t = teacher_log_probs - student_log_probs
            A_E = episode_advantages[batch_indices, first_valid].unsqueeze(-1)  # (bs, 1)
            A_S = step_advantages[batch_indices, first_valid].unsqueeze(-1)     # (bs, 1)
            A_O = opsd_advantages.reshape(-1, 1).to(A_S.dtype)                  # (bs, 1)
            c_k = reliability.reshape(-1, 1).to(A_S.dtype)                      # (bs, 1)
            # The step tier stops being "the environment's step advantage" and becomes
            # a fusion of two estimates of the same quantity. c_k == 1 recovers
            # 'decoupled' bit for bit, which is the positive control for this form.
            fused_S = c_k * A_S + (1.0 - c_k) * A_O
            # The token weight keys off the FUSED tier's sign, for the same reason
            # 'decoupled' keys off A^S's: w redistributes whatever step credit the
            # tier below it just asserted.
            sign_F = torch.sign(fused_S)
            w_t = torch.exp(sign_F * delta_t)
            w_t = torch.clamp(w_t, 1.0 - rlsd_clip_eps, 1.0 + rlsd_clip_eps)
            token_advantages = A_E + fused_S * ((1.0 - rlsd_lambda) + rlsd_lambda * w_t)

        if form in ("add", "hybrid"):
            delta_c = centered_delta(
                student_log_probs, teacher_log_probs, response_mask,
                add_clip=add_clip, center=center_delta,
            )
            add_adv = A_seq_expanded + rlsd_lambda * delta_c

        if form == "mult":
            token_advantages = mult_adv
        elif form == "add":
            token_advantages = add_adv
        elif form == "hybrid":
            token_advantages = torch.where(
                (A_seq == 0).unsqueeze(-1), add_adv, mult_adv,
            )

        token_advantages = token_advantages * response_mask

    return token_advantages
