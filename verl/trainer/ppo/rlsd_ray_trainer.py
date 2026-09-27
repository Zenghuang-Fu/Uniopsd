"""
RLSD (Reinforcement Learning with Self-Distillation) Trainer.

Extends the standard RayPPOTrainer with a teacher forward pass that uses
privileged information (skills) to construct token-level advantages.
"""

from pprint import pprint

import os
import numpy as np
import ray
import re
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    _timer,
    apply_invalid_action_penalty,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from gigpo import core_gigpo
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.rlsd_utils import (
    SkillProvider,
    centered_delta,
    compute_opsd_step_tier,
    compute_rlsd_token_advantage,
)
from verl.utils.metric import reduce_metrics
from verl.utils.torch_functional import masked_mean
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.utils.model import compute_position_id_with_mask

from agent_system.multi_turn_rollout import adjust_batch


PEER_HINDSIGHT_HEADER = (
    "[Privileged Hindsight Information]\n"
    "Another attempt at this exact task succeeded. Its action sequence was:\n"
)
_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL)
# Search-QA never emits <action>. Its projection accepts <search>...</search> and
# <answer>...</answer> instead (search/projection.py:55-56), so before this
# fallback existed every Search group extracted an empty action list, hit the
# `if not actions: continue` below, and got an empty prefix -- Delta == 0
# everywhere and fusion silently degraded to plain GiGPO. Measured 2026-08-16:
# teacher_coverage == 0 on all 5 smoke steps while episode/success_rate was 0.29.
# This fallback fires ONLY when <action> is absent, so ALFWorld and WebShop are
# bit-for-bit unchanged. It keeps the tags in the rendered text (unlike the
# <action> path, which renders bare) because a bare "Bonn" is indistinguishable
# from another query, whereas "<answer>Bonn</answer>" is not.
_SEARCH_ACT_RE = re.compile(r"<(search|answer)>(.*?)</\1>", re.DOTALL)


def _row_column(batch: DataProto, key: str):
    """Fetch a per-row column from either half of a DataProto as a numpy array."""
    if key in batch.non_tensor_batch:
        return np.asarray(batch.non_tensor_batch[key])
    if key in batch.batch.keys():
        return batch.batch[key].detach().cpu().numpy()
    raise KeyError(f"{key!r} is in neither batch.batch nor batch.non_tensor_batch")


def build_peer_hindsight_prefixes(
    batch: DataProto,
    tokenizer,
    success_threshold: float = 1.0,
    max_peer_steps: int = 0,
    hide_answer: bool = False,
) -> list:
    """StepOPSD's privileged teacher context: the first successful peer in the group.

    This is the no-skill teacher. Where SkillProvider hands every row the same
    static, hand-written text, here the privileged information is another
    trajectory that the policy itself produced on the SAME task in the SAME
    rollout group and that actually succeeded -- hindsight, not prior knowledge.
    StepOPSD (arXiv:2605.27140 §3.1): "When a GRPO group contains both successes
    and failures, we condition the teacher context of a failed trajectory on the
    first successful peer from that exact same group."

    Two consequences that the caller has to know about:

    * Rows of a SUCCESSFUL trajectory get an empty prefix, and build_teacher_batch
      then reuses their student input_ids verbatim, so Delta == 0 exactly there.
      Only failures are taught.
    * A group in which every rollout failed has no successful peer, so ALL of its
      rows get an empty prefix and Delta == 0. That is the same set of rows on
      which GiGPO's step advantage and the episode advantage are also exactly 0
      -- peer hindsight and the environment's local signal fail together, unlike
      the skill teacher which is defined everywhere. This is a property of the
      method, not a bug, but it is the thing to measure.

    Only the <action> spans of the peer are kept, not its full reasoning: a
    44-step ALFWorld trajectory averages 134 response tokens/step (~5.9k tokens),
    which does not fit the 2048-token prompt budget, while the action sequence
    alone is ~350 tokens. It is also the part that carries the hindsight -- what
    the successful attempt DID. On Search-QA, where there is no <action> tag, the
    equivalent spans are <search> and <answer> (see _SEARCH_ACT_RE).

    ON SEARCH THE HINDSIGHT CONTAINS THE LABEL, and the paper has to say so.
    peer_success_threshold=1.0 means the peer answered with EM=1, so its final
    <answer> IS the ground truth. Structurally this is the same deal ALFWorld
    offers -- a successful trajectory's last action is by definition the one that
    achieved the goal -- but the strength is not the same: on Search the student
    can copy the answer without doing the retrieval, whereas on ALFWorld it still
    has to execute the sequence. hide_answer=True drops the <answer> spans and
    hands over only the retrieval path; that is a DIFFERENT method (teach the
    search strategy, not the fact) and belongs in the ablation table, not the
    default.

    Args:
        batch: rollout batch; rows are (trajectory, step) pairs.
        tokenizer: used to decode the peer's responses back to text.
        success_threshold: a trajectory counts as successful when its episode
            reward reaches this (ALFWorld pays 10 on success, 0 otherwise).
        max_peer_steps: keep only the last N actions of the peer; 0 = keep all.
        hide_answer: Search-only ablation -- keep the peer's <search> queries but
            strip its <answer>, so the teacher sees the retrieval path without
            the label. No effect on environments that use <action>.

    Returns:
        list[str] of length bs -- the privileged prefix for each row, "" when the
        row gets no teacher.
    """
    uid = _row_column(batch, "uid")
    traj_uid = _row_column(batch, "traj_uid")
    turn_step = _row_column(batch, "turn_step").astype(np.int64)
    episode_rewards = _row_column(batch, "episode_rewards").astype(np.float64)

    # trajectory -> its rows, its group, its outcome. First-appearance order is
    # preserved so "the first successful peer" is well defined even after
    # balance_batch has reordered rows (traj_uid/turn_step travel with the row).
    traj_rows: dict = {}
    traj_group: dict = {}
    traj_reward: dict = {}
    group_trajs: dict = {}
    for i in range(len(traj_uid)):
        t = traj_uid[i]
        if t not in traj_rows:
            traj_rows[t] = []
            traj_group[t] = uid[i]
            traj_reward[t] = float(episode_rewards[i])
            group_trajs.setdefault(uid[i], []).append(t)
        traj_rows[t].append(i)

    # one decode call for the peers only -- decoding all ~5.6k rows would cost
    # far more than the teacher forward pass it feeds
    chosen_peer: dict = {}
    for g, trajs in group_trajs.items():
        for t in trajs:
            if traj_reward[t] >= success_threshold:
                chosen_peer[g] = t
                break

    peer_rows = sorted({i for t in chosen_peer.values() for i in traj_rows[t]})
    action_of_row: dict = {}
    if peer_rows:
        texts = tokenizer.batch_decode(
            batch.batch["responses"][peer_rows], skip_special_tokens=True,
        )
        for i, text in zip(peer_rows, texts):
            found = _ACTION_RE.findall(text)
            if found:
                action_of_row[i] = found[-1].strip()
                continue
            tagged = _SEARCH_ACT_RE.findall(text)
            if hide_answer:
                tagged = [(tag, body) for tag, body in tagged if tag != "answer"]
            if tagged:
                tag, body = tagged[-1]
                action_of_row[i] = f"<{tag}>{body.strip()}</{tag}>"
            else:
                action_of_row[i] = ""

    group_prefix: dict = {}
    for g, t in chosen_peer.items():
        rows = sorted(traj_rows[t], key=lambda i: int(turn_step[i]))
        # adjust_batch(mode="copy") pads the batch to a divisible size by
        # DUPLICATING random rows, so one turn_step can appear twice. Keep the
        # first occurrence, otherwise the peer's action list repeats a step.
        seen_steps = set()
        actions = []
        for i in rows:
            s = int(turn_step[i])
            if s in seen_steps:
                continue
            seen_steps.add(s)
            a = action_of_row.get(i, "")
            if a:
                actions.append(a)
        if not actions:
            continue
        if max_peer_steps > 0:
            actions = actions[-max_peer_steps:]
        body = "\n".join(f"{k + 1}. {a}" for k, a in enumerate(actions))
        group_prefix[g] = f"{PEER_HINDSIGHT_HEADER}{body}\n\n"

    prefixes = [""] * len(traj_uid)
    for t, rows in traj_rows.items():
        if traj_reward[t] >= success_threshold:
            continue                              # successes are not taught
        prefix = group_prefix.get(traj_group[t], "")
        for i in rows:
            prefixes[i] = prefix
    return prefixes


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    """Pearson correlation of two 1-D tensors; 0.0 if either is constant.

    Diagnostics only. Returns a float so it can go straight into the metrics
    dict, and degrades to 0.0 rather than NaN on a degenerate batch (e.g. every
    row in the same step group, so A^step is uniformly 0) -- a NaN would poison
    the logger for the whole run.
    """
    x = x.float().flatten()
    y = y.float().flatten()
    xc, yc = x - x.mean(), y - y.mean()
    denom = xc.norm() * yc.norm()
    if float(denom) < 1e-12:
        return 0.0
    return float((xc @ yc) / denom)


def build_teacher_batch(
    batch: DataProto,
    skill_provider: SkillProvider,
    tokenizer,
    max_prompt_length: int,
    truncation: str = "error",
    prefixes: list = None,
):
    """
    Build a teacher batch by prepending privileged info to each sample's prompt.

    The teacher sees (x, r) where r is the privileged information. We prepend it
    as text before the user prompt, then re-tokenize to get teacher
    input_ids/attention_mask/position_ids. The responses remain unchanged.

    Args:
        batch: The original student batch with prompts and responses.
        skill_provider: SkillProvider instance for loading skills. Ignored when
            ``prefixes`` is given.
        tokenizer: The tokenizer.
        max_prompt_length: Maximum prompt length.
        truncation: Truncation mode.
        prefixes: optional per-row privileged text (e.g. from
            build_peer_hindsight_prefixes). When supplied it REPLACES the skill
            lookup, which is how the no-skill teacher is selected. A row whose
            prefix is "" keeps its student input_ids byte-for-byte, so its
            teacher log probs equal its student log probs and Delta is exactly 0
            -- re-tokenising an unprefixed prompt would not guarantee that,
            because decode->encode is not always the identity.

    Returns:
        teacher_batch: A DataProto with modified input_ids/attention_mask/position_ids
            but the same responses, suitable for computing teacher log probs.
    """
    bs = batch.batch["input_ids"].size(0)
    response_length = batch.batch["responses"].size(1)

    teacher_input_ids_list = []
    teacher_attention_mask_list = []
    teacher_position_ids_list = []

    # Truncation here is LEFT truncation (keep the end of the prompt), and the
    # privileged text sits at the LEFT, so an over-long teacher prompt eats the
    # privileged text first -- silently turning the teacher back into the student.
    # Peer hindsight grows with the peer's trajectory length, so this has to be
    # measured, not assumed. Prefixes are per-group, so the encode cache is small.
    _prefix_len_cache: dict = {}
    n_prefixed = 0
    n_prefix_cut = 0
    n_prefix_lost = 0
    prefix_tok_sum = 0

    for i in range(bs):
        if prefixes is not None and not prefixes[i]:
            # no privileged information for this row -> teacher == student
            teacher_input_ids_list.append(batch.batch["input_ids"][i])
            teacher_attention_mask_list.append(batch.batch["attention_mask"][i])
            teacher_position_ids_list.append(batch.batch["position_ids"][i])
            continue

        # Decode the original prompt (student input minus response)
        original_input_ids = batch.batch["input_ids"][i]
        original_attention_mask = batch.batch["attention_mask"][i]
        prompt_length = original_input_ids.size(0) - response_length

        prompt_ids = original_input_ids[:prompt_length]
        prompt_mask = original_attention_mask[:prompt_length]

        # Find the first non-padding token in prompt
        valid_start = prompt_mask.nonzero(as_tuple=True)[0]
        if len(valid_start) > 0:
            valid_start = valid_start[0].item()
        else:
            valid_start = 0

        valid_prompt_ids = prompt_ids[valid_start:]
        prompt_text = tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)

        # Get privileged skill info based on gamefile, data_source, or prompt text
        gamefile = batch.non_tensor_batch.get("gamefile", None)
        data_source = batch.non_tensor_batch.get("data_source", None)
        if prefixes is not None:
            skill_prefix = prefixes[i]
        elif gamefile is not None:
            gf = gamefile[i]
            if gf is not None:
                gf = gf if isinstance(gf, str) else str(gf)
                skill_text = skill_provider.get_privileged_info(gf)
            elif data_source is not None:
                ds = data_source[i] if isinstance(data_source[i], str) else str(data_source[i])
                skill_text = skill_provider.get_privileged_info_from_data_source(ds, prompt_text)
            else:
                skill_text = skill_provider.get_privileged_info_from_prompt(prompt_text)
        elif data_source is not None:
            ds = data_source[i] if isinstance(data_source[i], str) else str(data_source[i])
            skill_text = skill_provider.get_privileged_info_from_data_source(ds, prompt_text)
        else:
            skill_text = skill_provider.get_privileged_info_from_prompt(prompt_text)

        # Construct teacher prompt: prepend the privileged text before the prompt
        if prefixes is None:
            skill_prefix = f"[Privileged Skill Information]\n{skill_text}\n\n"
        teacher_prompt_text = skill_prefix + prompt_text

        # Tokenize the teacher prompt
        teacher_prompt_ids = tokenizer.encode(teacher_prompt_text, add_special_tokens=False)

        # Account for how much of the privileged text survives left truncation
        if skill_prefix not in _prefix_len_cache:
            _prefix_len_cache[skill_prefix] = len(
                tokenizer.encode(skill_prefix, add_special_tokens=False))
        prefix_len = _prefix_len_cache[skill_prefix]
        dropped = max(0, len(teacher_prompt_ids) - max_prompt_length)
        n_prefixed += 1
        prefix_tok_sum += prefix_len
        if dropped > 0:
            n_prefix_cut += 1
            if dropped >= prefix_len:
                n_prefix_lost += 1

        # Truncate if needed (left truncation to keep the end of prompt)
        if len(teacher_prompt_ids) > max_prompt_length:
            teacher_prompt_ids = teacher_prompt_ids[-max_prompt_length:]

        teacher_prompt_ids = torch.tensor(teacher_prompt_ids, dtype=torch.long)
        actual_prompt_len = len(teacher_prompt_ids)

        # Pad to max_prompt_length (left padding)
        pad_length = max_prompt_length - actual_prompt_len
        if pad_length > 0:
            pad_ids = torch.full((pad_length,), tokenizer.pad_token_id, dtype=torch.long)
            teacher_prompt_ids = torch.cat([pad_ids, teacher_prompt_ids])
            t_prompt_mask = torch.cat([
                torch.zeros(pad_length, dtype=torch.long),
                torch.ones(actual_prompt_len, dtype=torch.long),
            ])
        else:
            t_prompt_mask = torch.ones(actual_prompt_len, dtype=torch.long)

        # Combine with response
        response_ids = batch.batch["responses"][i]
        response_mask = original_attention_mask[-response_length:]

        teacher_full_ids = torch.cat([teacher_prompt_ids, response_ids])
        teacher_full_mask = torch.cat([t_prompt_mask, response_mask])
        teacher_position_ids = compute_position_id_with_mask(teacher_full_mask.unsqueeze(0))[0]

        teacher_input_ids_list.append(teacher_full_ids)
        teacher_attention_mask_list.append(teacher_full_mask)
        teacher_position_ids_list.append(teacher_position_ids)

    teacher_input_ids = torch.stack(teacher_input_ids_list)
    teacher_attention_mask = torch.stack(teacher_attention_mask_list)
    teacher_position_ids = torch.stack(teacher_position_ids_list)

    teacher_batch = DataProto.from_dict(
        tensors={
            "input_ids": teacher_input_ids,
            "attention_mask": teacher_attention_mask,
            "position_ids": teacher_position_ids,
            "responses": batch.batch["responses"],
        },
    )
    teacher_batch.meta_info["prefix_stats"] = {
        "prefixed_frac": n_prefixed / max(bs, 1),
        "prefix_cut_frac": n_prefix_cut / max(n_prefixed, 1),
        "prefix_lost_frac": n_prefix_lost / max(n_prefixed, 1),
        "prefix_tokens_mean": prefix_tok_sum / max(n_prefixed, 1),
    }

    return teacher_batch


class RLSDRayTrainer(RayPPOTrainer):
    """
    RLSD trainer that extends RayPPOTrainer with self-distillation
    using privileged skill information as teacher signal.
    """

    def __init__(self, *args, skill_provider: SkillProvider = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.skill_provider = skill_provider
        # RLSD hyperparams from config
        rlsd_cfg = self.config.algorithm.get("rlsd", {})
        self.rlsd_lambda_init = rlsd_cfg.get("rlsd_lambda", 0.5)
        self.rlsd_lambda_warmdown_steps = rlsd_cfg.get("warmdown_steps", 50)
        self.rlsd_clip_eps = rlsd_cfg.get("clip_eps", 0.2)
        # How the teacher gap enters the advantage: mult (upstream) | add | hybrid |
        # decoupled | fusion. See compute_rlsd_token_advantage for the trade-off.
        self.rlsd_form = rlsd_cfg.get("form", "mult")
        if self.rlsd_form not in ("mult", "add", "hybrid", "decoupled", "fusion"):
            raise ValueError(
                f"unknown rlsd form {self.rlsd_form!r}; expected mult|add|hybrid|decoupled|fusion")
        if (self.rlsd_form in ("decoupled", "fusion")
                and self.config.algorithm.adv_estimator != AdvantageEstimator.GiGPO):
            # Fail here rather than eight minutes into the first rollout: decoupled
            # needs A^E and A^S as separate tensors, and only the GiGPO branch
            # recomputes them. Any other estimator hands back a single scalar per row.
            raise ValueError(
                f"algorithm.rlsd.form={self.rlsd_form} requires algorithm.adv_estimator=gigpo "
                f"(got {self.config.algorithm.adv_estimator!r}) -- it separates the "
                "episode and step tiers, which only GiGPO produces.")
        # How the fused step tier splits between the environment's estimate A^S and
        # the teacher's estimate A^OPSD. ``fusion`` only.
        #   invvar      c_k from inverse variance, each precision normalised by its own
        #               batch median and tilted by c_prior. This is the method.
        #   invvar_raw  the naive absolute c_k = prec_S/(prec_S+prec_O). Kept ONLY as an
        #               ablation: the two precisions are in incomparable units, so it
        #               collapses to an endpoint (measured c_mean 0.911, 86% of rows at
        #               c >= 0.9). Reported, not used.
        #   const       c_k = c_const everywhere. c_const=1.0 must reproduce 'decoupled'
        #               bit for bit -- that is the positive control -- and c_const=0.0 is
        #               the pure-teacher endpoint. This is the ablation axis.
        self.rlsd_c_mode = rlsd_cfg.get("c_mode", "invvar")
        if self.rlsd_c_mode not in ("invvar", "invvar_raw", "const", "corr"):
            raise ValueError(
                f"unknown rlsd c_mode {self.rlsd_c_mode!r}; expected invvar|invvar_raw|const|corr. "
                "A learned c_k is deliberately NOT implemented: nothing in this setup "
                "supervises it, and a gate trained on no signal is a free parameter "
                "dressed as a method.")
        self.rlsd_c_const = float(rlsd_cfg.get("c_const", 1.0))
        # Global trust tilt between the two step-tier estimates. 1.0 puts the batch's
        # median row at c = 1/2 and lets the per-row evidence do the arbitrating; >1
        # leans on the environment return, <1 on the teacher. The one free parameter
        # the fusion has, and it is a prior rather than something fitted.
        self.rlsd_c_prior = float(rlsd_cfg.get("c_prior", 1.0))
        # ---- role separation of A^OPSD (see compute_opsd_step_tier) ----
        # prior_beta: what a PRIOR is worth on the rows where the environment has no
        # opinion (A^S == 0 exactly). 1.0 = the published behaviour, which grants a
        # skill-conformance prior the same authority as a realised return; 0.0 =
        # 'decoupled', the teacher goes silent there. Both endpoints are measured, so
        # this axis is a genuine interpolation and not a new degree of freedom bolted on.
        self.rlsd_prior_beta = float(rlsd_cfg.get("prior_beta", 1.0))
        if not (0.0 <= self.rlsd_prior_beta <= 1.0):
            raise ValueError(f"rlsd prior_beta must be in [0,1], got {self.rlsd_prior_beta}")
        # opsd_gate: whether A^O is allowed to ARBITRATE against A^S on the rows where
        # both exist. 'off' always allows it (published behaviour). 'signtest'
        # requires corr(A^O, A^S) > 0 to be established first -- fusion's own
        # shared-latent premise, tested rather than assumed. Uses only PREVIOUS steps'
        # correlations, never the current batch's, so it cannot double-dip.
        self.rlsd_opsd_gate = str(rlsd_cfg.get("opsd_gate", "off"))
        if self.rlsd_opsd_gate not in ("off", "signtest"):
            raise ValueError(f"unknown rlsd opsd_gate {self.rlsd_opsd_gate!r}; expected off|signtest")
        self.rlsd_opsd_gate_z = float(rlsd_cfg.get("opsd_gate_z", 2.33))      # one-sided ~1%
        self.rlsd_opsd_gate_min_n = int(rlsd_cfg.get("opsd_gate_min_n", 8))   # z cannot clear the bar below this
        # Fails OPEN during warm-up so an established teacher is never penalised for
        # the first few steps; prior_beta already bounds the vacuum branch, which is
        # the larger channel (56-61% of rows under a skill teacher vs 6-10% under peer).
        self._opsd_gate_val = 1.0
        # ---- c_mode='corr': the trust level is ESTIMATED, not set ----
        # a_reliability (u) = Var(T)/Var(A^S): how much of the environment's own step
        # advantage is signal rather than sampling noise. It is a property of the
        # BACKBONE -- rollout count, env stochasticity -- not of the teacher, which is
        # exactly why it can stay fixed while c_prior could not: at one c_prior the
        # measured dose was 6-10% of rows under a peer teacher and 56-61% under a skill
        # teacher, a 5.6-9x swing driven purely by which teacher was plugged in.
        self.rlsd_a_reliability = float(rlsd_cfg.get("a_reliability", 0.5))
        if not (0.0 < self.rlsd_a_reliability <= 1.0):
            raise ValueError(f"rlsd a_reliability must be in (0,1], got {self.rlsd_a_reliability}")
        self.rlsd_rho_decay = float(rlsd_cfg.get("rho_decay", 0.9))
        self.rlsd_rho_z = float(rlsd_cfg.get("rho_shrink_z", 2.33))
        # None => step 1 runs at rho_hat=0, i.e. `decoupled`. Warming up teacher-OFF is
        # the safe direction: the failure this replaces was a teacher switched ON by a
        # weighting rule that had never checked whether it correlated with anything.
        self._rho_ema = None
        self._opsd_rho_n = 0
        self._opsd_rho_pos = 0
        self.rlsd_add_clip = rlsd_cfg.get("add_clip", 2.0)
        self.rlsd_center_delta = rlsd_cfg.get("center_delta", True)
        # Where the privileged information comes from. This is the ONLY thing that
        # differs between the skill arm and the no-skill (StepOPSD) arm -- the gap
        # Delta, the form, and the GiGPO advantage are all downstream of it, so an
        # ablation over this flag holds everything else fixed.
        #   skill  static retrieved skill text (SDAR/RLSD/AgentOPSD lineage)
        #   peer   the first successful peer trajectory in the same rollout group
        #          (StepOPSD). Defined only on groups that contain a success.
        self.rlsd_teacher = rlsd_cfg.get("teacher", "skill")
        if self.rlsd_teacher not in ("skill", "peer"):
            raise ValueError(f"unknown rlsd teacher {self.rlsd_teacher!r}; expected skill|peer")
        self.rlsd_peer_success_threshold = rlsd_cfg.get("peer_success_threshold", 1.0)
        self.rlsd_peer_max_steps = rlsd_cfg.get("peer_max_steps", 0)
        # Search-only ablation: hand the teacher the peer's <search> queries but not
        # its <answer>. Default False, i.e. the teacher gets the full successful
        # trajectory exactly as it does on ALFWorld/WebShop. See
        # build_peer_hindsight_prefixes' docstring for why this is an ablation and
        # not a default.
        self.rlsd_peer_hide_answer = bool(rlsd_cfg.get("peer_hide_answer", False))
        # ---- teacher role routing ----
        # 'estimate'  the teacher provides a second estimator of the same latent
        #             step advantage (correct for peer/hindsight). Routes through
        #             compute_opsd_step_tier into fused_S -> w_t.
        # 'prior'     the teacher is a static regulariser (skill library). The
        #             advantage path is HARD-CLOSED on vacuum rows (c=1 there) so
        #             a prior can never inject advantage on all-fail groups, and
        #             an ADDITIONAL gated-KL loss is applied to the actor with
        #             sdar_loss_coef. The advantage path's arbitration branch is
        #             left to rho_hat/gate, but under skill the corr path will
        #             push c* -> 1 so A^OPSD contributes ~0 there too.
        # 'auto' (default) peer->estimate, skill->prior.
        self.rlsd_teacher_role = str(rlsd_cfg.get("teacher_role", "auto"))
        if self.rlsd_teacher_role == "auto":
            self.rlsd_teacher_role = "estimate" if self.rlsd_teacher == "peer" else "prior"
        if self.rlsd_teacher_role not in ("estimate", "prior"):
            raise ValueError(
                f"unknown rlsd teacher_role {self.rlsd_teacher_role!r}; "
                f"expected auto|estimate|prior")
        self._prior_teacher = (self.rlsd_teacher_role == "prior")
        # KL-regularisation knobs for teacher_role='prior'. Gate mode mirrors
        # compute_sdar_loss; coef is deliberately tiny to match SDAR's sdar_coef=0.01.
        self.sdar_gate_beta = float(rlsd_cfg.get("sdar_gate_beta", 5.0))
        self.sdar_gate_mode = str(rlsd_cfg.get("sdar_gate_mode", "topq"))
        self.sdar_gate_topq = float(rlsd_cfg.get("sdar_gate_topq", 0.10))
        self.sdar_loss_coef = float(rlsd_cfg.get("sdar_loss_coef", 0.01))
        if self._prior_teacher and self.rlsd_form != "fusion":
            raise ValueError(
                f"teacher_role='prior' requires rlsd.form='fusion' (so c_routing "
                f"and the SDAR KL coexist); got form={self.rlsd_form!r}")
        print(f"[RLSD] teacher={self.rlsd_teacher} role={self.rlsd_teacher_role} "
              f"-> advantage-path prior_teacher={self._prior_teacher}, "
              f"sdar_coef={self.sdar_loss_coef} gate={self.sdar_gate_mode}")

    def _get_rlsd_lambda(self, step: int) -> float:
        """Linearly decay λ from rlsd_lambda_init to 0 over warmdown_steps."""
        if step >= self.rlsd_lambda_warmdown_steps:
            return 0.0
        return self.rlsd_lambda_init * (1.0 - step / self.rlsd_lambda_warmdown_steps)

    def _rho_corr_used(self):
        """Shrunk, non-negative rho for c_mode='corr'. None-safe: returns 0.0 before the
        first measurement, which makes step 1 run as `decoupled`."""
        if self._rho_ema is None:
            return 0.0
        _n = max(int(getattr(self, "_rho_n_row", 0)), 1) * max(1.0 / max(1.0 - self.rlsd_rho_decay, 1e-9), 1.0)
        _se = 1.0 / max(_n ** 0.5, 1e-9)
        return max(0.0, float(self._rho_ema) - self.rlsd_rho_z * _se)

    def _c_star(self):
        """The derived global trust level, logged every step so the dose is auditable.
        c* == 1 means the teacher is contributing nothing and fusion == decoupled."""
        u = float(min(max(self.rlsd_a_reliability, 1e-3), 1.0))
        r2 = min(min(max(self._rho_corr_used(), 0.0), 1.0) ** 2, u * (1.0 - 1e-6))
        den = u * u - 2.0 * u * r2 + r2
        return 1.0 if den <= 0 else min(max(u * (u - r2) / den, 0.0), 1.0)

    def fit(self):
        """
        The training loop of RLSD, extending the standard PPO/GRPO loop
        with teacher forward pass and token-level advantage computation.
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="RLSD Training")
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "env_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("env_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.traj_collector.multi_turn_loop(
                            gen_batch=gen_batch,
                            actor_rollout_wg=self.actor_rollout_wg,
                            envs=self.envs,
                            is_train=True,
                        )

                    del batch
                    batch = gen_batch_output

                    # GiGPO needs per-step discounted returns computed from the raw
                    # rollout, before adjust_batch/balance_batch reorder rows. The
                    # upstream RayPPOTrainer.fit does this at the same point; the RLSD
                    # fit() is a fork that never carried it over, so adv_estimator=gigpo
                    # used to die in compute_advantage on a missing batch['step_rewards'].
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.GiGPO:
                        step_rewards_tensor = core_gigpo.compute_step_discounted_returns(
                            batch=batch,
                            gamma=self.config.algorithm.gamma,
                        )
                        batch.batch['step_rewards'] = step_rewards_tensor

                    batch = adjust_batch(self.config, batch)
                    batch.batch["response_mask"] = compute_response_mask(batch)

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("reward", timing_raw):
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # Compute student log probs (old_log_probs)
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    # ---- RLSD: Teacher forward pass ----
                    with _timer("teacher_forward", timing_raw):
                        teacher_log_probs = self._compute_teacher_log_probs(batch)
                        batch.batch["teacher_log_probs"] = teacher_log_probs

                    if self.use_reference_policy:
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        if self.config.actor_rollout_ref.actor.get('use_invalid_action_penalty', True):
                            batch, invalid_metrics = apply_invalid_action_penalty(
                                batch,
                                invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
                            )
                            metrics.update(invalid_metrics)

                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute standard GRPO sequence-level advantages
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            use_pf_ppo=self.config.algorithm.use_pf_ppo,
                            pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                            pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                            step_advantage_w=self.config.algorithm.gigpo.step_advantage_w,
                            gigpo_mode=self.config.algorithm.gigpo.mode,
                            gigpo_enable_similarity=self.config.algorithm.gigpo.enable_similarity,
                            gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
                        )

                        # Per-row episode/step advantage, carried out of the GiGPO block so
                        # the RLSD block below can compare the step tier against the teacher
                        # gap. None whenever the estimator is not GiGPO (no step tier exists).
                        _ep_row = _st_row = None
                        _ep_adv = _st_adv_w = None
                        # form='fusion' only: the teacher-derived step estimate and the
                        # reliability weight that arbitrates between it and A^S.
                        _opsd_row = _c_k = None

                        if self.config.algorithm.adv_estimator == AdvantageEstimator.GiGPO:
                            # GiGPO's step advantage is a *relative* quantity inside an
                            # anchor-state group. An anchor state visited by only one
                            # trajectory has no sibling to be compared against, so its
                            # step advantage carries no information. singleton_frac is
                            # therefore the number that says whether the step level is
                            # doing anything at all on this env config.
                            step_group_uids = core_gigpo.build_step_group(
                                batch.non_tensor_batch['anchor_obs'],
                                batch.non_tensor_batch['uid'],
                                self.config.algorithm.gigpo.enable_similarity,
                                self.config.algorithm.gigpo.similarity_thresh,
                            )
                            _, group_counts = np.unique(step_group_uids, return_counts=True)
                            metrics["gigpo/n_step_groups"] = float(len(group_counts))
                            metrics["gigpo/step_group_size_mean"] = float(group_counts.mean())
                            metrics["gigpo/step_group_size_max"] = float(group_counts.max())
                            metrics["gigpo/singleton_frac"] = float((group_counts == 1).sum() / group_counts.sum())

                            # A^GiGPO = A^epi + w * A^step, and compute_gigpo_outcome_advantage
                            # only hands back the sum. Recomputing the two halves is cheap
                            # (reductions over (bs,) plus a tile) and answers two things the
                            # sum hides:
                            #   - whether step_advantage_w=1.0 leaves the step term dominant
                            #     or negligible next to the episode term;
                            #   - which of the two is responsible when the joint advantage
                            #     comes out exactly 0. A singleton step group yields
                            #     A^step == 0 exactly (step_norm_reward centres a group of
                            #     one on itself), and a uid group whose 8 trajectories all
                            #     score identically yields A^epi == 0 exactly.
                            _remove_std = self.config.algorithm.gigpo.mode == "mean_norm"
                            _rm = batch.batch["response_mask"]
                            _ep_adv = core_gigpo.episode_norm_reward(
                                batch.batch["token_level_rewards"], _rm,
                                batch.non_tensor_batch["uid"], batch.non_tensor_batch["traj_uid"],
                                1e-6, _remove_std,
                            )
                            _st_adv = core_gigpo.step_norm_reward(
                                batch.batch["step_rewards"], _rm, step_group_uids, 1e-6, _remove_std,
                            )
                            _ri = torch.arange(_rm.size(0), device=_rm.device)
                            _fv = _rm.int().argmax(dim=-1)
                            metrics["gigpo/episode_adv_absmean"] = masked_mean(_ep_adv.abs(), _rm).item()
                            metrics["gigpo/step_adv_absmean"] = masked_mean(_st_adv.abs(), _rm).item()
                            metrics["gigpo/episode_adv_zero_row_frac"] = (_ep_adv[_ri, _fv] == 0).float().mean().item()
                            metrics["gigpo/step_adv_zero_row_frac"] = (_st_adv[_ri, _fv] == 0).float().mean().item()

                            # ---- Do the two REWARD tiers actually disagree? ----
                            # A^GiGPO = A^epi + w*A^step is an unconditional sum: the step
                            # term is allowed to overturn the episode term's sign, and when
                            # it does, every downstream consumer that keys off sign(A) --
                            # including RLSD's own w_t = exp(sign(A)*Delta) -- silently
                            # switches which direction it calls "good". Nothing in GiGPO or
                            # in RLSD ever measures how often that happens.
                            #   step_overturn_frac  rows where the joint advantage ends up
                            #                       with the OPPOSITE sign to the episode
                            #                       advantage, i.e. the local evidence won.
                            #   ep_step_disagree_frac  rows where the two tiers point in
                            #                       opposite directions at all (whether or
                            #                       not the step term is strong enough to
                            #                       win). This is the raw conflict rate; the
                            #                       overturn rate is the subset where it has
                            #                       consequences.
                            # Both are restricted to rows where BOTH tiers are non-zero --
                            # a zero tier is an absence of evidence, not a disagreement.
                            _ep_row = _ep_adv[_ri, _fv]
                            _st_adv_w = _st_adv * self.config.algorithm.gigpo.step_advantage_w
                            _st_row = _st_adv_w[_ri, _fv]
                            _both_nz = (_ep_row != 0) & (_st_row != 0)
                            metrics["gigpo/both_tiers_nonzero_frac"] = _both_nz.float().mean().item()
                            if bool(_both_nz.any()):
                                _e, _s = _ep_row[_both_nz], _st_row[_both_nz]
                                metrics["gigpo/ep_step_disagree_frac"] = (
                                    torch.sign(_e) != torch.sign(_s)).float().mean().item()
                                metrics["gigpo/step_overturn_frac"] = (
                                    torch.sign(_e + _s) != torch.sign(_e)).float().mean().item()

                            # ---- OFFLINE DIAGNOSTIC DUMP (opt-in, env var) ----
                            # Writes the per-row quantities needed to test, offline and
                            # without any training, whether a SKILL-LEVEL grouping of the
                            # same step returns produces an estimate A^K that correlates
                            # with the anchor-state estimate A^S. That correlation is the
                            # precondition for arbitrating them; the log-prob teacher fails
                            # it by construction (rho_used == 0.000 measured over 3 arms).
                            # Off unless RLSD_DUMP_DIR is set, so published arms are unchanged.
                            _dump_dir = os.environ.get("RLSD_DUMP_DIR", "")
                            if _dump_dir:
                                import pickle as _pk
                                os.makedirs(_dump_dir, exist_ok=True)
                                _ntb = batch.non_tensor_batch
                                _sr = batch.batch["step_rewards"]
                                _sr_row = _sr[_ri, _fv] if _sr.dim() > 1 else _sr
                                _tlr = batch.batch["token_level_rewards"]

                                def _nt(key, default=None):
                                    v = _ntb.get(key, None)
                                    if v is None:
                                        return [default] * len(step_group_uids)
                                    return [v[i] for i in range(len(v))]

                                _resp = self.tokenizer.batch_decode(
                                    batch.batch["responses"], skip_special_tokens=True)
                                # 600-char head truncation drops the <action> tag on the
                                # ~35% of rows whose <think> block is long, and those rows
                                # are systematically longer (168 vs 115 tok) -- dropping
                                # them would bias any within-group test. Extract the action
                                # span at dump time instead, keeping a bounded tail as a
                                # fallback so nothing is silently lost.
                                def _act(s):
                                    m = re.search(r"<action>(.*?)</action>", s, re.S)
                                    if m:
                                        return m.group(1).strip()[:200]
                                    m = re.search(r"<action>(.*)", s, re.S)
                                    if m:
                                        return m.group(1).strip()[:200]
                                    return s[-200:].strip()

                                _rec = {
                                    "uid": [str(x) for x in _nt("uid")],
                                    "traj_uid": [str(x) for x in _nt("traj_uid")],
                                    "turn_step": _nt("turn_step", -1),
                                    "gamefile": [str(x) for x in _nt("gamefile", "")],
                                    "step_group_uid": [str(x) for x in step_group_uids],
                                    "anchor_obs": [str(x)[:2000] for x in _nt("anchor_obs", "")],
                                    "action_text": [_act(s) for s in _resp],
                                    "resp_tail": [s[-300:] for s in _resp],
                                    "has_action_tag": [bool(re.search(r"<action>.*?</action>", s, re.S)) for s in _resp],
                                    "A_S_row": _st_row.float().cpu().numpy(),
                                    "A_E_row": _ep_row.float().cpu().numpy(),
                                    "step_return": _sr_row.float().cpu().numpy(),
                                    "episode_return": _tlr.sum(dim=-1).float().cpu().numpy(),
                                    "resp_len": _rm.sum(dim=-1).float().cpu().numpy(),
                                    "is_action_valid": _nt("is_action_valid", True),
                                    "active_masks": _nt("active_masks", True),
                                    "step_advantage_w": float(self.config.algorithm.gigpo.step_advantage_w),
                                    "gigpo_mode": str(self.config.algorithm.gigpo.mode),
                                    "global_step": int(self.global_steps),
                                }
                                _fp = os.path.join(_dump_dir, f"rollout_step{self.global_steps}.pkl")
                                with open(_fp, "wb") as _f:
                                    _pk.dump(_rec, _f, protocol=4)
                                print(f"[RLSD-DUMP] wrote {_fp} rows={len(_rec['uid'])}", flush=True)

                            # ---- form=fusion: build the SECOND step-tier estimate ----
                            # A^OPSD asks the same question A^S asks -- "was this step
                            # better than its siblings at the same anchor state?" -- from
                            # the teacher gap instead of the environment return, centred in
                            # the same anchor group and scaled to the same spread. c_k then
                            # arbitrates by inverse variance. See compute_opsd_step_tier.
                            if self.rlsd_form == "fusion":
                                _opsd_row, _c_eff, _cdiag = compute_opsd_step_tier(
                                    student_log_probs=batch.batch["old_log_probs"],
                                    teacher_log_probs=batch.batch["teacher_log_probs"],
                                    response_mask=_rm,
                                    step_group_uids=step_group_uids,
                                    step_rewards=batch.batch["step_rewards"],
                                    step_adv_row=_st_row,
                                    c_prior=(1.0 if self.rlsd_c_mode == "corr" else self.rlsd_c_prior),
                                    prior_beta=(1.0 if self.rlsd_c_mode == "corr" else self.rlsd_prior_beta),
                                    gate=(1.0 if self.rlsd_c_mode == "corr" else self._opsd_gate_val),
                                    rho_hat=(self._rho_corr_used() if self.rlsd_c_mode == "corr" else None),
                                    a_reliability=self.rlsd_a_reliability,
                                    prior_teacher=self._prior_teacher,
                                )
                                _c_invvar = _cdiag["c_invvar"]
                                if self.rlsd_c_mode == "const":
                                    # const bypasses role separation entirely -- it IS the
                                    # positive control and must stay a single flat weight.
                                    _c_k = torch.full_like(_c_invvar, self.rlsd_c_const)
                                elif self.rlsd_c_mode == "corr":
                                    # c_invvar已在 compute_opsd_step_tier 内按 rho_hat 修正;
                                    # c_eff 与它相同(gate/prior_beta 在 corr 下被强制为惰性)
                                    _c_k = _c_eff
                                elif self.rlsd_c_mode == "invvar_raw":
                                    _c_k = _cdiag["c_raw"]
                                else:
                                    _c_k = _c_eff
                                _fused = _c_k * _st_row + (1.0 - _c_k) * _opsd_row

                                # What c_k actually does. If it collapses to ~0 or ~1
                                # everywhere the fusion is decorative and the arm reduces
                                # to one of its endpoints -- c_frac_mid is the number that
                                # says arbitration is happening on real data.
                                metrics["rlsd/c_mean"] = _c_k.mean().item()
                                metrics["rlsd/c_std"] = _c_k.std().item()
                                metrics["rlsd/c_frac_env"] = (_c_k >= 0.9).float().mean().item()
                                metrics["rlsd/c_frac_teacher"] = (_c_k <= 0.1).float().mean().item()
                                metrics["rlsd/c_frac_mid"] = (
                                    (_c_k > 0.1) & (_c_k < 0.9)).float().mean().item()
                                # Logged even under c_mode=const so the const arms still
                                # report what inverse variance would have chosen.
                                metrics["rlsd/c_invvar_mean"] = _c_invvar.mean().item()
                                # The naive absolute weight, always logged: it is the
                                # ablation's headline number and the reason for the median
                                # normalisation. prec_ratio is the raw unit mismatch that
                                # drives it -- expect it far from 1 in either direction.
                                metrics["rlsd/c_raw_mean"] = _cdiag["c_raw"].mean().item()
                                _pS, _pO = _cdiag["prec_S"], _cdiag["prec_O"]
                                _hs, _ht = _cdiag["has_step"], _cdiag["has_teacher"]
                                if bool(_hs.any()) and bool(_ht.any()):
                                    metrics["rlsd/prec_ratio_median"] = float(
                                        _pS[_hs].median() / _pO[_ht].median().clamp(min=1e-12))
                                metrics["rlsd/opsd_scale"] = float(_cdiag["scale"])
                                metrics["rlsd/opsd_absmean"] = _opsd_row.abs().mean().item()
                                metrics["rlsd/fused_absmean"] = _fused.abs().mean().item()
                                metrics["rlsd/n_anchor_mean"] = _cdiag["n_anchor"].mean().item()

                                # THE falsifiable prediction of this form: A^S is exactly 0
                                # on every anchor group with no return spread (measured 0.56
                                # of rows at step 60 of the decoupled run), and there the
                                # teacher tier should fill in rather than the step tier
                                # going silent. rescue_frac is that fill-in rate; if it is
                                # ~0 the fusion buys nothing over decoupled.
                                _st_dead = (_st_row == 0)
                                metrics["rlsd/fused_step_zero_row_frac"] = (
                                    _fused == 0).float().mean().item()
                                metrics["rlsd/opsd_rescue_frac"] = (
                                    _st_dead & (_fused != 0)).float().mean().item()
                                # THE DENOMINATOR, measured rather than reconstructed by
                                # hand afterwards. A row can only be rescued if it HAS a
                                # teacher, so the ceiling is not step_adv_zero_row_frac:
                                # it is the taught-but-no-step-evidence mass. With a peer
                                # teacher that is small (~0.09), because the teacher is
                                # absent on exactly the all-fail groups where A^S is also
                                # 0 -- the same co-location that motivates the form also
                                # caps it. Quote rescue_frac against THIS, never against
                                # step_adv_zero_row_frac.
                                metrics["rlsd/opsd_rescue_ceiling"] = (
                                    _cdiag["has_teacher"] & ~_cdiag["has_step"]).float().mean().item()

                                # The section-10 motivating measurement, restated on the two
                                # tiers that are actually fused. Restricted to rows where
                                # BOTH estimates exist -- absence is not disagreement.
                                _both_ev = _cdiag["has_step"] & _cdiag["has_teacher"]
                                metrics["rlsd/both_evidence_frac"] = _both_ev.float().mean().item()
                                if bool(_both_ev.any()):
                                    # c_prior calibration check on real data: both
                                    # precisions are normalised on THIS population, so at
                                    # c_prior=1 the median has to sit at ~0.5. Far from it
                                    # means the normalising population is wrong again.
                                    metrics["rlsd/c_median_both"] = float(_c_k[_both_ev].median())
                                    metrics["rlsd/c_frac_mid_of_both"] = float(
                                        ((_c_k[_both_ev] > 0.1) & (_c_k[_both_ev] < 0.9)).float().mean())
                                    _a, _b = _st_row[_both_ev], _opsd_row[_both_ev]
                                    metrics["rlsd/opsd_step_conflict_frac"] = (
                                        torch.sign(_a) != torch.sign(_b)).float().mean().item()
                                    if _a.numel() > 1 and float(_a.std()) > 0 and float(_b.std()) > 0:
                                        _rho = float(
                                            ((_a - _a.mean()) * (_b - _b.mean())).mean()
                                            / (_a.std(unbiased=False) * _b.std(unbiased=False)))
                                        metrics["rlsd/opsd_step_corr"] = _rho
                                        # ---- update the ESTIMATED trust level (c_mode='corr') ----
                                        # Shrink before use. The naive per-step rho has
                                        # se ~ 1/sqrt(n_row); the EMA over 1/(1-decay)
                                        # effective steps has se ~ 1/sqrt(n_row*n_eff).
                                        # rho is autocorrelated across steps so n_eff is
                                        # optimistic, but the resulting error points toward
                                        # trusting the teacher MORE, so the shrinkage is
                                        # reported, not relied on, as the safety margin.
                                        _d = self.rlsd_rho_decay
                                        self._rho_ema = (_rho if self._rho_ema is None
                                                         else _d * self._rho_ema + (1.0 - _d) * _rho)
                                        self._rho_n_row = int(_a.numel())
                                        # ---- update the validity test for the NEXT step ----
                                        # Sign test on rho across steps. Deliberately a TEST
                                        # and not a scaling: a linear g(rho) would cut the
                                        # peer teacher's share 12.9% -> 4.9% and silently
                                        # invalidate three already-published numbers, whereas
                                        # a test leaves an established teacher untouched
                                        # (peer: 100% of steps rho>0, z ~ +12 => g == 1) and
                                        # rejects an orthogonal one outright (skill: 20-40%,
                                        # z ~ -3.8 => g == 0).
                                        self._opsd_rho_n += 1
                                        self._opsd_rho_pos += 1 if _rho > 0.0 else 0
                                        _n = self._opsd_rho_n
                                        _z = ((self._opsd_rho_pos - 0.5 * _n)
                                              / max((0.25 * _n) ** 0.5, 1e-9))
                                        metrics["rlsd/opsd_rho_z"] = float(_z)
                                        metrics["rlsd/opsd_rho_pos_frac"] = float(self._opsd_rho_pos) / max(_n, 1)
                                        if self.rlsd_opsd_gate == "signtest" and _n >= self.rlsd_opsd_gate_min_n:
                                            self._opsd_gate_val = 1.0 if _z > self.rlsd_opsd_gate_z else 0.0

                                # ---- what the role separation actually did ----
                                # A^O's total dose is mean(1 - c_k). Splitting it by WHICH
                                # branch delivered it is the number that was invisible
                                # before: at identical settings the vacuum branch carried
                                # 58-79% of the dose in every arm, and its row count swings
                                # 6% (peer, coverage .10-.25) -> 61% (skill, coverage 1.00).
                                # Quote these two, never the total alone.
                                _dose = (1.0 - _c_k)
                                _vac, _bth = _cdiag["vacuum"].to(_dose.dtype), _cdiag["both_ev"].to(_dose.dtype)
                                metrics["rlsd/opsd_dose_total"] = _dose.mean().item()
                                metrics["rlsd/opsd_dose_vacuum"] = (_dose * _vac).mean().item()
                                metrics["rlsd/opsd_dose_arbitrate"] = (_dose * _bth).mean().item()
                                metrics["rlsd/opsd_gate"] = float(self._opsd_gate_val)
                                if self._rho_ema is not None:
                                    metrics["rlsd/rho_ema"] = float(self._rho_ema)
                                metrics["rlsd/rho_used"] = float(self._rho_corr_used() or 0.0)
                                metrics["rlsd/c_star"] = float(self._c_star())
                                metrics["rlsd/prior_beta"] = float(self.rlsd_prior_beta)

                        # ---- RLSD: Replace sequence-level advantage with token-level advantage ----
                        seq_advantages = batch.batch["advantages"]
                        student_log_probs = batch.batch["old_log_probs"]
                        teacher_log_probs = batch.batch["teacher_log_probs"]
                        response_mask = batch.batch["response_mask"]

                        current_lambda = self._get_rlsd_lambda(self.global_steps)
                        token_advantages = compute_rlsd_token_advantage(
                            seq_advantages=seq_advantages,
                            student_log_probs=student_log_probs,
                            teacher_log_probs=teacher_log_probs,
                            response_mask=response_mask,
                            rlsd_lambda=current_lambda,
                            rlsd_clip_eps=self.rlsd_clip_eps,
                            form=self.rlsd_form,
                            add_clip=self.rlsd_add_clip,
                            center_delta=self.rlsd_center_delta,
                            # None unless the estimator is GiGPO, which is exactly the
                            # condition under which form='decoupled' is well-defined --
                            # it needs the two tiers separately, not their sum.
                            episode_advantages=_ep_adv,
                            step_advantages=_st_adv_w,
                            # None unless form='fusion'. See compute_opsd_step_tier.
                            opsd_advantages=_opsd_row,
                            reliability=_c_k,
                        )

                        batch.batch["advantages"] = token_advantages

                        # Where the teacher is allowed to act at all. This is the price
                        # tag on 'decoupled': confining w to the step tier means the
                        # teacher is inert on every row whose anchor group had no return
                        # spread (A^S == 0), whereas 'mult' rides the joint advantage and
                        # only dies when BOTH tiers are zero. Measured on ALFWorld the
                        # gap is large -- ~70% inert vs ~19% -- so the decoupled arm buys
                        # teacher precision with teacher reach, and this number is how
                        # much reach it paid.
                        if self.rlsd_form == "fusion" and _c_k is not None:
                            # In 'fusion' the teacher reaches a row through TWO channels --
                            # the A^OPSD share of the fused tier, and the token weight w --
                            # and both need the fused tier to be non-zero. So the reach is
                            # the taught rows whose fused tier survived. Directly comparable
                            # to the decoupled number below (0.45 mean over 60 steps), and
                            # it should be HIGHER: fusion revives rows where A^S == 0.
                            metrics["rlsd/teacher_active_frac"] = (
                                _cdiag["has_teacher"] & (_fused != 0)).float().mean().item()
                        elif self.rlsd_form == "decoupled" and _st_row is not None:
                            metrics["rlsd/teacher_active_frac"] = (
                                _st_row != 0).float().mean().item()
                        elif _ep_row is not None and _st_row is not None:
                            metrics["rlsd/teacher_active_frac"] = (
                                seq_advantages[_ri, _fv] != 0).float().mean().item()

                        # Log RLSD-specific metrics
                        delta_t = (teacher_log_probs - student_log_probs) * response_mask
                        metrics["rlsd/teacher_student_gap_mean"] = masked_mean(delta_t, response_mask).item()
                        metrics["rlsd/teacher_student_gap_std"] = masked_mean(delta_t ** 2, response_mask).sqrt().item()
                        metrics["rlsd/lambda"] = current_lambda
                        metrics["rlsd/clip_eps"] = self.rlsd_clip_eps

                        # ---- Diagnostics for the token level ----
                        # Two things have to hold for a token-level reweighting to be
                        # more than cosmetic:
                        #   (a) Delta must vary WITHIN a row. Each row here is one env
                        #       step, so within-row spread is the only spread the token
                        #       level can redistribute over; if it is ~0 relative to the
                        #       between-row spread, w_t just rescales whole steps.
                        #   (b) w_t must not be pinned to the clip boundary. Anything
                        #       saturated collapses to one of two constants and loses
                        #       its per-token resolution.
                        with torch.no_grad():
                            d_mean = masked_mean(delta_t, response_mask)
                            metrics["rlsd/gap_std_true"] = masked_mean(
                                (delta_t - d_mean) ** 2, response_mask).sqrt().item()

                            # How many rows got ANY privileged information. 1.0 for the
                            # skill teacher by construction; for the peer teacher it is
                            # the fraction of rows in a failed trajectory whose group
                            # also contains a success, and it is the ceiling on how much
                            # of the batch the teacher tier can possibly touch.
                            metrics["rlsd/teacher_coverage"] = getattr(self, "_teacher_coverage", 1.0)
                            metrics["rlsd/delta_zero_row_frac"] = (
                                (delta_t.abs() * response_mask).sum(dim=-1) == 0
                            ).float().mean().item()
                            # Did the privileged text actually reach the teacher? The
                            # prompt is left-truncated, and the privileged text is on
                            # the left, so an over-long prompt eats it first.
                            for _k, _v in getattr(self, "_teacher_prefix_stats", {}).items():
                                metrics[f"rlsd/{_k}"] = _v

                            row_n = response_mask.sum(dim=-1)
                            row_mean = (delta_t * response_mask).sum(dim=-1) / row_n.clamp(min=1.0)
                            row_var = ((delta_t - row_mean.unsqueeze(-1)) ** 2 * response_mask).sum(dim=-1) / row_n.clamp(min=1.0)
                            multi_tok = row_n > 1
                            if bool(multi_tok.any()):
                                metrics["rlsd/gap_var_within_step"] = row_var[multi_tok].mean().item()
                                metrics["rlsd/gap_var_between_step"] = row_mean[multi_tok].var(unbiased=False).item()

                            # ---- Do the ENVIRONMENT and the TEACHER disagree? ----
                            # This is the load-bearing measurement for the whole
                            # "dual-evidence" framing, and no paper in this line reports it.
                            # Two independent estimates of whether step k was a good move:
                            #   A^step  local return-to-go vs siblings at the same anchor
                            #           state (environment evidence, GiGPO)
                            #   Delta   how much MORE likely the privileged teacher was to
                            #           emit these tokens than the student (teacher
                            #           evidence, RLSD/SDAR/StepOPSD)
                            # Delta is row-centred by the batch mean first: Delta carries a
                            # global offset (~-0.13 here) that is pure bias, and leaving it
                            # in would make almost every row read "teacher disapproves".
                            # Three outcomes, three different papers:
                            #   corr >> 0  the two are redundant -- the teacher is just
                            #              re-encoding the return, and there is nothing to
                            #              arbitrate.
                            #   corr ~ 0   they are complementary. This is what AgentOPSD
                            #              asserts ("complementary signal sources") without
                            #              measuring, and it is the premise the fusion rests
                            #              on.
                            #   conflict_frac large  they routinely contradict each other,
                            #              and an unconditional sum/product is picking a
                            #              winner arbitrarily on that fraction of the batch.
                            # If evidence_conflict_frac comes back tiny (<5%), the conflict
                            # story has no empirical support and should be dropped.
                            if _st_row is not None:
                                _d_row = row_mean - d_mean                  # centred per-row Delta
                                _sel = _st_row != 0                         # step evidence exists
                                metrics["rlsd/step_evidence_frac"] = _sel.float().mean().item()
                                if int(_sel.sum()) > 1:
                                    _sv, _dv = _st_row[_sel], _d_row[_sel]
                                    _sp, _dp = _sv > 0, _dv > 0
                                    metrics["rlsd/evidence_conflict_frac"] = (_sp != _dp).float().mean().item()
                                    metrics["rlsd/quad_env_pos_teacher_pos"] = (_sp & _dp).float().mean().item()
                                    metrics["rlsd/quad_env_pos_teacher_neg"] = (_sp & ~_dp).float().mean().item()
                                    metrics["rlsd/quad_env_neg_teacher_pos"] = (~_sp & _dp).float().mean().item()
                                    metrics["rlsd/quad_env_neg_teacher_neg"] = (~_sp & ~_dp).float().mean().item()
                                    metrics["rlsd/step_teacher_corr"] = _pearson(_sv, _dv)
                                    if _ep_row is not None:
                                        # Contrast: if Delta correlates with the EPISODE tier
                                        # but not the step tier, the teacher is carrying
                                        # outcome information, not local information.
                                        metrics["rlsd/ep_teacher_corr"] = _pearson(_ep_row[_sel], _dv)

                                # The same three numbers restricted to rows the teacher
                                # ACTUALLY SAW privileged information on. Mandatory for the
                                # peer teacher and a no-op for the skill teacher: peer
                                # hindsight leaves ~78% of rows with Delta identically 0,
                                # and a zero row, once the batch mean is subtracted, reads
                                # as a constant "teacher approves" (-d_mean > 0 whenever
                                # the mean gap is negative, which it always is). Those
                                # untaught rows are not a random subset -- every successful
                                # trajectory is in there, and successes are exactly where
                                # A^step is positive -- so leaving them in manufactures a
                                # correlation out of the coverage pattern alone. Measured
                                # on step 1: 0.522 unrestricted. This is what to trust.
                                _taught = getattr(self, "_teacher_taught", None)
                                if _taught is not None:
                                    _tb = _taught.to(_st_row.device) > 0
                                    _tmask = response_mask * _tb.unsqueeze(-1).to(response_mask.dtype)
                                    if float(_tmask.sum()) > 0:
                                        _tm = masked_mean(delta_t, _tmask)
                                        metrics["rlsd/gap_mean_taught"] = _tm.item()
                                        metrics["rlsd/gap_std_taught"] = masked_mean(
                                            (delta_t - _tm) ** 2, _tmask).sqrt().item()
                                    _selt = _sel & _tb
                                    metrics["rlsd/taught_and_step_evidence_frac"] = _selt.float().mean().item()
                                    if int(_selt.sum()) > 1:
                                        _svt = _st_row[_selt]
                                        # centre Delta WITHIN the taught rows, not against
                                        # a mean that is dominated by the zeros
                                        _dvt = row_mean[_selt] - row_mean[_selt].mean()
                                        _spt, _dpt = _svt > 0, _dvt > 0
                                        metrics["rlsd/evidence_conflict_frac_taught"] = (
                                            _spt != _dpt).float().mean().item()
                                        metrics["rlsd/step_teacher_corr_taught"] = _pearson(_svt, _dvt)
                                        if _ep_row is not None:
                                            metrics["rlsd/ep_teacher_corr_taught"] = _pearson(
                                                _ep_row[_selt], _dvt)

                            first_valid = response_mask.int().argmax(dim=-1)
                            row_idx = torch.arange(seq_advantages.size(0), device=seq_advantages.device)
                            A_row = seq_advantages[row_idx, first_valid]
                            sign_A = torch.sign(A_row).unsqueeze(-1)
                            w_raw = torch.exp(sign_A * delta_t)
                            saturated = ((w_raw <= 1.0 - self.rlsd_clip_eps) |
                                         (w_raw >= 1.0 + self.rlsd_clip_eps)).float()
                            metrics["rlsd/clip_saturation_frac"] = masked_mean(saturated, response_mask).item()

                            # A row whose advantage is exactly 0 gets sign(A)=0 -> w=1, so it
                            # can never register as saturated AND it receives no token credit
                            # at all under the multiplicative form A * w: the token tier is
                            # simply inert there. Those rows therefore dilute the plain
                            # saturation number above. The _nonzero variant is the one that
                            # actually says how much per-token resolution survives the clip.
                            nz_row = (A_row != 0)
                            nz_mask = response_mask * nz_row.unsqueeze(-1).to(response_mask.dtype)
                            metrics["rlsd/zero_adv_row_frac"] = (~nz_row).float().mean().item()
                            metrics["rlsd/zero_adv_token_frac"] = 1.0 - masked_mean(
                                nz_row.unsqueeze(-1).to(response_mask.dtype).expand_as(response_mask),
                                response_mask).item()
                            if float(nz_mask.sum()) > 0:
                                metrics["rlsd/clip_saturation_frac_nonzero"] = masked_mean(saturated, nz_mask).item()

                            # ---- Does the teacher gap survive as a usable signal? ----
                            # These three decide whether `form` was the right choice, and
                            # they are the numbers the ablation table is built from:
                            #   dead_token_frac  the multiplicative form's cost. Fraction of
                            #                    tokens with Ahat == 0, i.e. receiving no
                            #                    gradient at ANY tier. Under mult this equals
                            #                    zero_adv_token_frac; add/hybrid should drive
                            #                    it to ~0 by construction.
                            #   sign_flip_frac   the additive form's cost, and the reason mult
                            #                    was the default: add can make Ahat disagree
                            #                    in sign with the outcome credit A, i.e. punish
                            #                    a token that belonged to a winning rollout
                            #                    purely because the privileged teacher liked it
                            #                    less. mult cannot (w > 0). If this is large the
                            #                    teacher is overriding the reward, not refining it.
                            #   add_to_base_ratio  scale sanity. |lambda*Delta_c| / |A|. Near 0
                            #                    means the additive term is decoration; >~1 means
                            #                    the teacher, not the return, is driving the update.
                            token_adv = batch.batch["advantages"]
                            metrics["rlsd/dead_token_frac"] = 1.0 - masked_mean(
                                (token_adv != 0).to(response_mask.dtype), response_mask).item()
                            flipped = ((torch.sign(token_adv) * torch.sign(A_row).unsqueeze(-1)) < 0
                                       ).to(response_mask.dtype)
                            if float(nz_mask.sum()) > 0:
                                metrics["rlsd/sign_flip_frac"] = masked_mean(flipped, nz_mask).item()

                            if self.rlsd_form in ("add", "hybrid"):
                                delta_c = centered_delta(
                                    student_log_probs, teacher_log_probs, response_mask,
                                    add_clip=self.rlsd_add_clip, center=self.rlsd_center_delta,
                                )
                                add_absmean = masked_mean(
                                    (current_lambda * delta_c).abs(), response_mask).item()
                                base_absmean = masked_mean(
                                    A_row.unsqueeze(-1).abs().expand_as(response_mask),
                                    response_mask).item()
                                metrics["rlsd/add_term_absmean"] = add_absmean
                                metrics["rlsd/base_adv_absmean"] = base_absmean
                                metrics["rlsd/add_to_base_ratio"] = add_absmean / max(base_absmean, 1e-8)
                                metrics["rlsd/add_clip_frac"] = masked_mean(
                                    (delta_c.abs() >= self.rlsd_add_clip - 1e-6).to(response_mask.dtype),
                                    response_mask).item()
                                # Of the rows the multiplicative form would have zeroed out,
                                # how many now carry gradient. This is the whole point of add.
                                zero_mask = response_mask * (~nz_row).unsqueeze(-1).to(response_mask.dtype)
                                if float(zero_mask.sum()) > 0:
                                    metrics["rlsd/rescued_token_frac"] = masked_mean(
                                        (token_adv != 0).to(response_mask.dtype), zero_mask).item()

                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            # For 'prior' teachers (static skill library) the SDAR
                            # gated-KL is added inside dp_actor alongside the policy
                            # gradient. Routing these flags through meta_info keeps
                            # dp_actor's existing plumbing (use_sdar_loss/sdar_coef/
                            # gate_beta) usable without a new API surface.
                            if self._prior_teacher:
                                batch.meta_info["algorithm.use_sdar_loss"] = True
                                batch.meta_info["algorithm.sdar_loss_coef"] = self.sdar_loss_coef
                                batch.meta_info["algorithm.sdar_gate_beta"] = self.sdar_gate_beta
                                batch.meta_info["algorithm.sdar_gate_mode"] = self.sdar_gate_mode
                                batch.meta_info["algorithm.sdar_gate_topq"] = self.sdar_gate_topq
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    test_start_step = self.config.trainer.get("test_start_step", 0)
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or (self.global_steps >= test_start_step and self.global_steps % self.config.trainer.test_freq == 0)):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                metrics.update({
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                })
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

    def _compute_teacher_log_probs(self, batch: DataProto) -> torch.Tensor:
        """
        Compute teacher log probs by running a forward pass with privileged info
        prepended to the prompt. Uses the same model π_θ but conditioned on (x, r).

        Both teachers share this single forward pass; only the text of r differs
        (see self.rlsd_teacher). Coverage -- the fraction of rows that get any r
        at all -- is recorded here because it is 100% for the skill teacher and
        strictly less for the peer teacher, and that difference is the whole
        point of the comparison.
        """
        prefixes = None
        if self.rlsd_teacher == "peer":
            prefixes = build_peer_hindsight_prefixes(
                batch=batch,
                tokenizer=self.tokenizer,
                success_threshold=self.rlsd_peer_success_threshold,
                max_peer_steps=self.rlsd_peer_max_steps,
                hide_answer=self.rlsd_peer_hide_answer,
            )
            self._teacher_coverage = float(np.mean([1.0 if p else 0.0 for p in prefixes]))
            self._teacher_taught = torch.tensor(
                [1.0 if p else 0.0 for p in prefixes], dtype=torch.float32)
        else:
            self._teacher_coverage = 1.0
            self._teacher_taught = None

        teacher_batch = build_teacher_batch(
            batch=batch,
            skill_provider=self.skill_provider,
            tokenizer=self.tokenizer,
            max_prompt_length=self.config.data.max_prompt_length,
            truncation=self.config.data.get("truncation", "left"),
            prefixes=prefixes,
        )
        self._teacher_prefix_stats = teacher_batch.meta_info.get("prefix_stats", {})

        # Use the same actor to compute teacher log probs
        # Hand the teacher forward's cached blocks back before update_actor's
        # backward runs; see fsdp_workers.compute_log_prob. Numerically inert.
        teacher_batch.meta_info["rlsd_empty_cache_after"] = True
        teacher_batch.meta_info["rlsd_probe_tag"] = "teacher"
        teacher_output = self.actor_rollout_wg.compute_log_prob(teacher_batch)
        teacher_log_probs = teacher_output.batch["old_log_probs"]

        return teacher_log_probs
