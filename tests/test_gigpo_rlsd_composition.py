"""
Offline (CPU, no model, no env) check of the GiGPO x RLSD composition.

Exercises the exact call chain that RLSDRayTrainer.fit() now takes with
adv_estimator=gigpo, on a synthetic multi-turn batch, so shape/dtype/key bugs
surface in seconds instead of ~25 minutes into a real launch.

Layout: 3 tasks (uid) x 2 trajectories (traj_uid) x 3 steps = 18 rows.
anchor_obs is chosen so some states are revisited across sibling trajectories
(-> step group size 2+, step advantage is informative) and some are visited once
(-> singleton, step advantage carries no information).
taskC is an ALL-FAIL group: neither of its trajectories scores, so the group has
no outcome variance and both the episode and the step advantage are exactly 0
there. That is the regime the additive form exists for, and it is 33% of rows in
the real ALFWorld batches, so the fixture has to contain it.
"""
import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from gigpo import core_gigpo
from verl.trainer.ppo.rlsd_ray_trainer import (
    PEER_HINDSIGHT_HEADER,
    _pearson,
    build_peer_hindsight_prefixes,
)
from verl.trainer.ppo.rlsd_utils import centered_delta, compute_rlsd_token_advantage
from verl.utils.torch_functional import masked_mean

torch.manual_seed(0)
np.random.seed(0)

BS, RESP_LEN = 18, 8
CLIP_EPS = 0.2

uid = np.array(["taskA"] * 6 + ["taskB"] * 6 + ["taskC"] * 6, dtype=object)
traj_uid = np.array(["A0"] * 3 + ["A1"] * 3 + ["B0"] * 3 + ["B1"] * 3
                    + ["C0"] * 3 + ["C1"] * 3, dtype=object)
# A0 and A1 share their first two states, then diverge. B0/B1 share only the first.
anchor_obs = np.array(
    ["a_s0", "a_s1", "a_s2",
     "a_s0", "a_s1", "a_s9",
     "b_s0", "b_s1", "b_s2",
     "b_s0", "b_s7", "b_s8",
     "c_s0", "c_s1", "c_s2",
     "c_s0", "c_s1", "c_s3"], dtype=object)

# per-step env rewards; only the last step of a successful trajectory pays out
rewards = np.array([0., 0., 1.,
                    0., 0., 0.,
                    0., 0., 1.,
                    0., 0., 0.,
                    0., 0., 0.,
                    0., 0., 0.], dtype=object)
active_masks = np.array([True] * BS, dtype=object)

response_mask = torch.ones(BS, RESP_LEN)
response_mask[:, -2:] = 0.0          # ragged: last two positions are padding
response_mask[3, 1:] = 0.0           # one row with a single valid token
# episode-level reward, placed on the last valid token as verl does
token_level_rewards = torch.zeros(BS, RESP_LEN)
for i in range(BS):
    token_level_rewards[i, int(response_mask[i].sum()) - 1] = float(rewards[i])

batch = DataProto(
    batch=TensorDict({
        "input_ids": torch.zeros(BS, RESP_LEN, dtype=torch.long),
        "response_mask": response_mask,
        "token_level_rewards": token_level_rewards,
    }, batch_size=[BS]),
    non_tensor_batch={
        "rewards": rewards, "active_masks": active_masks,
        "traj_uid": traj_uid, "uid": uid, "anchor_obs": anchor_obs,
    },
)

# ---- 1. the block added to RLSDRayTrainer.fit() ----
step_rewards = core_gigpo.compute_step_discounted_returns(batch=batch, gamma=0.95)
print("step_rewards (discounted return per row):", np.round(step_rewards.numpy(), 3))
assert step_rewards.shape == (BS,), step_rewards.shape

# ---- 2. GiGPO advantage ----
adv, ret = core_gigpo.compute_gigpo_outcome_advantage(
    token_level_rewards=batch.batch["token_level_rewards"],
    step_rewards=step_rewards,
    response_mask=response_mask,
    anchor_obs=anchor_obs, index=uid, traj_index=traj_uid,
    step_advantage_w=1.0, mode="mean_norm",
)
assert adv.shape == (BS, RESP_LEN), adv.shape
# the property compute_rlsd_token_advantage relies on: one scalar per row
for i in range(BS):
    vals = adv[i][response_mask[i].bool()]
    assert torch.allclose(vals, vals[0].expand_as(vals)), f"row {i} not constant: {vals}"
print("GiGPO A per row:", np.round(adv[:, 0].numpy(), 3), "(constant across valid tokens: OK)")

# ---- 3. the step-group diagnostic added to fit() ----
step_group_uids = core_gigpo.build_step_group(anchor_obs, uid, False, 0.95)
_, counts = np.unique(step_group_uids, return_counts=True)
print(f"gigpo/n_step_groups        = {len(counts)}")
print(f"gigpo/step_group_size_mean = {counts.mean():.3f}")
print(f"gigpo/step_group_size_max  = {counts.max()}")
print(f"gigpo/singleton_frac       = {(counts == 1).sum() / counts.sum():.3f}")

# episode / step split of the joint advantage
ep_adv = core_gigpo.episode_norm_reward(
    batch.batch["token_level_rewards"], response_mask, uid, traj_uid, 1e-6, True)
st_adv = core_gigpo.step_norm_reward(step_rewards, response_mask, step_group_uids, 1e-6, True)
assert torch.allclose(ep_adv + st_adv, adv, atol=1e-5), "recomputed halves do not sum to the joint advantage"
_ri, _fv = torch.arange(BS), response_mask.int().argmax(dim=-1)
print(f"gigpo/episode_adv_absmean  = {masked_mean(ep_adv.abs(), response_mask).item():.4f}")
print(f"gigpo/step_adv_absmean     = {masked_mean(st_adv.abs(), response_mask).item():.4f}")
print(f"gigpo/episode_adv_zero_row_frac = {(ep_adv[_ri, _fv] == 0).float().mean().item():.3f}")
print(f"gigpo/step_adv_zero_row_frac    = {(st_adv[_ri, _fv] == 0).float().mean().item():.3f}")
# a singleton step group is centred on itself, so its step advantage is exactly 0
singleton_rows = [i for i in range(BS)
                  if (step_group_uids == step_group_uids[i]).sum() == 1]
for i in singleton_rows:
    assert st_adv[i, _fv[i]] == 0, f"singleton row {i} has non-zero step advantage"
print(f"(verified {len(singleton_rows)} singleton rows have A_step == 0 exactly)")

# ---- 4. RLSD token reweighting on top of the GiGPO advantage ----
student_lp = torch.randn(BS, RESP_LEN) * 0.5 - 1.0
teacher_lp = student_lp + torch.randn(BS, RESP_LEN) * 0.32 - 0.133   # match observed Delta
tok_adv = compute_rlsd_token_advantage(
    seq_advantages=adv, student_log_probs=student_lp, teacher_log_probs=teacher_lp,
    response_mask=response_mask, rlsd_lambda=0.5, rlsd_clip_eps=CLIP_EPS,
)
assert tok_adv.shape == (BS, RESP_LEN), tok_adv.shape
assert torch.equal(tok_adv * (1 - response_mask), torch.zeros_like(tok_adv)), "leaked into padding"
# the token level must actually differentiate tokens within a row
within_spread = [(tok_adv[i][response_mask[i].bool()]).std().item()
                 for i in range(BS) if response_mask[i].sum() > 1 and adv[i, 0] != 0]
print(f"within-row std of A_token (rows with A!=0): {np.round(within_spread, 4)}")
assert max(within_spread) > 0, "token level produced no within-row variation"

# ---- 5. the Delta diagnostics added to fit() ----
delta_t = (teacher_lp - student_lp) * response_mask
d_mean = masked_mean(delta_t, response_mask)
row_n = response_mask.sum(dim=-1)
row_mean = (delta_t * response_mask).sum(dim=-1) / row_n.clamp(min=1.0)
row_var = ((delta_t - row_mean.unsqueeze(-1)) ** 2 * response_mask).sum(dim=-1) / row_n.clamp(min=1.0)
multi_tok = row_n > 1
first_valid = response_mask.int().argmax(dim=-1)
sign_A = torch.sign(adv[torch.arange(BS), first_valid]).unsqueeze(-1)
w_raw = torch.exp(sign_A * delta_t)
saturated = ((w_raw <= 1.0 - CLIP_EPS) | (w_raw >= 1.0 + CLIP_EPS)).float()
print(f"rlsd/teacher_student_gap_mean = {d_mean.item():+.4f}")
print(f"rlsd/gap_std_true             = {masked_mean((delta_t - d_mean) ** 2, response_mask).sqrt().item():.4f}")
print(f"rlsd/gap_var_within_step      = {row_var[multi_tok].mean().item():.4f}")
print(f"rlsd/gap_var_between_step     = {row_mean[multi_tok].var(unbiased=False).item():.4f}")
print(f"rlsd/clip_saturation_frac     = {masked_mean(saturated, response_mask).item():.3f}")

A_row = adv[torch.arange(BS), first_valid]
nz_row = A_row != 0
nz_mask = response_mask * nz_row.unsqueeze(-1).to(response_mask.dtype)
print(f"rlsd/zero_adv_row_frac        = {(~nz_row).float().mean().item():.3f}")
print(f"rlsd/zero_adv_token_frac      = "
      f"{1.0 - masked_mean(nz_row.unsqueeze(-1).to(response_mask.dtype).expand_as(response_mask), response_mask).item():.3f}")
print(f"rlsd/clip_saturation_frac_nonzero = {masked_mean(saturated, nz_mask).item():.3f}")
# a zero-advantage row can never be counted as saturated: sign(A)=0 -> w=exp(0)=1
for i in range(BS):
    if A_row[i] == 0:
        assert saturated[i].sum() == 0, f"row {i} has A=0 yet counted as saturated"
assert masked_mean(saturated, nz_mask).item() >= masked_mean(saturated, response_mask).item() - 1e-9, \
    "restricting to A!=0 must not lower the saturation fraction"

print("\nALL CHECKS PASSED")

# ---- 6. additive / hybrid forms ----
# The properties that separate the three forms, checked exactly rather than
# argued: mult can never flip a sign but zeroes out all-fail rows; add always
# carries gradient but can flip; hybrid is mult wherever A != 0 and add exactly
# where A == 0.
print("\n=== forms ===")
zero_rows = (A_row == 0)
print(f"rows with A == 0 (all-fail groups): {int(zero_rows.sum())}/{BS}")
assert bool(zero_rows.any()), "synthetic batch has no A==0 row; the test is vacuous"

kw = dict(seq_advantages=adv, student_log_probs=student_lp, teacher_log_probs=teacher_lp,
          response_mask=response_mask, rlsd_lambda=0.5, rlsd_clip_eps=CLIP_EPS)
forms = {f: compute_rlsd_token_advantage(form=f, **kw) for f in ("mult", "add", "hybrid")}

delta_c = centered_delta(student_lp, teacher_lp, response_mask, add_clip=2.0, center=True)
assert abs(masked_mean(delta_c, response_mask).item()) < 1e-5, "centered delta is not zero-mean"
assert torch.equal(delta_c * (1 - response_mask), torch.zeros_like(delta_c)), "delta_c leaked into padding"

for name, ta in forms.items():
    assert torch.equal(ta * (1 - response_mask), torch.zeros_like(ta)), f"{name} leaked into padding"
    dead = 1.0 - masked_mean((ta != 0).float(), response_mask).item()
    flip = masked_mean(((torch.sign(ta) * torch.sign(A_row).unsqueeze(-1)) < 0).float(), nz_mask).item()
    resc = masked_mean((ta != 0).float(),
                       response_mask * zero_rows.unsqueeze(-1).float()).item()
    print(f"{name:6s} dead_token_frac={dead:.3f}  sign_flip_frac={flip:.3f}  rescued_token_frac={resc:.3f}")

# mult: sign is preserved by construction (w > 0), and A==0 rows stay dead
assert masked_mean(((torch.sign(forms['mult']) * torch.sign(A_row).unsqueeze(-1)) < 0).float(),
                   nz_mask).item() == 0.0, "mult flipped a sign, which w>0 makes impossible"
for i in torch.nonzero(zero_rows).flatten().tolist():
    assert forms["mult"][i].abs().sum() == 0, f"mult row {i} has A==0 but nonzero advantage"
    assert forms["hybrid"][i].abs().sum() > 0, f"hybrid failed to rescue A==0 row {i}"
    assert forms["add"][i].abs().sum() > 0, f"add failed to rescue A==0 row {i}"
# hybrid == mult wherever A != 0, == add wherever A == 0
nz_rows_idx = torch.nonzero(~zero_rows).flatten().tolist()
assert torch.allclose(forms["hybrid"][nz_rows_idx], forms["mult"][nz_rows_idx]), \
    "hybrid diverges from mult on rows with outcome credit"
assert torch.allclose(forms["hybrid"][torch.nonzero(zero_rows).flatten().tolist()],
                      forms["add"][torch.nonzero(zero_rows).flatten().tolist()]), \
    "hybrid diverges from add on all-fail rows"
# lambda = 0 must collapse every form back to the plain sequence advantage
for f in ("mult", "add", "hybrid"):
    ta0 = compute_rlsd_token_advantage(**{**kw, "rlsd_lambda": 0.0}, form=f)
    assert torch.allclose(ta0, adv * response_mask, atol=1e-6), f"{f} does not reduce to A at lambda=0"
print("(lambda=0 reduces all three forms to the plain sequence advantage)")

try:
    compute_rlsd_token_advantage(form="bogus", **kw)
    raise AssertionError("unknown form was silently accepted")
except ValueError:
    pass

print("\nALL FORM CHECKS PASSED")

# ---- 7. conflict diagnostics ----
# Two independent estimates of "was this step good?" -- the environment's local
# return (A^step) and the privileged teacher's likelihood gap (Delta) -- and the
# two reward tiers (A^epi vs A^step) among themselves. The fusion story only has
# something to arbitrate if these actually disagree, so the disagreement rate is
# measured rather than assumed. This section checks the metric CODE (partition,
# masking, degenerate cases); the rates themselves are meaningless on synthetic
# log-probs and only mean something on a real batch.
print("\n=== conflict ===")
STEP_W = 1.0
ep_row = ep_adv[_ri, _fv]
st_row = st_adv[_ri, _fv] * STEP_W

both_nz = (ep_row != 0) & (st_row != 0)
print(f"gigpo/both_tiers_nonzero_frac = {both_nz.float().mean().item():.3f}")
assert bool(both_nz.any()), "no row has both tiers non-zero; the conflict metric is vacuous"
e, s = ep_row[both_nz], st_row[both_nz]
disagree = (torch.sign(e) != torch.sign(s))
overturn = (torch.sign(e + s) != torch.sign(e))
print(f"gigpo/ep_step_disagree_frac   = {disagree.float().mean().item():.3f}")
print(f"gigpo/step_overturn_frac      = {overturn.float().mean().item():.3f}")
# An overturn requires a disagreement: if both tiers point the same way their sum
# cannot land on the other side of zero.
assert bool((overturn <= disagree).all()), "a sign was overturned without the two tiers disagreeing"

# Delta must be centred before its sign is read: it has a global offset (measured
# ~-0.13 on ALFWorld/Qwen2.5-3B), and an offset shared by every row is bias, not
# evidence. Without centring nearly every row would read "teacher disapproves".
d_row = row_mean - d_mean
sel = st_row != 0
sv, dv = st_row[sel], d_row[sel]
sp, dp = sv > 0, dv > 0
quads = {
    "quad_env_pos_teacher_pos": (sp & dp).float().mean().item(),
    "quad_env_pos_teacher_neg": (sp & ~dp).float().mean().item(),
    "quad_env_neg_teacher_pos": (~sp & dp).float().mean().item(),
    "quad_env_neg_teacher_neg": (~sp & ~dp).float().mean().item(),
}
conflict = (sp != dp).float().mean().item()
print(f"rlsd/step_evidence_frac       = {sel.float().mean().item():.3f}")
print(f"rlsd/evidence_conflict_frac   = {conflict:.3f}")
for k, v in quads.items():
    print(f"rlsd/{k} = {v:.3f}")
assert abs(sum(quads.values()) - 1.0) < 1e-6, "the four quadrants do not partition the selected rows"
assert abs((quads["quad_env_pos_teacher_neg"] + quads["quad_env_neg_teacher_pos"]) - conflict) < 1e-6, \
    "conflict_frac is not the sum of the two off-diagonal quadrants"

print(f"rlsd/step_teacher_corr        = {_pearson(sv, dv):+.4f}")
print(f"rlsd/ep_teacher_corr          = {_pearson(ep_row[sel], dv):+.4f}")
# the guard that matters: a degenerate (constant) input must not emit NaN
assert _pearson(torch.zeros(5), torch.randn(5)) == 0.0, "constant input did not return 0.0"
assert _pearson(torch.randn(5), torch.zeros(5)) == 0.0, "constant input did not return 0.0"
_x = torch.randn(64)
assert abs(_pearson(_x, _x) - 1.0) < 1e-5 and abs(_pearson(_x, -_x) + 1.0) < 1e-5, "_pearson is miscalibrated"

print("\nALL CONFLICT CHECKS PASSED")

# ---- 8. peer-trajectory hindsight (the no-skill / StepOPSD teacher) ----
# The fixture already has exactly the three group types that matter: taskA and
# taskB each have one success and one failure (teachable), taskC has neither
# (untouchable). What is checked is the row->prefix assignment, not the text.
print("\n=== peer hindsight ===")
ALF_SUCCESS_REWARD = 10.0        # agent_system/.../alfworld/envs.py: 10.0 * won
traj_success = {"A0": True, "A1": False, "B0": True, "B1": False,
                "C0": False, "C1": False}
episode_rewards = np.array(
    [ALF_SUCCESS_REWARD if traj_success[t] else 0.0 for t in traj_uid], dtype=object)
turn_step = np.array([0, 1, 2] * 6, dtype=object)

# stand-in for the tokenizer: rows are ids into a table of decoded responses
resp_text = [f"<think>...</think><action> act_{traj_uid[i]}_{turn_step[i]} </action>"
             for i in range(BS)]
responses = torch.arange(BS).unsqueeze(-1).expand(BS, RESP_LEN).contiguous()


class _FakeTokenizer:
    def batch_decode(self, ids, skip_special_tokens=True):
        return [resp_text[int(row[0])] for row in ids]


peer_batch = DataProto(
    batch=TensorDict({"responses": responses}, batch_size=[BS]),
    non_tensor_batch={"uid": uid, "traj_uid": traj_uid,
                      "turn_step": turn_step, "episode_rewards": episode_rewards},
)
prefixes = build_peer_hindsight_prefixes(
    peer_batch, _FakeTokenizer(), success_threshold=1.0, max_peer_steps=0)
assert len(prefixes) == BS, len(prefixes)
coverage = float(np.mean([1.0 if p else 0.0 for p in prefixes]))
print(f"rlsd/teacher_coverage        = {coverage:.3f}")
print("prefix of row 3 (failed A1):\n" + "\n".join("  " + l for l in prefixes[3].splitlines()))

# successes are not taught; all-fail groups have nobody to learn from
for i in range(BS):
    if traj_success[traj_uid[i]]:
        assert prefixes[i] == "", f"row {i} is a success yet was given a teacher"
    elif uid[i] == "taskC":
        assert prefixes[i] == "", f"row {i} is in an all-fail group yet was given a teacher"
    else:
        assert prefixes[i].startswith(PEER_HINDSIGHT_HEADER), f"row {i} has no hindsight"
assert abs(coverage - 6 / BS) < 1e-9, f"expected 6/18 rows taught, got {coverage}"

# every row of one failed trajectory sees the SAME peer, in trajectory order
assert prefixes[3] == prefixes[4] == prefixes[5], "rows of one trajectory disagree on the peer"
assert prefixes[3] != prefixes[9], "taskA and taskB were given the same peer"
body = prefixes[3][len(PEER_HINDSIGHT_HEADER):].strip().splitlines()
assert body == ["1. act_A0_0", "2. act_A0_1", "3. act_A0_2"], body

# the coverage claim that the whole comparison rests on: peer hindsight is
# non-empty on a SUBSET of the rows where GiGPO's step advantage is non-zero,
# never on a row where the group has no outcome variance at all.
taught = torch.tensor([1.0 if p else 0.0 for p in prefixes])
assert bool(((taught > 0) <= (torch.tensor(A_row) != 0)).all()), \
    "a row with zero GiGPO advantage was nevertheless given peer hindsight"
print(f"taught rows {int(taught.sum())}/{BS}; A!=0 rows {int((A_row != 0).sum())}/{BS} "
      f"(taught is a subset: OK)")

# ordering must come from turn_step, not from row order
shuf = np.array([2, 0, 1, 5, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
shuf_batch = DataProto(
    batch=TensorDict({"responses": responses[shuf]}, batch_size=[BS]),
    non_tensor_batch={"uid": uid[shuf], "traj_uid": traj_uid[shuf],
                      "turn_step": turn_step[shuf], "episode_rewards": episode_rewards[shuf]},
)
shuf_prefixes = build_peer_hindsight_prefixes(shuf_batch, _FakeTokenizer(), 1.0, 0)
assert shuf_prefixes[3] == prefixes[3], "peer action order depends on row order, not turn_step"

# max_peer_steps keeps the TAIL (the steps that actually closed the task)
tail = build_peer_hindsight_prefixes(peer_batch, _FakeTokenizer(), 1.0, max_peer_steps=2)
assert tail[3][len(PEER_HINDSIGHT_HEADER):].strip().splitlines() == \
    ["1. act_A0_1", "2. act_A0_2"], tail[3]

# adjust_batch(mode="copy") duplicates random rows to reach a divisible batch
# size, so the peer's action list must not repeat a step it already listed
dup = np.concatenate([np.arange(BS), np.array([1, 1])])
dup_batch = DataProto(
    batch=TensorDict({"responses": responses[dup]}, batch_size=[len(dup)]),
    non_tensor_batch={"uid": uid[dup], "traj_uid": traj_uid[dup],
                      "turn_step": turn_step[dup], "episode_rewards": episode_rewards[dup]},
)
dup_prefixes = build_peer_hindsight_prefixes(dup_batch, _FakeTokenizer(), 1.0, 0)
assert dup_prefixes[3] == prefixes[3], f"duplicated rows leaked into the peer action list:\n{dup_prefixes[3]}"

# a group where everyone succeeds is teachable-by-nobody for the opposite reason
all_win = np.array([ALF_SUCCESS_REWARD] * BS, dtype=object)
win_batch = DataProto(
    batch=TensorDict({"responses": responses}, batch_size=[BS]),
    non_tensor_batch={"uid": uid, "traj_uid": traj_uid,
                      "turn_step": turn_step, "episode_rewards": all_win},
)
assert all(p == "" for p in build_peer_hindsight_prefixes(win_batch, _FakeTokenizer(), 1.0, 0)), \
    "an all-success batch produced a teacher"

print("\nALL PEER HINDSIGHT CHECKS PASSED")

# ---- 9. the coverage artifact in step_teacher_corr ----
# Untaught rows have Delta identically 0. Subtracting the batch mean (which is
# negative, because the teacher is on average less confident) turns every one of
# them into a constant "teacher approves". Untaught rows are not random -- every
# successful trajectory is one -- so the unrestricted correlation partly measures
# the coverage pattern rather than the teacher. Only 3 rows here are both taught
# and have step evidence, far too few for the restricted correlation to settle at
# its true value, so what is asserted is the DIRECTION (restricting shrinks it)
# and the mechanism (untaught rows all collapse to the same side), not a number.
print("\n=== coverage artifact ===")
taught_b = taught > 0
art_delta = torch.zeros(BS, RESP_LEN)
# taught rows: a pattern that does not follow the fixture's A^step
pattern = torch.tensor([+0.5, -0.5, +0.5, -0.5, +0.5, -0.5])
art_delta[taught_b] = (pattern[:int(taught_b.sum())].unsqueeze(-1) - 0.30) * response_mask[taught_b]
art_delta = art_delta * response_mask

art_row_n = response_mask.sum(dim=-1)
art_row_mean = (art_delta * response_mask).sum(dim=-1) / art_row_n.clamp(min=1.0)
art_d_mean = masked_mean(art_delta, response_mask)
sel_all = st_row != 0
corr_all = _pearson(st_row[sel_all], (art_row_mean - art_d_mean)[sel_all])

sel_taught = sel_all & taught_b
dv_taught = art_row_mean[sel_taught] - art_row_mean[sel_taught].mean()
corr_taught = _pearson(st_row[sel_taught], dv_taught)
print(f"rlsd/step_teacher_corr        = {corr_all:+.4f}   (all rows with step evidence)")
print(f"rlsd/step_teacher_corr_taught = {corr_taught:+.4f}   (rows the teacher actually saw)")
print(f"rlsd/taught_and_step_evidence_frac = {sel_taught.float().mean().item():.3f}")
assert abs(corr_all) > abs(corr_taught), \
    "the fixture failed to reproduce the artifact, so this check proves nothing"
# every untaught row must land on the same side after global centring -- that is
# the mechanism, stated as an assertion rather than an argument
untaught_centred = (art_row_mean - art_d_mean)[~taught_b]
assert bool((untaught_centred > 0).all()) or bool((untaught_centred < 0).all()), \
    "untaught rows did not collapse to one side; the artifact mechanism is misstated"

print("\nALL ARTIFACT CHECKS PASSED")

# ---- 10. decoupled form: A^E + w*A^S*[(1-lam) + lam*w_t] ----
# The teacher is confined to the STEP tier. Three tiers, three jobs: A^E says the
# trajectory succeeded, A^S says the step was good, w_t says which tokens inside
# the step carry that step credit. What that buys and what it costs are both
# checkable exactly, so neither is left as an argument:
#   BUYS   the sign question "does the teacher follow sign(A^E) or sign(A^E+wA^S)
#          when the two disagree?" stops existing. w_t keys off sign(A^S), the
#          only tier it multiplies, and the shaping factor (1-lam)+lam*w_t is
#          strictly positive, so the step tier's direction is preserved and the
#          episode tier is returned bit-exact.
#   COSTS  the teacher is identically inert wherever A^S == 0 -- not attenuated,
#          absent. mult only dies when the JOINT advantage is 0, which is rarer.
print("\n=== decoupled ===")
st_adv_w = st_adv * STEP_W
dec_kw = dict(seq_advantages=adv, student_log_probs=student_lp, teacher_log_probs=teacher_lp,
              response_mask=response_mask, rlsd_clip_eps=CLIP_EPS, form="decoupled",
              episode_advantages=ep_adv, step_advantages=st_adv_w)
dec = compute_rlsd_token_advantage(rlsd_lambda=0.5, **dec_kw)
assert torch.equal(dec * (1 - response_mask), torch.zeros_like(dec)), "decoupled leaked into padding"

# lambda = 0 collapses to the plain GiGPO advantage, exactly as the other forms do
dec0 = compute_rlsd_token_advantage(rlsd_lambda=0.0, **dec_kw)
assert torch.allclose(dec0, adv * response_mask, atol=1e-6), \
    "decoupled does not reduce to A^E + w*A^S at lambda=0"

# the episode tier is untouched, bit for bit, wherever the step tier is silent
inert = (st_adv_w[_ri, _fv] == 0)
active = ~inert
print(f"rlsd/teacher_active_frac (decoupled) = {active.float().mean().item():.3f}   [A^S != 0]")
print(f"rlsd/teacher_active_frac (mult)      = {(A_row != 0).float().mean().item():.3f}   [A^E + w*A^S != 0]")
assert bool(inert.any()), "no row has A^S == 0; the inertness check is vacuous"
for i in torch.nonzero(inert).flatten().tolist():
    assert torch.allclose(dec[i], (ep_adv[i] * response_mask[i])), \
        f"row {i} has A^S == 0 but decoupled did not return A^E untouched"
# The two "dead" numbers are not the same notion and must not be compared naively.
# Under mult a dead row has NO gradient at any tier; under decoupled an inert row
# still carries its episode credit and only loses the teacher. Generically the
# mult-dead set is contained in the decoupled-inert set -- A^E + w*A^S == 0 with
# A^S != 0 needs an exact cancellation -- so decoupled strictly reduces teacher
# REACH while strictly increasing the number of rows carrying gradient.
mult_dead = torch.nonzero(A_row == 0).flatten().tolist()
assert all(bool(inert[i]) for i in mult_dead), \
    "a row was dead under mult but active under decoupled: the containment is not exact here"
alive_under_dec = [i for i in mult_dead if dec[i].abs().sum() > 0]
print(f"(mult-dead rows {len(mult_dead)}, of which alive under decoupled: {len(alive_under_dec)} "
      f"-- the rest have A^E == 0 too, so nothing can rescue them)")

# the guarantee that replaces the sign question: the teacher may rescale the step
# tier's contribution but never reverse it, because (1-lam)+lam*w_t >= 1-lam*eps > 0
step_contrib = dec - ep_adv * response_mask
for i in torch.nonzero(active).flatten().tolist():
    m = response_mask[i].bool()
    assert bool((torch.sign(step_contrib[i][m]) == torch.sign(st_adv_w[i, _fv[i]])).all()), \
        f"row {i}: the teacher reversed the step tier's sign, which the shaping factor forbids"
    lo, hi = (1 - 0.5) + 0.5 * (1 - CLIP_EPS), (1 - 0.5) + 0.5 * (1 + CLIP_EPS)
    ratio = step_contrib[i][m] / st_adv_w[i, _fv[i]]
    assert bool(((ratio >= lo - 1e-5) & (ratio <= hi + 1e-5)).all()), \
        f"row {i}: shaping factor left [{lo}, {hi}]"
print(f"(step contribution stays within [{lo:.2f}, {hi:.2f}] x A^S on every active row)")

# it must also differentiate tokens WITHIN an active row, or it is just a rescale
dec_spread = [step_contrib[i][response_mask[i].bool()].std().item()
              for i in torch.nonzero(active).flatten().tolist()
              if response_mask[i].sum() > 1]
assert min(dec_spread) > 0, "decoupled produced a constant step contribution within a row"
print(f"within-row std of the step contribution: {np.round(dec_spread, 4)}")

# the worked example from the method spec, reproduced numerically:
#   A^E = -1, omega = 0.5, lambda = 0.2, w_t = 1.2 -> factor 0.8 + 0.24 = 1.04
#   bad step  A^S = -1.2 -> -1 + 0.5*(-1.2)*1.04 = -1.624
#   good step A^S = +1.2 -> -1 + 0.5*(+1.2)*1.04 = -0.376
# Delta = -+1.0 saturates the clip at 1.2 in both cases (exp(1.0) = 2.718 > 1.2).
ex_mask = torch.ones(2, 3)
ex_ep = torch.full((2, 3), -1.0)
ex_st = torch.stack([torch.full((3,), 0.5 * -1.2), torch.full((3,), 0.5 * 1.2)])
ex_delta = torch.stack([torch.full((3,), -1.0), torch.full((3,), 1.0)])
ex = compute_rlsd_token_advantage(
    seq_advantages=ex_ep + ex_st, student_log_probs=torch.zeros(2, 3),
    teacher_log_probs=ex_delta, response_mask=ex_mask, rlsd_lambda=0.2,
    rlsd_clip_eps=0.2, form="decoupled", episode_advantages=ex_ep, step_advantages=ex_st)
print(f"worked example: bad step {ex[0, 0].item():+.4f} (spec -1.6240), "
      f"good step {ex[1, 0].item():+.4f} (spec -0.3760)")
assert abs(ex[0, 0].item() - (-1.624)) < 1e-5 and abs(ex[1, 0].item() - (-0.376)) < 1e-5, \
    "the implementation disagrees with the worked example in the method spec"

# decoupled needs the tiers, not their sum -- silently falling back to seq_advantages
# would turn it into mult without saying so
try:
    compute_rlsd_token_advantage(seq_advantages=adv, student_log_probs=student_lp,
                                 teacher_log_probs=teacher_lp, response_mask=response_mask,
                                 rlsd_lambda=0.5, rlsd_clip_eps=CLIP_EPS, form="decoupled")
    raise AssertionError("decoupled ran without the two tiers")
except ValueError:
    pass

print("\nALL DECOUPLED CHECKS PASSED")
