# Today's sprint — 2026-09-05

**Uploads due 16:30.** Currently: **v4 submitted, scored 53.1/100, rank 5/9**
(L1 13.2/25, L2 22.6/35, L3 17.3/40). `scripts/arena_score.py` reproduces this
to within 0.6. v4 stays the submitted result unless a new file beats it on
both the public score *and* a generalisation profile (see the active plan).

Full plan for the rest of the window:
`/Users/vivekkumarsingh/.claude/plans/ok-we-have-come-bright-creek.md`

Legend: `[x]` done · `[~]` in progress · `[ ]` not started

---

## Done today (Blocks 1–5, condensed — see PROGRESS.md §1–10 for detail)

- [x] **Fixed 12 classes of training data.** Downloaded all 12 (was 5).
      Relabelled from `description_summary` (`scripts/relabel_train.py`) —
      520/2,952 rows changed, non-destructive, `io.py` prefers it
      automatically. Found `road_spill_or_debris` had **0** genuine examples
      (was 99% mislabelled collisions), `wrong_way_driving` down to 35 from
      one camera.
- [x] **VLM gate tested and rejected** as a Stage-2 filter: removed 14 false
      alarms/0 real events but *lowered* score 29.6→27.6 — it can drop, not
      relabel. **VLM-as-classifier worked instead**: L1 12.0→14.1.
- [x] **Trained the temporal head** on relabelled data (`vad/head.py`,
      `out/head.pt`). This is what actually moved the score:
      zero-shot 24.5 → head 53.7 emulated (53.1 real). Overfits hard
      (train 0.011 / val 0.296, random per-video split) but still
      generalises far better than zero-shot on the public 34.
- [x] Diagnosed why: head output is **saturated at 0/1** (`loitering` fires
      at 1.000 on >10% of all test windows) — threshold tuning has nothing
      to grip. Three classes (stalled/wrong_way/road_spill) self-suppressed
      to max score <0.2, learned from the corrected labels that they're
      unreliable.
- [x] Tried cropping false-alarm classes by name (v5) and gating the head's
      output (v6/full-gate) — both **rejected**: v5 is a rule fitted to the
      public 34 (breaks on hidden data if it has real fighting/loitering
      events); full gate silenced 15 videos and lost the L2/L3 "alert
      credit," costing 9 points.
- [x] Deck published as the write-up/PPT:
      https://claude.ai/code/artifact/acc697a2-08de-4f8b-951f-41499f601615
      — covers the cascade, the relabelling finding, and all four negative
      results (bigger model tied, gate hurt, prompt hypothesis backwards
      twice, cropping v5 rejected).

## Robustness plan — executed, all four experiments discarded (15:11)

Full plan: `/Users/vivekkumarsingh/.claude/plans/ok-we-have-come-bright-creek.md`.
**v4 (`out/head.pt`, no VLM) remains the submitted, final result — 53.1
real / 49.0 emulated under the corrected formula.** Nothing built after it
beat it; every step below is a genuine, verified negative, not an
unfinished attempt.

- [x] **Step 0** — repo pushed: https://github.com/vivlinux/ach-repo.
- [x] **Step 1** — `scripts/head_report.py` built and validated: reproduces
      v4 exactly (49.0, 66.7% saturation, self-suppressed classes at their
      known values). This also **found and fixed a real bug**: the L1
      scoring formula in `scripts/arena_score.py` was accuracy-based, not
      precision/recall as the real arena panel showed. Corrected; v4's own
      emulated score moved 53.7→49.0 as a result (nothing about v4 changed,
      only the yardstick).
- [x] **Step 2, discarded**: de-saturated head (label smoothing 0.1,
      pos-weight-max 3, feat noise 0.03) collapsed the score **49.0→35.9**
      and lost alert-credit on 2 event videos. Saturation genuinely dropped
      66.7%→14.2% and the train/val gap closed to ~0 — but the model
      underfit rather than generalised; best-val landed at epoch 1/25.
      Per the plan's own rule ("run a second config only if the first is
      close"), 35.9 vs a required ≥49.0 is not close — stopped iterating
      rather than spend the budget chasing it. Code/CLI flags kept for
      future use; `out/head_reg.pt` not adopted.
- [x] **Step 3, built and safe, but unproven**: `NoveltyBank` wired as an
      out-of-distribution dampener on the 5 `FRAGILE_CLASSES`, after fixing
      3 real bugs in `vad/openset.py` (self-referential calibration, a
      distance/novelty unit mismatch, a normal-only bank that would call
      ordinary traffic "novel"). Measured **zero effect** on the public 34:
      only 0.2% of test windows cross the OOD threshold, and none of those
      overlap a fragile-class signal. The 3 false alarms that do remain on
      normal videos (`traffic_congestion`, `smoke`, `traffic_accident` on
      T003/T004) aren't even fragile-class targets — a different failure
      mode (confident-and-wrong on a camera-diverse class, not "confused by
      an unfamiliar scene"). Provably doesn't hurt v4; can't demonstrate it
      helps without OOD test data we don't have. Off by default in the
      submitted file.
- [x] **Step 4, net neutral**: `scripts/merge_l1.py` (head OR VLM for
      anomaly/normal, VLM's class on disagreement) changed exactly 4 of 24
      L1 videos — 1 regression (T013: head correctly said `fire`, merge
      overrode it to `smoke`, now wrong), 1 fix (T015: head wrongly said
      `fire`, merge corrected it to `smoke`). They cancel exactly; L1 score
      identical (12.5) either way. Genuinely inconclusive on n=4; not
      adopted.

**Net result of the whole robustness pass**: no submission change. The
value was in what got *verified* — a real scoring-formula bug fixed, a
bad hyperparameter combination caught before shipping it, an abstention
mechanism proven safe (if not yet proven useful), and a plausible-sounding
L1 policy shown to be a wash rather than assumed to help.

## Decision rule that governed all four (for the record)

Keep a change only if: emulator total ≥ 49.0 (the corrected baseline; ±1.5
is one video, treat as noise) **and** the generalisation profile improves
(saturation ↓, normal-video fire rate ↓, train/val gap ↓, no event-video
loses its last interval). None of the four met both.

---

## Not in today's critical path (parked, not forgotten)

- Object detector + tracker for per-object trajectory — the real fix for
  `wrong_way_driving`/`vehicle_blocking_traffic`, not a today-build.
- `--zeroshot-classes` blending (vs. today's overwrite) — no measurable
  signal on 1–3 public examples per class; high risk if enabled blind.
- Camera-grouped CV on train — no source ids exist; folder ≈ class, so
  holding out a folder holds out a class instead.
- A true LLM tool-loop, and Holmes-VAD/Vad-R1 (no confirmed Apple Silicon
  path) — evaluated and declined earlier today.
- Fine-tuning any VLM — declined for a one-day clock.

---

## Earlier crash recovery (session died ~13:17, recovered — kept for record)

Lost mid-flight: head training run, Qwen3-VL-8B L1 run. Both were restarted
successfully; results are in "Done today" above. Root cause was a Metal OOM
from Ollama holding Gemma's 17 GB resident while Qwen3-8B and head training
competed for GPU memory — Gemma is now fully removed from the stack
(Qwen3-VL only, per user's later decision), so this can't recur the same way.
