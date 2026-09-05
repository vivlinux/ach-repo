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

## Now — executing the approved plan (uploads due 16:30)

- [ ] **Step 0 (10 min, deliverable)**: `.gitignore`, first commit, push to
      `vivlinux/ach-repo`, confirm URL loads. Record v4's baseline in
      PROGRESS.md.
- [ ] **Step 1 (15 min)**: `scripts/head_report.py` — one command reporting
      emulated score, saturation, per-class fire rate on the 6 normal
      videos, train/val gap. The keep/discard oracle for every step after.
- [ ] **Step 2 (30 min)**: de-saturate the head — label smoothing 0.1,
      pos-weight-max 3 (loitering's likely cause), feature noise 0.03.
      New checkpoint `out/head_reg.pt`, `out/head.pt` (v4) untouched.
      Risk: medium — may un-suppress the fragile classes, which Step 3
      exists to handle.
- [ ] **Step 3 (35 min)**: wire `NoveltyBank` (`vad/openset.py`) into
      `predict` as an abstention signal — dampens fragile-class scores on
      out-of-distribution windows, never silences a video that had
      real intervals (the alert-credit lesson from v6, made permanent as
      `postprocess.never_silence()`). Fixes 3 bugs found in the module
      first (self-referential calibration, distance/novelty unit mismatch).
- [ ] **Step 4 (10 min)**: merge L1 answers — union of head+VLM for
      anomaly/normal (recall favoured, since 20/24 L1 videos are
      anomalous), VLM's class on disagreement (better class accuracy).
      Rejected "agree-only": correct for a false-alert-budget objective,
      wrong for this scoring metric.
- [ ] **Final (by 16:30)**: `submit --honest-timing`, validate, upload
      **only if** the decision rule holds (≥53.1 + better generalisation
      profile). Otherwise v4 stands as final.

## Decision rule (same for every step above)

Keep a change only if: emulator total ≥ 53.1 (±1.5 is one video, treat as
noise) **and** the generalisation profile improves (saturation ↓, normal-
video fire rate ↓, train/val gap ↓, no event-video loses its last interval).
A result in the 52–53.1 band with a clearly better profile is a judgment
call for the user, never auto-uploaded.

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
