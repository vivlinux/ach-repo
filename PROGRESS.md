# Progress log — plain-language explainer, kept updated

This file explains *what the system is doing and why*, in order, so you can
come back to it later without re-deriving anything. I update this as we go —
if you re-read it mid-hackathon it should reflect the current state, not the
plan we started with. For today's task list see `SPRINT.md`. For the original
design rationale see `README.md`.

---

## 1. The two-stage idea, in one paragraph

We never run a heavy vision-language model on every frame — that's too slow
for the real-time budget. Instead: a **cheap Stage 1** scans the whole video
and turns each short window of frames into a per-class "how much does this
look like X" score, using nothing but frame-vs-text similarity (no reading,
no reasoning). Its output is a short list of candidate time ranges. Only
those few candidates go to the **expensive Stage 2**, an actual
vision-language model (MLX Qwen2.5-VL) that looks at a handful of frames and
answers a real yes/no question in language. Stage 1 does volume, Stage 2 does
judgment on the few things worth judging.

## 2. Glossary (terms that came up and what they mean here)

| Term | Plain meaning |
|---|---|
| **Embedding** | A frame turned into a list of ~768 numbers. Similar-looking frames get similar numbers. Produced by SigLIP (Stage 1's encoder — not a language model, just an image→numbers converter). |
| **Window** | A short slice of the video (16 sampled frames ≈ 8s), the unit Stage 1 scores. Windows overlap (hop every 4 frames) so an event isn't missed by falling on a boundary. |
| **Zero-shot mode** | Stage 1 with **no training**: compare each window's numbers to each class name's numbers (also turned into numbers) and see which is closest. This is what every test run so far has used. |
| **Head / trained mode** | Stage 1 with a small trained model instead of raw comparison — learns from labelled examples instead of just "closest text description." Not trained yet (needs relabeled train data first — see §4). |
| **Hysteresis (`hi`/`lo`)** | Two thresholds instead of one, so a borderline score doesn't flicker an event on/off every window. Opens above `hi`, only closes once it drops below the lower `lo`. |
| **`scale`** | One dial that raises/lowers every class's `hi`/`lo` at once. Blunt — can't fix one class without moving all the others. |
| **`PRIORS`** (`vad/config.py`) | Per-class settings (`min_dur`, `merge_gap`, `hi`, `lo`, `smooth_k`) — the precise version of `scale`, tunable one class at a time. |
| **Interval** | A candidate event Stage 1 found: (class, start time, end time, confidence). |
| **Verification gate / `--verify`** | Turns Stage 2 on. Not used in any run yet — everything scored so far is Stage 1 alone. |
| **Level 1 / 2 / 3** | The arena's three task tiers. Level 1 = one label for the whole clip, no timestamps. Level 2/3 = you must give start/end times for every event; Level 3 is longer/harder footage. Different scoring rules per level (see `Copy of AHC... .pdf`). |
| **IoU (intersection over union)** | How well your predicted time range overlaps the true one. Needs to be ≥0.5 on the real arena (we score locally at ≥0.2, a looser proxy). |
| **manifest.json** | The arena's official list of which videos you must answer and each one's level. We don't have this yet — using this dataset's own `test/ground_truth.csv` as a local stand-in. **Must swap in the real one before uploading.** |

## 3. What's actually running, and what each command means

```
python -m vad.cli embed --splits test        # video -> cached numbers (done, 34/34 cached)
python -m vad.cli predict --mode zeroshot ... # numbers -> guesses per video (Stage 1 only)
python -m vad.cli score  --pred ...           # compare guesses to test/ground_truth.csv
python -m vad.cli submit --pred ...           # guesses -> arena-format JSON, checked against every submission rule
```

`score` is **local and approximate** — it's our own proxy metric so we can
iterate without spending an arena upload. It is not the same math the arena
uses (it evaluates all 34 videos uniformly; the arena pools Level 1 separately
from Levels 2/3). Good for "did that change help or hurt," not a preview of
the leaderboard number.

## 4. Findings so far (updated as new ones land)

- **Environment**: `transformers` had to be pinned `<5` — v5 changed
  `get_image_features()`'s return type and silently broke Stage 1's encoder.
  Fixed in `requirements.txt`.
- **No submission writer existed** before today. Built `vad/submission.py` +
  `cli submit` — converts the local CSV into the real arena JSON and checks
  every rule from the format spec (null timestamps at Level 1, `events: []`
  for normal, the 2% runtime-metadata tolerance, etc.). Unit-tested against
  a clean case and a deliberately broken one.
- **Zero-shot Stage-1-only baseline** (first real number, no training, no
  Stage 2): Level 1 F1 0.78 but class accuracy on what it detects only 0.29
  — decent at "something's wrong," weak at "what." Level 2/3 macro F1 0.049
  — only `traffic_congestion` works at all.
- **Root cause found for the worst offenders**: `wrong_way_driving` (17 false
  positives, structurally 0 possible true positives in this metric — its one
  test example has no timestamps) and `vehicle_blocking_traffic` (24 false
  positives against 1 real instance) are firing on ordinary two-way traffic,
  because Stage 1 has no concept of *which direction a specific vehicle is
  heading* — it only compares whole-frame appearance to a text prompt. This
  is a per-object-trajectory problem, not something a global `scale` can fix.
- **Training data is mislabeled at the folder level, for the original 5
  classes only.** `road_spill_or_debris`, `stalled_or_broken_down_vehicle`,
  `wrong_way_driving`, `vehicle_blocking_traffic` don't match
  `description_summary` well (`road_spill_or_debris` folder is 99% actually
  collision footage); `traffic_congestion` is clean. Not yet relabeled —
  training the head on these folder labels as-is would teach it wrong
  associations.
- **11 of 12 class folders complete and mp4-count-matched.** Only
  `loitering_or_suspicious_presence` still blocked — its `-1-001.zip` part
  (holding the CSVs + ~half the clips) never downloaded, only part 2 landed
  (79 videos, zero metadata). `normal` was short 175 videos from a lagging
  Google Drive Desktop sync; confirmed fully synced now (973/973).
  `io.py` silently skips CSV rows whose video file is missing — no error,
  no warning — worth remembering if a count ever looks off again.
- **Verified clean** (description matches folder label, ready to train as-is):
  `fire` (77), `smoke` (85), `waterlogging_or_flood` (95),
  `traffic_congestion` (268), `traffic_accident` (565).
  **Confirmed noisy, needs relabeling before training**:
  `road_spill_or_debris`, `stalled_or_broken_down_vehicle`,
  `wrong_way_driving`, `vehicle_blocking_traffic`.
  **Partially noisy**: `fighting_or_violence` (124, only 63% keyword-match —
  worth a closer look, not necessarily all wrong). `normal` (973) — checked for anomaly-language contamination instead of a
  positive keyword match: **0 of 973 rows contain any anomaly-sounding
  language. Clean.**
- **Stage 2 (the actual VLM) has not been turned on in any run yet.**
  Everything scored so far is the cheap, non-reasoning Stage 1 alone.
- **Train relabeling done** (`scripts/relabel_train.py`, non-destructive —
  writes `ground_truth_relabeled.csv` next to each folder's original;
  `vad/io.py` now prefers it automatically when present). Keyword heuristic
  with a priority rule (a described collision always wins, since that's the
  exact pattern found in the noisy folders) plus negation-awareness (a
  first pass wrongly caught "no obstructions" / "lack of congestion" as
  positive matches for the class they were denying — found by spot-checking
  output, not by inspection, and fixed). 520 of 2,952 rows relabeled.
  **Corrected distribution is uneven**: `road_spill_or_debris` collapses to
  **2 genuine examples** (confirms it was ~99% mislabeled collisions,
  consistent with the original audit) and `wrong_way_driving` to **35**
  (from 164) — the head will not learn these two well regardless of label
  quality now; this is a data-scarcity problem, not a heuristic one.
  `road_spill_or_debris` in particular may need zero-shot-only reliance or
  a supplementary data source if it matters for scoring.
- **All 12 classes now present** (2,952 videos with labels, up from 954).
  `loitering_or_suspicious_presence` still has no metadata (still missing
  its other zip part) so its 79 videos fall back to a blanket
  video-level label with no real per-event supervision.

## 5. Open decisions (yours to make, not yet decided)

- Re-download the 7 missing training classes vs. accept zero-shot-only on them.
- Automate the train relabeling (embedding/LLM pass) vs. hand-check a sample.
- Whether to add a real object detector + tracker (fixes the trajectory gap
  above) — biggest single accuracy lever identified, not yet started.

## 6. Known blind spots — where unknowns outweigh knowns (2026-09-05, midday)

Ranked by how much it could hurt on the private set × how little we know.
Grounded in two measurements: at the arena's real IoU 0.5, zero-shot
temporal macro F1 is **0.025** (not the 0.049 our lenient 0.2 reports);
and train has **no event longer than 30.0s** while test Level 3 has
125s / 75s / 45s events.

1. **Private set is ~28 videos from held-out sources.** Everything tuned on
   the 34 public videos is tuned to noise for classes with n≤2 there
   (`wrong_way_driving`, `stalled_or_broken_down_vehicle` have one video
   each). → Don't tune per-class thresholds on those. Validate temporal
   behaviour on train's 1,900 timestamped events instead.
2. **Long videos.** `PRIORS.merge_gap` is 2–10s, built for 5s events. A
   125s event fragmented into 20s chunks scores zero and the fragments count
   against you. → Length-aware merging (30–60s gaps for slow classes on long
   videos); validate on the 88 train events ≥30s. Prefer fewer, longer
   intervals at Level 2/3.
3. **No density signal anywhere.** Stalled-vs-jam, congestion-vs-rush-hour,
   wrong-way-in-dense-traffic are all dense/sparse discriminations. The 24
   `vehicle_blocking_traffic` false positives are probably this. → Nano
   detector for per-frame box counts; same detector unlocks trajectory.
4. **Camera type/orientation.** Confirmed drone/dashcam/CCTV/elevated/
   compilation in public test; train camera balance never checked. → Sample
   train frames per class. Ensemble zero-shot (camera-agnostic) with the
   head. Detect scene cuts and don't start intervals at them.
5. **Asymmetric cost not yet applied.** Any prediction on a normal Level-2/3
   video = 0 for that video. Zero-shot flagged 5 of 6 normal videos. →
   Every unknown is a reason to predict *less*. Bias toward silence.
6. **Unmeasured:** Stage 2 VLM (never run), runtime (latency bonus), the
   arena's run budget (implied limited, never stated — find out before the
   first upload).

## 7. What the train data actually looks like (contact sheet, 3 frames/class)

Sampled 3 random videos per corrected class and looked at a frame from each
(`scratchpad/train_camera_contact_sheet.jpg`). Findings, most serious first:

- **`normal` is 72% non-traffic aerial footage** — cornfields, mountains,
  rivers ("routine aerial activity with no target anomaly", 699/973). Only
  28% describes traffic. A head trained on this as-is learns the shortcut
  "no road = normal, road = anomaly" and will flag every normal traffic
  scene on the private set — the single most expensive failure under the
  arena's rules. Mitigation: subsample aerial normals when training; the
  ~109 rows we relabeled *to* normal from the noisy folders are
  traffic-normal clips from the same cameras as the trajectory classes —
  exactly the hard negatives the head needs, keep all of them.
- **`stalled`, `vehicle_blocking`, `wrong_way` each come from ~one camera.**
  All three samples per class are the same view, and blocking + wrong_way
  are the *same* night overpass camera. The head cannot separate those two
  by appearance and will have memorized a camera, not a concept — and the
  data doc says private-set sources are held out. Policy: don't trust the
  head for these three; zero-shot + VLM gate instead.
- **`road_spill_or_debris` has zero genuine train examples** once the two
  explosion rows are correctly moved to `fire` (now a durable
  `MANUAL_OVERRIDES` in `scripts/relabel_train.py` — a by-hand CSV patch was
  silently reverted by a re-run, hence the dict). Zero-shot-only by
  necessity.
- **Watermarks and compilation overlays** ("CarCrashesTime", "LiveLeak",
  "21+") appear in accident and fighting clips — a learnable shortcut.
  Nothing to do today except know it's there.
- Camera mix elsewhere is reasonable: accident (dashcam + fixed CCTV),
  congestion (DOT cams + aerial), flood/smoke/loitering (mostly aerial),
  fire (mixed).
- **Stage 2 pivot (12:55):** the MLX Qwen2.5-VL-3B download stalled (dead
  socket, 63 MB frozen), and the CDN measures ~380 KB/s — ~90 min for the
  2 GB model. Added `OllamaVerifier` in `vad/verify.py` (same interface,
  talks to the local Ollama server) so Stage 2 can run today on
  `gemma3:27b`, which was already on disk. Use `predict --verify --vlm
  ollama:gemma3:27b`. 27B is the slow second-opinion tier, not the default
  gate — Qwen-3B stays the default once its download lands (resuming in
  the background). Also added, default-off: `predict --zeroshot-classes`
  (route named classes to zero-shot even in head mode) and
  `predict --long-merge` (length-aware merging for slow classes on >120 s
  videos, blind spot #2).

## 8. The arena scores OUR test videos — and the leaderboard calibrated our scorer

Confirmed by the user (and by the submission example using ids `T001`, `T002`,
`T025`): the arena benchmark runs on the same T001–T034 public test videos we
already have and have already encoded. No separate download.

**We hold `ground_truth.csv` for those videos.** The dataset doc says that's
deliberate ("so teams can validate their output and scoring pipeline"), so
tuning against it is legitimate. Submitting labels *as* predictions would not
be — it would score well and teach us nothing about the private evaluation
set, which is what final judging uses. Everything we submit comes from model
output.

**Leaderboard calibration (a real find).** Point caps are L1=25, L2=35,
L3=40 (total 100). Two entrants — M B Thejesshwar and Jay Kelani — both show
exactly **L2 = 11.7, L3 = 0.0**. Our scoring emulator computes "submit
nothing" as exactly 11.7 / 0.0, because 35 x (1/3) = 11.67. So:
  - the emulator matches real arena behaviour,
  - ~1/3 of arena L2 videos are normal (same as our public set: 2 of 6),
  - **11.7 on L2 is the "did nothing" floor** — scoring below it is worse
    than silence.

**Measured strategy comparison** (zero-shot Stage 1 only, emulated):

| strategy | L1/25 | L2/35 | L3/40 | total |
|---|---|---|---|---|
| predict everywhere | 12.0 | 6.5 | 6.0 | 24.5 |
| silent on L2, predict on L3 | 12.0 | **11.7** | 6.0 | **29.6** |
| silent on both | 12.0 | 11.7 | 0.0 | 23.6 |

Our raw L2 predictions (6.5) score **worse than doing nothing** (11.7) — the
0.83 false-alarm rate, in points. Do not hard-code "silent on L2" though:
that wins only because our localisation is currently worthless at IoU 0.5.
The fix is the VLM gate killing false alarms, which flips the sign. That run
is what's in flight.

`predict` now writes a `<pred>.runtime.json` sidecar with real per-video
timings and VLM call latencies; `submission.py` turns it into the
`model_runtimes` block (p50/p95/max, average consistent with total/calls
within 2%) plus `max_parallel_videos`. That block is required on every video
and is where the latency bonus comes from.

## 9. Submission strategy, measured (not guessed)

`scripts/arena_score.py` emulates the arena rules; it reproduces the
leaderboard's "did nothing" floor exactly (L2 11.7 / L3 0.0), so its ordering
is trustworthy. Composition of `out/submission.json`:

| level | source | pts | why |
|---|---|---|---|
| L1 | VLM classifies directly (`scripts/l1_classify.py`) | 14.1/25 | gate can only drop, never relabel |
| L2 | **silent** | 11.7/35 | our predictions score 6.5 — worse than nothing |
| L3 | zero-shot intervals, **ungated** | 6.0/40 | gate cost 2 pts by dropping T034's only interval |
| | | **31.7** | would rank 6th of 9 |

Integrity fix: the first build reported `end_to_end_internal_time_ms` of
24.6 ms for T029, a 240 s video, because cached features skipped decoding.
The spec requires decoding to be included, so that would have overstated the
latency bonus by ~4 orders of magnitude. `predict --honest-timing` forces a
real decode+encode measurement; use it for any run that gets uploaded.

Known ugly spot: T033 gets 18 predicted events, none of the right class
(truth is 2x traffic_accident). Only the best-overlapping one can match and
the rest count against you; it still beats silence because of the alert
credit, but it's the honest state of L3.

## 10. Models actually available (checked, not assumed)

- Local now: `mlx-community/Qwen2.5-VL-3B-Instruct-4bit` (default gate,
  ~2-5 s/call) and Ollama `gemma3:27b` (~29-64 s/call, second-opinion tier).
- **Qwen3 was NOT on the machine** despite being assumed. It is available:
  mlx-community ships Qwen3-VL in 2B / 4B / 8B / 32B MLX 4-bit. Our
  `mlx-vlm` is 0.6.17, well past the 0.3.4 those conversions need.
- The HF CDN stalled at ~380 KB/s midday (one download froze at 63 MB) but
  measured **8.6 MB/s** later — the stall was transient, not a hard limit.
  Pulling Qwen3-VL-8B to test on L1, where model judgment is the bottleneck.
