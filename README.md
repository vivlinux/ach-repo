# AHC Visual Intelligence Hackathon — cascaded VAD on an M2 Mac

Near-real-time video anomaly detection that runs entirely on Apple Silicon. No CUDA,
no bitsandbytes, no Unsloth (all CUDA-only). The design works *with* a laptop instead
of pretending it is a GPU box.

## Architecture

```
video ──► sample @2fps ──► SigLIP vision tower (frozen, MPS) ──► frame embeddings [D]
                                                                        │
                                          sliding window 16 frames / 8s │ hop 2s
                                                                        ▼
                                      temporal transformer head (~1M params, trained)
                                                                        ▼
                                     per-window scores over 12 classes  ──► hysteresis
                                                                             + per-class
                                                                             min duration
                                                                        ▼
                                                        candidate intervals (few per video)
                                                                        ▼
                                    Qwen2.5-VL-3B-4bit on MLX — yes/no gate, verification only
                                                                        ▼
                                                 predictions.csv (level 1 + level 2/3 rows)
```

Why this shape:

- **The encoder is frozen and its outputs are cached.** Encoding the dataset is a
  one-time cost. After that, training the head, sweeping thresholds and re-running
  inference are seconds-to-minutes operations. On a laptop that is the difference
  between five experiments and fifty.
- **The head is temporal, not per-frame.** Congestion, loitering and a stalled vehicle
  are only anomalies because of persistence. A per-frame classifier cannot see that.
- **Per-class temporal priors** live in `vad/config.py` (`PRIORS`). An accident fires on
  a single window; a stalled vehicle needs 8 seconds of persistence. This is the part
  the problem statement is really asking about, and it is one dict you can tune live.
- **The VLM only sees candidates.** It never has to keep up with the stream, so the
  real-time budget is spent on the cheap stage while false alarms get killed by the
  expensive one. This is the Cerberus cascade idea.
- **Nothing hosted is in the runtime path.** Constraint satisfied by construction.

## Setup (do this first)

```bash
cd ahc-vad
./setup.sh                                   # venv + deps, prints MPS/MLX status
source .venv/bin/activate
export VAD_DATA=/path/to/dataset             # folder containing train/ and test/
python -m vad.cli inspect                    # confirms CSV columns parsed correctly
```

`inspect` is not optional — it tells you whether `videos.csv` / `ground_truth.csv`
were parsed, how many videos carry timestamps, and the class distribution. If the
counts look wrong, fix `vad/io.py` before burning an hour on encoding.

## Run order

```bash
# 1. Encode everything once (the long step; ~15 GB of video)
python -m vad.cli embed --splits test          # start with test, it is only ~56 min
python -m vad.cli embed --splits train

# 2. Zero-shot baseline — a full valid submission with no training at all
python -m vad.cli predict --split test --mode zeroshot --out out/pred_zs.csv
python -m vad.cli score  --split test --pred out/pred_zs.csv

# 3. Train the temporal head on cached features (minutes)
python -m vad.cli train --epochs 25
python -m vad.cli predict --split test --mode head
python -m vad.cli score  --split test

# 4. Tune thresholds without re-encoding anything (seconds per setting)
python -m vad.cli sweep --split test --mode head
python -m vad.cli predict --split test --scale 0.9

# 5. Add the VLM gate and re-score — this is where the false-alarm rate drops
python -m vad.cli predict --split test --verify --describe --out out/pred_verified.csv
python -m vad.cli score  --split test --pred out/pred_verified.csv

# 6. The demo
python -m vad.cli realtime /path/to/some.mp4
```

`realtime` streams a video once, prints alerts as they fire, and ends with a measured
frames/s and a realtime factor — "≈N concurrent streams on this machine". Show that
number rather than claiming real time.

## Suggested time-boxing (build window 11:00–18:00)

| Time | Do this |
|---|---|
| 11:00 | `inspect`, then kick off `embed --splits test` and let it run |
| 11:30 | Zero-shot baseline scored. You now have a submittable result. |
| 12:00 | `embed --splits train` running in a second terminal while you read the CSVs |
| 13:30 | Train the head, score it, compare against zero-shot |
| 14:30 | `sweep` for thresholds; hand-tune `PRIORS` per class against per-class F1 |
| 15:30 | Turn on `--verify`; measure the false-alarm drop with and without |
| 16:30 | `realtime` run, capture the throughput numbers, freeze the submission |
| 17:00 | Slides: the cascade diagram, the ablation table, the measured fps |

The ablation table is the demo. Four rows — zero-shot / +head / +temporal priors /
+VLM gate — with precision, recall and false-alarm rate. That answers the question the
hackathon actually poses ("can a small VLM do this reliably in real time") with
evidence rather than a single score.

## Knobs that matter

| Where | Knob | Effect |
|---|---|---|
| `config.PRIORS` | `min_dur`, `hi/lo`, `smooth_k` | per-event-type temporal behaviour; biggest lever on false alarms |
| `config.FPS/WIN/STRIDE` | 2 fps / 16 / 4 | context length vs latency. Long events want WIN 24–32 |
| `predict --scale` | 0.7–1.3 | scales all thresholds at once; the precision/recall dial |
| `predict --keep-score` | 0.85 | above this the VLM check is skipped, saving time |
| `--dtype float16` | | ~1.7x faster encoding; check for NaNs on MPS first |
| `config.EMBED_MODELS` | siglip2-base | swap to `siglip-so400m-patch14-384` if you have headroom and time |

## If you get stuck

- **MPS out of memory** — drop `--dtype float16`, or lower the batch in `Embedder.encode_images`.
- **SigLIP tokenizer error** — needs `sentencepiece` and `protobuf`; both are in requirements.
- **`mlx-vlm` fails to load Qwen2.5-VL** — it falls back to Qwen2-VL-2B and SmolVLM2
  automatically; or run without `--verify`, everything else still works.
- **Encoding is too slow** — cut `--fps 1.0` and raise `WIN` to keep the same context.
  Halves the cost with little accuracy loss for the slow event classes.
- **Class collapse to `normal`** — the head is heavily skewed; raise the `pos_weight`
  clip in `head.train_head` or drop a fraction of the normal videos.

## Files

```
vad/config.py       labels, prompts, per-class temporal priors, model ids
vad/io.py           resilient CSV parsing, video indexing, event objects
vad/frames.py       decode + uniform temporal sampling
vad/embed.py        frozen encoder on MPS, .npz caching
vad/windows.py      sliding windows + interval-overlap labelling
vad/zeroshot.py     prompt-ensembled text baseline
vad/head.py         temporal transformer + training (strong labels + MIL for weak)
vad/postprocess.py  smoothing, hysteresis, per-class min duration, intervals
vad/verify.py       MLX VLM yes/no gate + description generation
vad/score.py        level-1 and level-2/3 local metrics
vad/realtime.py     streaming demo with measured throughput
vad/cli.py          entrypoint
```

---

# Layer 2 — the deployment gaps

The problem statement is a clip-classification brief. These modules address what it
leaves out. Each is separately switchable so you can produce an honest ablation.

## New commands

```bash
python -m vad.cli novelty                    # build the open-set bank from train features
python -m vad.cli bench                      # measured throughput -> per-device capacity + $/stream-hour
python -m vad.cli stream --split test        # causal, online run: alerts, dedup, latency
python scripts/validate_motion.py            # ego-motion accuracy + negative control
```

`stream` is the deployment-shaped path. It never reads a frame after the alert
timestamp, so its numbers are comparable to what a live system would produce —
unlike `predict`, which sees whole clips.

```bash
# realistic uplink: 3% frame loss, a 20 s outage every 5 minutes
python -m vad.cli stream --split test --drop-rate 0.03 --outage-every 300 --outage-len 20

# with telemetry, alerts come out geo-tagged and dedup works across drones
python -m vad.cli stream --split test --telemetry flight.csv
```

## What each piece addresses

| Gap | Module | Status |
|---|---|---|
| Alert economics — false alerts *per drone-hour*, not clip precision | `alerts.py`, `eval_stream.py` | works |
| Cross-drone and cross-time dedup | `alerts.AlertManager` | works — verified merging two drones on one junction |
| Causal evaluation, latency-to-detect p50/p95 | `eval_stream.py`, `stream.py` | works |
| Adaptive compute — 1 fps idle, 10 fps on suspicion | `adaptive.py` | works — **71% encoder saving** vs fixed 10 fps on the unit test |
| Edge budget, $/stream-hour, W/stream | `bench.py` | works; device factors are order-of-magnitude |
| Link loss and store-and-forward | `stream.LinkModel` | works |
| Open-set — the thirteenth event type | `openset.py` | works; threshold calibrated on normal footage |
| Geo-tagged alerts from telemetry | `geo.py` | works at nadir; degrades with oblique gimbal pitch |
| Ego-motion compensation | `motion.py` | **conditional — read below** |
| Detector-free stopped-vehicle cue | `motion.stopped_object_score` | **does not work — off by default** |

## The two honest negatives

Both were found by testing against a control rather than by inspection, and both are
worth more in a demo than a slide claiming success.

**The stopped-object cue does not discriminate.** Residual flow after ego-motion
removal cannot separate a stalled car from empty tarmac — both are quiet. Adding an
objectness gate did not fix it: on a controlled synthetic pair the control scene
(traffic only, no stalled vehicle) fired *more* often than the positive. It is
therefore behind `--dwell-gate`, off by default. The correct version feeds
`WorldDwell.update_from_detections()` from a real detector; a nano-scale model costs
about 3 ms/frame on this hardware, which the budget can absorb.

**Visual ego-motion is not robust on low-texture ground.** On textured scenes the
estimator recovers a known pan to ~1% with >0.95 inliers. On smooth tarmac the
vehicles carry more trackable texture than the road, and the robust fit converges on
*traffic* motion instead — pinning to the vehicle velocity whatever the true camera
motion is. Neither the inlier ratio nor a spatial-spread test rejects it, because the
traffic is itself spread across the frame.

The fix is not a better visual estimator. A drone knows its own velocity and altitude,
so pass `expected_px` from `geo.expected_image_translation()` and the problem becomes
verification instead of estimation. Run `scripts/validate_motion.py` on your own
footage before trusting anything downstream of it.

The ego-motion **features** (translation, rotation, scale, residual statistics) are
still concatenated to the visual embedding and are useful to the head regardless —
they tell it whether the camera is moving, which the appearance features cannot.

## Judging

`JUDGING.md` has a weighted rubric, the eight questions that sort the field fastest,
and reference numbers for sanity-checking throughput and cost claims.
