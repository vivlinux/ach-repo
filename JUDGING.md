# Judging sheet — AHC Visual Intelligence Hackathon

The problem statement is a strong ML brief and a weak systems brief. These criteria
score the gap: whether a team built something FlytBase could fly, or a notebook that
scores well on 34 clips.

## Scoring

Team: ________________________  Judge: ________________  Score: ______ / 100

| # | Criterion | Weight | What full marks looks like | Score |
|---|---|---|---|---|
| 1 | **Detection quality** | 25 | Per-class F1 *and* false alerts per drone-hour measured on continuous footage, not just clip precision | /25 |
| 2 | **Real-time evidence** | 20 | Measured latency and throughput on stated hardware. Inference is causal — no frame after the alert timestamp was used | /20 |
| 3 | **Temporal event types** | 15 | Dwell-based classes handled with camera motion removed, or an explicit account of why not | /15 |
| 4 | **Cost / efficiency** | 15 | $/stream-hour computed. Adaptive sampling or a cascade, with the saving measured | /15 |
| 5 | **Deployment realism** | 10 | Edge/cloud split named with a compute budget. Behaviour under frame loss and link outage demonstrated | /10 |
| 6 | **Open-set behaviour** | 10 | A thirteenth event type produces something useful, not a forced label or silence | /10 |
| 7 | **Operator experience** | 5 | Dedup across time and drones, geo-tagged alerts, triage ranking, evidence clip | /5 |

Negative marks, applied after: **−10** for any headline number produced with lookahead;
**−10** for a hosted model in the runtime path (explicitly out of scope per the brief).

## What the statement leaves undefined — press on these

- "Real time" — no latency target. Ask for glass-to-alert p95.
- "Limited GPU" — no device named, so teams will target a desktop card. Ask what it runs on in the airframe.
- "Economical" — no cost target. Ask $/drone-hour at fifty drones.
- "False alarms matter" — no metric. Clip precision hides the base-rate problem; live footage is ~99.9% ordinary.
- "Not a fixed list" — but scoring uses twelve labels. Ask how a team gets credit for a correct thirteenth detection.
- Ego-motion is never mentioned, yet three of the twelve classes are defined by persistence, which is meaningless in a moving camera's pixel coordinates.
- Train mixes CCTV, dashcam and drone; nothing forces a drone-only test split.
- Thermal/IR is standard on inspection drones and no VLM handles it zero-shot.

## Eight questions that sort the field fast

1. Show me false alerts per hour on an hour of *ordinary* footage, not the test set.
2. Your model says a vehicle was stopped 30 seconds — where is camera motion removed?
3. Did your model see any frames from after the alert timestamp?
4. Where does this run, and what does it cost per drone-hour at fifty drones?
5. Same event, two drones — how many alerts does my operator get?
6. A gas leak plume appears. Not in your twelve. What happens?
7. Uplink drops for ninety seconds mid-flight. What does the system do?
8. Where does the alert's latitude and longitude come from?

Questions 1 and 3 eliminate most entries on their own.

## Credit intellectual honesty

A team that built a cue, tested it against a negative control, found no separation and
disabled it has done better engineering than a team that shipped the same cue untested
with a confident slide. Ask every team which of their components they tried to break,
and what broke. Reward the ones with an answer.

## Reference numbers for sanity-checking claims

| Quantity | Order of magnitude |
|---|---|
| 1080p30 H.264 per feed | ~4–8 Mbps → ~150 feeds saturates 1 Gbps |
| Operator false-alert budget | 1–2 per hour across the whole fleet |
| Jetson Orin NX | ~100 TOPS, 10–25 W, thermally throttled in an enclosure |
| Base ViT at 224px on Orin NX | roughly M2-laptop class, fp16 |
| Anomaly base rate in live urban footage | well under 1% of frames |
