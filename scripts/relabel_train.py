"""Fix folder-level label noise using each row's own `description_summary`.

Why this exists: several train class folders don't match their own name.
`road_spill_or_debris` is 99% collision narratives (debris is a mentioned
side-effect of a crash, not the primary event); `stalled_or_broken_down_vehicle`
is ~55% the same. Training the head on the folder label as-is teaches it
wrong associations. This is a keyword heuristic, not an oracle — it's a first
pass, and the sprint plan calls for spot-checking a sample of its output by
hand before trusting it fully.

Non-destructive: writes `ground_truth_relabeled.csv` next to the original
`ground_truth.csv` in each class folder. Nothing here overwrites the source
files. `vad/io.py` prefers the relabeled file when present, falls back to the
original otherwise.

Priority when a description matches more than one class's keywords: an
accident described takes priority over everything else, because a collision
narrative that also mentions scattered debris or a vehicle stalling
afterward is a traffic_accident event first — the debris/stall is a
downstream detail, not the primary anomaly type this taxonomy means by
those other class names.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT.parent / "data" / "train" if (ROOT.parent / "data" / "train").exists() else ROOT.parent / "train"

LABELS = [
    "traffic_accident", "traffic_congestion", "stalled_or_broken_down_vehicle",
    "vehicle_blocking_traffic", "wrong_way_driving", "road_spill_or_debris",
    "waterlogging_or_flood", "fire", "smoke", "fighting_or_violence",
    "loitering_or_suspicious_presence",
]

# Order matters: checked top-to-bottom, first match wins EXCEPT
# traffic_accident, which is checked first regardless of folder (see above).
SIGNATURES = {
    "traffic_accident": r"collision|collide|crash|rear-end|t-bone|impact|overturn|struck|\bhits?\b|smash",
    "traffic_congestion": r"congest|traffic jam|bumper.to.bumper|\bqueue|barely mov|heavy traffic|slow-moving|gridlock",
    "stalled_or_broken_down_vehicle": r"stalled|broken.down|disabled vehicle|hazard light|breakdown|stopped abnormally|stopped on the shoulder",
    "vehicle_blocking_traffic": r"\bblocks?\b|blocking|blocked|obstruct|illegally occup|double-park|imped",
    "wrong_way_driving": r"wrong.way|against the flow|opposite direction of traffic|counterflow|facing oncoming",
    "road_spill_or_debris": r"\bdebris\b|\bspill\b|scattered (object|cargo)|fallen load",
    "waterlogging_or_flood": r"\bflood|waterlog|submerg|standing water|inundat",
    "fire": r"\bfire\b|\bflame|burning|\bblaze|arson",
    "smoke": r"\bsmoke\b|smoky|smog|\bhaze",
    "fighting_or_violence": r"\bfight|violen|assault|\bpunch|\battack|brawl|altercation",
    "loitering_or_suspicious_presence": r"\bloiter|\blinger|suspicious|remains? (beside|near)|prolonged period",
}
NORMAL_SIG = (r"without interruption|no interference|without interference|standard flow|steady speed|"
              r"clear of obstruction|normal flow|uneventful|without incident|maintains? a steady|"
              r"free-flowing|free flowing|without any obstruction|lack of congestion|"
              r"without conflict|no obstruction")

# A keyword hit doesn't count if it's negated nearby -- "no obstructions",
# "without interference due to the lack of congestion", "free of debris" all
# describe the ABSENCE of the thing, and a dumb positive-keyword match reads
# those backwards. Found by spot-checking wrong_way_driving's output: 4
# genuinely normal ("free-flowing... no obstructions") descriptions were
# getting pushed to vehicle_blocking_traffic / traffic_congestion because
# "obstruct"/"congest" appeared right after a negation word.
NEGATION = r"\b(no|not|without|lack of|absence of|free of|free from|never|none|nor)\b"
NEGATION_WINDOW = 45  # chars to look back from the match start


def _matches(d: str) -> set[str]:
    out = set()
    for cls, sig in SIGNATURES.items():
        for m in re.finditer(sig, d):
            window = d[max(0, m.start() - NEGATION_WINDOW):m.start()]
            if re.search(NEGATION, window):
                continue
            out.add(cls)
            break
    return out


# Hand-verified corrections the heuristic can't make. Applied last, so they
# survive re-runs (a previous by-hand CSV patch was silently reverted the
# next time this script ran -- don't patch the output, patch this dict).
#   TR00210 / TR02845: "massive explosion ... plume of fire, black smoke, and
#   debris" -- matched the road_spill_or_debris folder on "debris" but the
#   primary event is the fire/explosion. With these moved, road_spill_or_debris
#   has ZERO genuine train examples.
MANUAL_OVERRIDES = {
    "TR00210": "fire",
    "TR02845": "fire",
}


def classify(desc: str, folder_cls: str) -> tuple[str, str]:
    """Returns (corrected_class, confidence). confidence in {high, medium, low}."""
    d = desc.lower().strip()
    if not d:
        return folder_cls, "low"  # nothing to go on, keep folder label, flag for review

    matches = _matches(d)
    is_normal_ish = bool(re.search(NORMAL_SIG, d)) and not matches

    if "traffic_accident" in matches:
        # a described collision is the primary event even if debris/stall/etc
        # are also mentioned as a consequence
        return "traffic_accident", "high"
    if folder_cls == "normal" and not matches:
        # folder_cls "normal" is never itself a SIGNATURES key, so it can
        # never satisfy `folder_cls in matches` below -- handle explicitly.
        # No anomaly keyword hit at all is the actual positive signal here.
        return "normal", "high"
    if folder_cls in matches:
        return folder_cls, "high"
    if len(matches) == 1:
        return next(iter(matches)), "medium"
    if is_normal_ish:
        return "normal", "medium"
    if len(matches) > 1:
        return folder_cls, "low"   # ambiguous multi-match, don't guess
    return folder_cls, "low"       # no signal either way, don't guess


def relabel_folder(folder: Path) -> dict:
    gt = folder / "ground_truth.csv"
    if not gt.exists():
        return {}
    with open(gt, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    fieldnames = list(rows[0].keys())
    extra = ["orig_class_name", "label_confidence"]
    out_fields = fieldnames + [c for c in extra if c not in fieldnames]

    changed = []
    counts = {"high": 0, "medium": 0, "low": 0, "manual": 0}
    for r in rows:
        folder_cls = r.get("class_name", folder.name)
        desc = r.get("description_summary", "") or ""
        vid = str(r.get("video_id", "")).strip()
        if vid in MANUAL_OVERRIDES:
            corrected, conf = MANUAL_OVERRIDES[vid], "manual"
        else:
            corrected, conf = classify(desc, folder_cls)
        r["orig_class_name"] = folder_cls
        r["label_confidence"] = conf
        counts[conf] += 1
        if corrected != folder_cls:
            r["class_name"] = corrected
            r["is_anomaly"] = "false" if corrected == "normal" else "true"
            changed.append((r.get("video_id"), folder_cls, corrected, conf, desc[:100]))

    out = folder / "ground_truth_relabeled.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        w.writerows(rows)

    return {"folder": folder.name, "n": len(rows), "changed": changed, "counts": counts}


def main():
    if not TRAIN.is_dir():
        sys.exit(f"train dir not found at {TRAIN}")
    summary = []
    for d in sorted(TRAIN.iterdir()):
        if d.is_dir() and (d / "ground_truth.csv").exists():
            res = relabel_folder(d)
            if res:
                summary.append(res)

    print(f"{'folder':36s} {'n':>5s} {'changed':>8s} {'high':>6s} {'med':>6s} {'low':>6s} {'manual':>7s}")
    total_changed = 0
    all_changes = []
    for s in summary:
        n_changed = len(s["changed"])
        total_changed += n_changed
        c = s["counts"]
        print(f"{s['folder']:36s} {s['n']:5d} {n_changed:8d} {c['high']:6d} {c['medium']:6d} {c['low']:6d} {c['manual']:7d}")
        all_changes += [(s["folder"], *ch) for ch in s["changed"]]

    print(f"\ntotal rows relabeled: {total_changed}")
    print(f"wrote ground_truth_relabeled.csv into each folder (originals untouched)")

    review = ROOT / "out" / "relabel_review_sample.txt"
    review.parent.mkdir(parents=True, exist_ok=True)
    with open(review, "w") as f:
        for folder, vid, orig, new, conf, desc in all_changes:
            f.write(f"[{folder}] {vid}: {orig} -> {new} ({conf})\n  {desc}\n\n")
    print(f"full change list -> {review}  ({len(all_changes)} rows, for spot-checking)")


if __name__ == "__main__":
    main()
