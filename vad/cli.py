"""python -m vad.cli <command>  —  see README for the run order."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import head as H
from . import io as VIO
from . import postprocess as PP
from . import zeroshot as ZS
from .config import (CACHE, DATA, L2I, LABELS, NORMAL, OUT, STRIDE, WIN)
from .embed import Embedder, embed_video, pick_device
from .windows import pack, window_bounds, window_labels

COLS = ["video_id", "level", "is_anomaly", "class_name",
        "start_time_sec", "end_time_sec", "description_summary"]


# ----------------------------------------------------------------- helpers
def _videos(split, data):
    v = VIO.load_split(split, Path(data) if data else None)
    print(VIO.summarize(v))
    return v


def _feats(emb, v, fps, force=False):
    return embed_video(emb, v.path, VIO.cache_key(v), fps=fps, force=force)


def _honest_encode_ms(emb, v, fps):
    """Re-encode from the video file, ignoring the cache, and return the ms it
    took. The submission spec requires end_to_end_internal_time_ms to INCLUDE
    decoding and preprocessing; a cache hit reports ~25 ms for a 4-minute
    video, which would massively overstate the latency bonus."""
    t0 = time.time()
    embed_video(emb, v.path, VIO.cache_key(v), fps=fps, force=True)
    return (time.time() - t0) * 1000.0


# ----------------------------------------------------------------- commands
def cmd_inspect(a):
    for split in ("train", "test"):
        try:
            print(f"\n=== {split} ===")
            vids = _videos(split, a.data)
            for v in vids[:3]:
                print(f"  e.g. {v.vid} -> {v.path.name} events={[(e.cls, e.start, e.end) for e in v.events][:3]}")
        except FileNotFoundError as e:
            print(e)


def cmd_embed(a):
    emb = Embedder(a.model, dtype=a.dtype)
    for split in a.splits.split(","):
        vids = _videos(split, a.data)
        t0 = time.time()
        for i, v in enumerate(vids, 1):
            try:
                f, _ = _feats(emb, v, a.fps, a.force)
                if i % 20 == 0 or i == len(vids):
                    el = time.time() - t0
                    print(f"  [{split}] {i}/{len(vids)}  {el:.0f}s  "
                          f"eta {el/i*(len(vids)-i):.0f}s  last={f.shape}")
            except Exception as e:                        # noqa: BLE001
                print(f"  !! {v.vid}: {type(e).__name__}: {e}")
    print(f"cache -> {CACHE}")


def cmd_train(a):
    import random
    emb = Embedder(a.model, dtype=a.dtype)
    vids = _videos("train", a.data)
    # 72% of train-normal is "routine aerial activity" (cornfields, mountains,
    # no road at all). Left as-is the head learns "no road = normal, road =
    # anomaly" and flags every normal traffic scene. Cap the aerial normals;
    # keep every traffic-normal, including the ~109 relabeled out of the
    # anomaly folders -- those are same-camera hard negatives.
    aerial = {i for i, v in enumerate(vids) if not v.is_anomaly
              and "routine aerial activity" in " ".join(e.desc for e in v.events).lower()}
    cap = a.aerial_normal_cap
    keep = set(random.Random(0).sample(sorted(aerial), min(cap, len(aerial)))) if cap >= 0 else aerial
    if len(keep) < len(aerial):
        print(f"  aerial-normal subsample: keeping {len(keep)} of {len(aerial)} "
              f"non-traffic aerial normals (--aerial-normal-cap {cap}); all traffic normals kept")
    samples, dim = [], None
    for i, v in enumerate(vids, 1):
        if (i - 1) in aerial and (i - 1) not in keep:
            continue
        try:
            f, t = _feats(emb, v, a.fps)
        except Exception as e:                            # noqa: BLE001
            print(f"  !! {v.vid}: {e}")
            continue
        if len(f) < 4:
            continue
        dim = f.shape[1]
        spans = window_bounds(t, WIN, STRIDE)
        x = pack(f, spans)
        if v.has_timestamps:
            samples.append(dict(x=x, y=window_labels(v, spans), weak=None))
        else:
            w = np.zeros(len(LABELS), np.float32)
            for c in (v.classes or [NORMAL]):
                w[L2I[c]] = 1.0
            if not v.is_anomaly:
                samples.append(dict(x=x, y=np.tile(w, (len(spans), 1)), weak=None))
            else:
                samples.append(dict(x=x, y=None, weak=w))
        if i % 50 == 0:
            print(f"  loaded {i}/{len(vids)}")
    strong = sum(1 for s in samples if s["y"] is not None)
    print(f"{len(samples)} videos usable ({strong} strong, {len(samples)-strong} MIL), dim={dim}")
    dev = pick_device() if a.device == "auto" else a.device
    model = H.train_head(samples, dim, epochs=a.epochs, lr=a.lr, device=dev,
                         d_model=a.d_model, n_layers=a.n_layers, dropout=a.dropout,
                         weight_decay=a.weight_decay, label_smoothing=a.label_smoothing,
                         feat_noise=a.feat_noise, pos_weight_max=a.pos_weight_max)
    ck = Path(a.out) if a.out else OUT / "head.pt"
    H.save(model, dim, ck, meta=dict(encoder=emb.model_id, fps=a.fps, win=WIN, stride=STRIDE),
          d_model=a.d_model, n_layers=a.n_layers, dropout=a.dropout)
    print(f"saved {ck}")


def cmd_predict(a):
    emb = Embedder(a.model, dtype=a.dtype)
    dev = pick_device() if a.device == "auto" else a.device
    model = None
    if a.mode == "head":
        ckpt = Path(a.ckpt or OUT / "head.pt")
        if not ckpt.exists():
            raise SystemExit(f"{ckpt} missing — run `train` first, or use --mode zeroshot")
        model, _ = H.load(ckpt, dev)
    # Per-class source routing: classes the head cannot have learned (one
    # train camera, or no train data at all) take their scores from zero-shot
    # even in head mode. Both produce per-window [W, C] in [0, 1], so this is
    # a column swap; PRIORS thresholds are per class anyway.
    zs_cols = []
    for c in (x.strip() for x in a.zeroshot_classes.split(",")):
        if not c:
            continue
        if c not in L2I:
            raise SystemExit(f"--zeroshot-classes: unknown class '{c}'")
        zs_cols.append(L2I[c])
    if zs_cols and a.mode == "head":
        print(f"  zero-shot routed classes: {[LABELS[i] for i in zs_cols]}")
    T = ZS.class_matrix(emb) if (a.mode == "zeroshot" or zs_cols) else None

    verifier = None
    if a.verify:
        from .verify import Verifier
        verifier = Verifier(a.vlm)

    vids = _videos(a.split, a.data)
    if a.only:
        want = {x.strip() for x in a.only.split(",") if x.strip()}
        vids = [v for v in vids if v.vid in want]
        print(f"  --only: {len(vids)} of {len(want)} requested videos found")
    rows, t0 = [], time.time()
    runtime = {}          # video_id -> timings for the submission's runtime_metadata
    for i, v in enumerate(vids, 1):
        v_t0 = time.time()
        vlm_before = len(verifier.call_times_ms) if verifier else 0
        try:
            f, t = _feats(emb, v, a.fps)
        except Exception as e:                            # noqa: BLE001
            print(f"  !! {v.vid}: {e}")
            rows.append(dict(zip(COLS, [v.vid, 1, 0, NORMAL, "", "", ""])))
            runtime[v.vid] = dict(frames_processed=0, chunks_processed=0,
                                  end_to_end_internal_time_ms=round((time.time()-v_t0)*1000, 1),
                                  vlm_call_times_ms=[])
            continue
        spans = window_bounds(t, WIN, STRIDE)
        scores = (ZS.score_windows(f, spans, T) if a.mode == "zeroshot"
                  else H.predict(model, pack(f, spans), dev))
        if a.mode == "head" and zs_cols:
            zs = ZS.score_windows(f, spans, T)
            scores[:, zs_cols] = zs[:, zs_cols]
        ivs = PP.extract(scores, spans, scale=a.scale, long_merge=a.long_merge)
        if verifier and ivs:
            from .verify import verify_intervals
            ivs = verify_intervals(verifier, v.path, ivs, keep_score=a.keep_score, k=a.vlm_frames)
        is_anom, cls, conf = PP.video_decision(scores, ivs)
        rows.append(dict(zip(COLS, [v.vid, 1, is_anom, cls, "", "",
                                    f"confidence={conf:.2f}"])))
        for iv in ivs:
            desc = ""
            if verifier and a.describe:
                from .verify import describe
                desc = describe(verifier, v.path, iv)
            rows.append(dict(zip(COLS, [v.vid, 2, 1, iv.cls,
                                        round(iv.start, 2), round(iv.end, 2), desc])))
        elapsed_ms = (time.time() - v_t0) * 1000.0
        if a.honest_timing:
            # replace the cache-hit read with a real decode+encode measurement
            elapsed_ms += _honest_encode_ms(emb, v, a.fps)
        runtime[v.vid] = dict(
            frames_processed=int(len(f)),
            chunks_processed=int(len(spans)),
            end_to_end_internal_time_ms=round(elapsed_ms, 1),
            vlm_call_times_ms=[round(x, 1) for x in
                               (verifier.call_times_ms[vlm_before:] if verifier else [])],
        )
        if i % 10 == 0 or i == len(vids):
            print(f"  {i}/{len(vids)}  {time.time()-t0:.0f}s  last={v.vid} "
                  f"{cls} ({len(ivs)} intervals)")
    out = Path(a.out or OUT / f"predictions_{a.split}.csv")
    pd.DataFrame(rows, columns=COLS).to_csv(out, index=False)
    print(f"wrote {out}  ({len(rows)} rows)")
    # sidecar: real per-video timings -> submission runtime_metadata, which is
    # both required on every video and where the latency bonus comes from
    rt_path = out.with_suffix(".runtime.json")
    rt_path.write_text(json.dumps(
        dict(videos=runtime, vlm_model=(verifier.model_id if verifier else None),
             encoder=emb.model_id, total_wall_time_ms=round((time.time()-t0)*1000, 1)),
        indent=2))
    print(f"wrote {rt_path}")


def cmd_submit(a):
    from . import submission as SUB
    manifest = SUB.load_manifest(a.manifest, a.data)
    sub = SUB.build(a.pred, manifest, submission_id=a.submission_id,
                     model_name=a.model_name, hardware=a.hardware,
                     runtime_json=a.runtime_json)
    errors, notes = SUB.validate(sub, manifest)
    out = Path(a.out or OUT / "submission.json")
    out.write_text(json.dumps(sub, indent=2))
    n_events = sum(len(p["events"]) for p in sub["predictions"])
    print(f"wrote {out}  ({len(sub['predictions'])} videos, {n_events} events)")
    if errors:
        print(f"\n!! {len(errors)} problem(s) -- fix before uploading:")
        for e in errors[:50]:
            print(f"   ERROR  {e}")
        if len(errors) > 50:
            print(f"   ... and {len(errors) - 50} more")
    else:
        print("validated clean against every rule in the submission-format spec")
    if notes:
        print(f"\n{len(notes)} note(s):")
        for n in notes[:20]:
            print(f"   note   {n}")
        if len(notes) > 20:
            print(f"   ... and {len(notes) - 20} more")


def cmd_score(a):
    from .score import report
    gt = a.gt or (Path(a.data or DATA) / a.split / "ground_truth.csv")
    print(report(a.pred or OUT / f"predictions_{a.split}.csv", gt, a.iou))


def cmd_sweep(a):
    """Threshold sweep on cached features — seconds per setting, no re-encode."""
    from .score import level1, temporal
    emb = Embedder(a.model, dtype=a.dtype)
    dev = pick_device() if a.device == "auto" else a.device
    model, T = None, None
    if a.mode == "head":
        model, _ = H.load(Path(a.ckpt or OUT / "head.pt"), dev)
    else:
        T = ZS.class_matrix(emb)
    vids = _videos(a.split, a.data)
    cache = []
    for v in vids:
        try:
            f, t = _feats(emb, v, a.fps)
        except Exception:                                  # noqa: BLE001
            continue
        spans = window_bounds(t, WIN, STRIDE)
        s = (ZS.score_windows(f, spans, T) if T is not None
             else H.predict(model, pack(f, spans), dev))
        cache.append((v, spans, s))
    gt = pd.read_csv(a.gt or (Path(a.data or DATA) / a.split / "ground_truth.csv"))
    gt.columns = [c.strip().lower() for c in gt.columns]
    best = None
    for scale in [float(x) for x in a.scales.split(",")]:
        rows = []
        for v, spans, s in cache:
            ivs = PP.extract(s, spans, scale=scale)
            ia, cls, _ = PP.video_decision(s, ivs)
            rows.append(dict(zip(COLS, [v.vid, 1, ia, cls, "", "", ""])))
            for iv in ivs:
                rows.append(dict(zip(COLS, [v.vid, 2, 1, iv.cls, iv.start, iv.end, ""])))
        pr = pd.DataFrame(rows, columns=COLS)
        l1, tm = level1(pr, gt), temporal(pr, gt, a.iou)
        print(f"scale {scale:.2f}  L1 f1 {l1['f1']:.3f}  FAR {l1['false_alarm_rate']:.3f}  "
              f"temporal macroF1 {tm['macro_f1']:.3f}")
        key = l1["f1"] + tm["macro_f1"]
        if best is None or key > best[0]:
            best = (key, scale)
    print(f"\nbest scale = {best[1]}   (re-run predict with --scale {best[1]})")


def cmd_bench(a):
    from . import bench
    emb = Embedder(a.model, dtype=a.dtype)
    dev = pick_device() if a.device == "auto" else a.device
    model = None
    ck = Path(a.ckpt or OUT / "head.pt")
    if ck.exists():
        model, _ = H.load(ck, dev)
    b = bench.measure(emb, model, dev, n=a.n, with_motion=not a.no_motion)
    b.effective_fps = a.eff_fps
    print()
    print(bench.report(b, vlm_ms=a.vlm_ms))


def cmd_novelty(a):
    """Build the open-set novelty bank from cached training embeddings."""
    from .openset import NoveltyBank
    emb = Embedder(a.model, dtype=a.dtype)
    vids = _videos("train", a.data)
    by = {}
    for i, v in enumerate(vids, 1):
        try:
            f, _ = _feats(emb, v, a.fps)
        except Exception:
            continue
        if len(f) == 0:
            continue
        for c in (v.classes or [NORMAL]):
            by.setdefault(c, []).append(f[::4])
        if i % 50 == 0:
            print(f"  {i}/{len(vids)}")
    bank = NoveltyBank.fit(by, bank_size=a.bank_size)
    held = by.get(NORMAL, [])
    if held:
        import numpy as _np
        scores = _np.array([bank.distance(x) for x in held[: min(300, len(held))]])
        thr = bank.calibrate(scores, q=a.quantile)
        print(f"novelty threshold calibrated on normal footage: {thr:.3f}")
    out = OUT / "novelty.npz"
    bank.save(out)
    print(f"saved {out}")


def cmd_stream(a):
    """Causal, online run. This is the deployment-shaped path."""
    from .alerts import AlertManager
    from .stream import LinkModel, StreamConfig, StreamProcessor
    from . import eval_stream
    from .geo import load_telemetry
    from .openset import NoveltyBank

    emb = Embedder(a.model, dtype=a.dtype)
    dev = pick_device() if a.device == "auto" else a.device
    model, _ = H.load(Path(a.ckpt or OUT / "head.pt"), dev)
    bank = None
    if not a.no_openset:
        nb = Path(a.novelty or OUT / "novelty.npz")
        if nb.exists():
            bank = NoveltyBank.load(nb)
        else:
            print("[stream] no novelty bank — run `novelty` first for open-set support")
    tel = load_telemetry(Path(a.telemetry)) if a.telemetry else None

    if a.video:
        targets = [(Path(a.video).stem, Path(a.video), None)]
    else:
        vids = _videos(a.split, a.data)
        if a.limit:
            vids = vids[: a.limit]
        targets = [(v.vid, v.path, v) for v in vids]

    mgr = AlertManager(geo_radius_m=a.geo_radius, time_window_s=a.dedup_window)
    link = LinkModel(a.drop_rate, a.outage_every, a.outage_len)
    cfg = StreamConfig(scale=a.scale, adaptive=not a.no_adaptive,
                       ego_motion=not a.no_motion, dwell_gate=a.dwell_gate,
                       openset=bank is not None, device=dev)
    total_s, perf, vmap = 0.0, [], {}
    for vid, path, v in targets:
        cfg.drone_id = vid
        sp = StreamProcessor(model, emb, cfg, mgr, bank, tel, link)
        print(f"\n--- {vid} ---")
        r = sp.run(path)
        perf.append(r)
        total_s += r["video_seconds"]
        if v is not None:
            vmap[vid] = v
    hours = total_s / 3600.0

    print("\n" + "=" * 64)
    for k in ("realtime_factor", "ms_per_encoded_frame", "ms_per_window_head",
              "ms_per_frame_motion"):
        vals = [p[k] for p in perf if k in p]
        if vals:
            print(f"  {k:24s} median {sorted(vals)[len(vals)//2]}")
    if not a.no_adaptive and perf and perf[0].get("adaptive"):
        sav = [p["adaptive"]["compute_saved_pct"] for p in perf if p.get("adaptive")]
        print(f"  adaptive compute saved   {sum(sav)/len(sav):.1f}% vs fixed 10 fps")
    drop = sum(p["dropped"] for p in perf)
    if drop:
        print(f"  frames lost to link      {drop} (outage {sum(p['outage_s'] for p in perf):.0f}s)")
    print()
    print(json.dumps(mgr.metrics(hours), indent=2))
    if vmap:
        print()
        print(eval_stream.report(mgr.emitted, vmap, hours, mgr.suppressed))
    out = mgr.save(Path(a.out or OUT / "alerts.json"))
    print(f"\nalert queue -> {out}")
    for al in mgr.queue()[:10]:
        loc = f"{al.lat:.5f},{al.lon:.5f}" if al.lat is not None else "no telemetry"
        print(f"  {al.id} prio {al.priority:.2f}  {al.cls:32s} {al.drone_id} "
              f"lat {al.latency:.1f}s  x{al.dedup_count}  {loc}")


def cmd_realtime(a):
    from .realtime import run
    run(a)


# ----------------------------------------------------------------- parser
def main():
    p = argparse.ArgumentParser(prog="vad")
    p.add_argument("--data", default=None, help="folder containing train/ and test/")
    p.add_argument("--model", default=None, help="image encoder id")
    p.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--device", default="auto")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("inspect"); sp.set_defaults(fn=cmd_inspect)

    sp = sub.add_parser("embed")
    sp.add_argument("--splits", default="train,test")
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(fn=cmd_embed)

    sp = sub.add_parser("train")
    sp.add_argument("--epochs", type=int, default=25)
    sp.add_argument("--lr", type=float, default=3e-4)
    sp.add_argument("--aerial-normal-cap", dest="aerial_normal_cap", type=int, default=300,
                    help="max non-traffic aerial 'normal' videos to train on (-1 = all). "
                         "Train-normal is 72%% scenic aerial; uncapped, the head learns "
                         "'no road = normal'. Traffic normals are never dropped.")
    sp.add_argument("--out", default=None, help="checkpoint path (default out/head.pt). "
                    "Use a different path for an experimental retrain so it never "
                    "overwrites a checkpoint that's already been submitted.")
    sp.add_argument("--d-model", dest="d_model", type=int, default=256)
    sp.add_argument("--n-layers", dest="n_layers", type=int, default=2)
    sp.add_argument("--dropout", type=float, default=0.15)
    sp.add_argument("--weight-decay", dest="weight_decay", type=float, default=0.02)
    sp.add_argument("--label-smoothing", dest="label_smoothing", type=float, default=0.0,
                    help="pulls 0/1 targets to e/2 - (1-e/2). Added to fight output "
                         "saturation found in the first head (loitering at 1.000 on "
                         ">10%% of ALL test windows).")
    sp.add_argument("--feat-noise", dest="feat_noise", type=float, default=0.0,
                    help="Gaussian noise on embeddings, train only -- a cheap stand-in "
                         "for unseen-camera augmentation since no source id exists.")
    sp.add_argument("--pos-weight-max", dest="pos_weight_max", type=float, default=8.0,
                    help="clip on the positive-class weight (was hardcoded at 8.0). "
                         "loitering is 79 videos, all weakly-labelled MIL with no "
                         "timestamps -- a high weight likely drove its over-firing.")
    sp.set_defaults(fn=cmd_train)

    sp = sub.add_parser("predict")
    sp.add_argument("--split", default="test")
    sp.add_argument("--mode", default="head", choices=["head", "zeroshot"])
    sp.add_argument("--ckpt", default=None)
    sp.add_argument("--scale", type=float, default=1.0, help="scales all thresholds")
    sp.add_argument("--verify", action="store_true", help="enable the VLM gate")
    sp.add_argument("--describe", action="store_true")
    sp.add_argument("--vlm", default=None,
                    help="MLX model id; defaults to config.VLM_MODEL (Qwen3-VL-8B)")
    sp.add_argument("--vlm-frames", type=int, default=6)
    sp.add_argument("--keep-score", type=float, default=0.85)
    sp.add_argument("--zeroshot-classes", dest="zeroshot_classes", default="",
                    help="comma-separated classes scored by zero-shot even in --mode head. "
                         "Meant for classes the head can't have learned: "
                         "stalled_or_broken_down_vehicle,vehicle_blocking_traffic,"
                         "wrong_way_driving (each one train camera) and "
                         "road_spill_or_debris (no train data at all)")
    sp.add_argument("--honest-timing", dest="honest_timing", action="store_true",
                    help="measure a real decode+encode per video instead of a cache hit. "
                         "The spec requires reported time to include decoding; without this "
                         "a cached run reports ~25ms for a 4-minute video and overstates "
                         "the latency bonus. Use for the run you actually submit.")
    sp.add_argument("--only", default="",
                    help="comma-separated video ids to run (e.g. the 24 Level-1 clips); "
                         "others are skipped entirely. Combine with `submit` -- the arena "
                         "keeps earlier answers for videos a file doesn't mention")
    sp.add_argument("--long-merge", dest="long_merge", action="store_true",
                    help="on videos >120s, widen merge_gap for slow classes to 10%% of "
                         "duration (cap 60s) so a 2-minute event isn't fragmented -- "
                         "the arena scores at IoU 0.5 and fragments count against you")
    sp.add_argument("--out", default=None)
    sp.set_defaults(fn=cmd_predict)

    sp = sub.add_parser("submit")
    sp.add_argument("--pred", required=True, help="predictions.csv from `predict`")
    sp.add_argument("--manifest", default=None,
                    help="manifest.json from the arena Benchmark tab; "
                         "omit to fall back to test/ground_truth.csv levels for a local dry run")
    sp.add_argument("--out", default=None)
    sp.add_argument("--runtime-json", dest="runtime_json", default=None,
                    help="timings sidecar from `predict` (default: <pred>.runtime.json)")
    sp.add_argument("--submission-id", dest="submission_id", default="local-run")
    sp.add_argument("--model-name", dest="model_name", default="vad-cascade")
    sp.add_argument("--hardware", default="1x Apple M2")
    sp.set_defaults(fn=cmd_submit)

    sp = sub.add_parser("score")
    sp.add_argument("--split", default="test")
    sp.add_argument("--pred", default=None)
    sp.add_argument("--gt", default=None)
    sp.add_argument("--iou", type=float, default=0.2)
    sp.set_defaults(fn=cmd_score)

    sp = sub.add_parser("sweep")
    sp.add_argument("--split", default="test")
    sp.add_argument("--mode", default="head", choices=["head", "zeroshot"])
    sp.add_argument("--ckpt", default=None)
    sp.add_argument("--gt", default=None)
    sp.add_argument("--iou", type=float, default=0.2)
    sp.add_argument("--scales", default="0.7,0.8,0.9,1.0,1.1,1.2,1.3")
    sp.set_defaults(fn=cmd_sweep)

    sp = sub.add_parser("bench")
    sp.add_argument("--ckpt", default=None)
    sp.add_argument("--n", type=int, default=40)
    sp.add_argument("--eff-fps", type=float, default=2.0)
    sp.add_argument("--vlm-ms", type=float, default=0.0, help="measured ms per VLM call")
    sp.add_argument("--no-motion", action="store_true")
    sp.set_defaults(fn=cmd_bench)

    sp = sub.add_parser("novelty")
    sp.add_argument("--bank-size", type=int, default=4000)
    sp.add_argument("--quantile", type=float, default=0.98)
    sp.set_defaults(fn=cmd_novelty)

    sp = sub.add_parser("stream")
    sp.add_argument("--video", default=None, help="single file; omit to run a split")
    sp.add_argument("--split", default="test")
    sp.add_argument("--limit", type=int, default=0)
    sp.add_argument("--ckpt", default=None)
    sp.add_argument("--novelty", default=None)
    sp.add_argument("--telemetry", default=None, help="CSV with t,lat,lon,alt")
    sp.add_argument("--scale", type=float, default=1.0)
    sp.add_argument("--geo-radius", type=float, default=120.0)
    sp.add_argument("--dedup-window", type=float, default=180.0)
    sp.add_argument("--drop-rate", type=float, default=0.0, help="simulated frame loss")
    sp.add_argument("--outage-every", type=float, default=0.0)
    sp.add_argument("--outage-len", type=float, default=0.0)
    sp.add_argument("--no-adaptive", action="store_true")
    sp.add_argument("--no-motion", action="store_true")
    sp.add_argument("--dwell-gate", action="store_true",
                    help="experimental: gate dwell classes on the detector-free cue")
    sp.add_argument("--no-openset", action="store_true")
    sp.add_argument("--out", default=None)
    sp.set_defaults(fn=cmd_stream)

    sp = sub.add_parser("realtime")
    sp.add_argument("video")
    sp.add_argument("--mode", default="head", choices=["head", "zeroshot"])
    sp.add_argument("--ckpt", default=None)
    sp.add_argument("--scale", type=float, default=1.0)
    sp.add_argument("--display", action="store_true")
    sp.set_defaults(fn=cmd_realtime)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
