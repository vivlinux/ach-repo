"""Central configuration. Everything tunable lives here."""
import os
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(os.environ.get("VAD_ROOT", Path(__file__).resolve().parent.parent))
DATA = Path(os.environ.get("VAD_DATA", ROOT / "data"))       # contains train/ and test/
CACHE = Path(os.environ.get("VAD_CACHE", ROOT / "cache"))    # embeddings
OUT = Path(os.environ.get("VAD_OUT", ROOT / "out"))          # checkpoints + predictions
for _p in (CACHE, OUT):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- labels
NORMAL = "normal"
LABELS = [
    "normal",
    "traffic_accident",
    "traffic_congestion",
    "stalled_or_broken_down_vehicle",
    "vehicle_blocking_traffic",
    "wrong_way_driving",
    "road_spill_or_debris",
    "waterlogging_or_flood",
    "fire",
    "smoke",
    "fighting_or_violence",
    "loitering_or_suspicious_presence",
]
ANOMALY_LABELS = [c for c in LABELS if c != NORMAL]
L2I = {c: i for i, c in enumerate(LABELS)}

# ---------------------------------------------------------------- sampling
FPS = 2.0          # frames sampled per second of video
WIN = 16           # frames per window  -> 8 s of context
STRIDE = 4         # window hop         -> 2 s
FRAME_SIZE = 224

# ---------------------------------------------------------------- models
# Stage 1: frozen image encoder (runs on MPS). Fallbacks are tried in order.
EMBED_MODELS = [
    os.environ.get("VAD_EMBED_MODEL", "google/siglip2-base-patch16-224"),
    "google/siglip-base-patch16-224",
    "openai/clip-vit-base-patch32",
]
# Stage 2: VLM (runs on MLX / Apple Silicon). Single family by choice --
# Qwen3-VL only. Qwen2.5-VL-3B and the Ollama/Gemma-3-27B escalation tier
# were both measured and dropped: 3B made judgment errors no threshold can
# fix (called both fighting clips and a loitering clip "normal"), and
# Gemma-3-27B was 29-64 s/call versus Qwen's 2-5 s, too slow to earn its
# place. One model is also one thing to explain in the write-up.
VLM_MODEL = os.environ.get("VAD_VLM_MODEL", "mlx-community/Qwen3-VL-8B-Instruct-4bit")
VLM_FALLBACKS = [
    "mlx-community/Qwen3-VL-4B-Instruct-4bit",
    "mlx-community/Qwen3-VL-2B-Instruct-4bit",
]

# ---------------------------------------------------------------- temporal priors
# Different events look different in time. These encode that directly.
#   min_dur   : an interval shorter than this is dropped
#   merge_gap : two intervals of the same class closer than this are merged
#   hi / lo   : hysteresis thresholds (enter above hi, stay until below lo)
#   smooth_k  : median filter width; 1 = none, so brief events survive
PRIORS = {
    "traffic_accident":               dict(min_dur=0.5,  merge_gap=2.0,  hi=0.55, lo=0.35, smooth_k=1),
    "traffic_congestion":             dict(min_dur=6.0,  merge_gap=8.0,  hi=0.55, lo=0.40, smooth_k=5),
    "stalled_or_broken_down_vehicle": dict(min_dur=8.0,  merge_gap=8.0,  hi=0.55, lo=0.40, smooth_k=5),
    "vehicle_blocking_traffic":       dict(min_dur=4.0,  merge_gap=6.0,  hi=0.55, lo=0.40, smooth_k=3),
    "wrong_way_driving":              dict(min_dur=1.5,  merge_gap=3.0,  hi=0.55, lo=0.38, smooth_k=1),
    "road_spill_or_debris":           dict(min_dur=3.0,  merge_gap=6.0,  hi=0.55, lo=0.40, smooth_k=3),
    "waterlogging_or_flood":          dict(min_dur=4.0,  merge_gap=10.0, hi=0.55, lo=0.40, smooth_k=5),
    "fire":                           dict(min_dur=1.5,  merge_gap=4.0,  hi=0.50, lo=0.35, smooth_k=1),
    "smoke":                          dict(min_dur=2.0,  merge_gap=4.0,  hi=0.50, lo=0.35, smooth_k=3),
    "fighting_or_violence":           dict(min_dur=1.0,  merge_gap=3.0,  hi=0.55, lo=0.38, smooth_k=1),
    "loitering_or_suspicious_presence": dict(min_dur=8.0, merge_gap=8.0, hi=0.60, lo=0.45, smooth_k=5),
}
DEFAULT_PRIOR = dict(min_dur=2.0, merge_gap=4.0, hi=0.55, lo=0.40, smooth_k=3)

# ---------------------------------------------------------------- zero-shot prompts
# Used by the CLIP/SigLIP text tower for the no-training baseline.
PROMPTS = {
    "normal": [
        "an ordinary street scene with traffic flowing normally",
        "a quiet road with nothing unusual happening",
        "normal pedestrian activity on a sidewalk",
        "an empty road seen from a drone",
    ],
    "traffic_accident": [
        "a road traffic accident with a crashed vehicle",
        "two vehicles that have collided on the road",
        "a car crash scene with damaged cars",
    ],
    "traffic_congestion": [
        "a traffic jam with many vehicles queued bumper to bumper",
        "heavy congested traffic barely moving on a road",
        "a long queue of stopped cars on a highway",
    ],
    "stalled_or_broken_down_vehicle": [
        "a broken down vehicle stopped on the shoulder of a highway",
        "a single stationary car stopped in a live traffic lane",
        "a disabled vehicle parked where it should not be",
    ],
    "vehicle_blocking_traffic": [
        "a vehicle stopped across the road blocking other traffic",
        "a truck obstructing the flow of traffic at a junction",
    ],
    "wrong_way_driving": [
        "a vehicle driving the wrong way against oncoming traffic",
        "a car travelling in the wrong direction on a one way road",
    ],
    "road_spill_or_debris": [
        "debris and scattered objects lying on the road surface",
        "a spill of material blocking part of the carriageway",
    ],
    "waterlogging_or_flood": [
        "a flooded road covered with standing water",
        "waterlogging with vehicles driving through deep water",
    ],
    "fire": [
        "an open fire with visible flames",
        "a burning vehicle or building with flames",
    ],
    "smoke": [
        "thick smoke rising from the ground",
        "a plume of smoke over a street",
    ],
    "fighting_or_violence": [
        "people fighting violently in the street",
        "a physical altercation between several people",
    ],
    "loitering_or_suspicious_presence": [
        "a person loitering alone in a restricted area",
        "someone standing around suspiciously for a long time at night",
    ],
}

# ---------------------------------------------------------------- verifier questions
# Asked as a yes/no gate on candidate intervals only.
VERIFY_Q = {
    "traffic_accident": "Do these frames show a traffic accident or collision (crashed or damaged vehicles)?",
    "traffic_congestion": "Do these frames show heavy traffic congestion, with vehicles queued and barely moving?",
    "stalled_or_broken_down_vehicle": "Do these frames show a vehicle stopped or broken down where it should not be, such as on a highway shoulder or in a live lane?",
    "vehicle_blocking_traffic": "Do these frames show a vehicle obstructing or blocking the flow of other traffic?",
    "wrong_way_driving": "Do these frames show a vehicle driving against the direction of the surrounding traffic?",
    "road_spill_or_debris": "Do these frames show a spill or debris lying on the road surface?",
    "waterlogging_or_flood": "Do these frames show a flooded or waterlogged road?",
    "fire": "Do these frames show visible flames or an active fire?",
    "smoke": "Do these frames show a visible plume of smoke?",
    "fighting_or_violence": "Do these frames show people physically fighting or an act of violence?",
    "loitering_or_suspicious_presence": "Do these frames show a person loitering or behaving suspiciously in a place they should not linger?",
}

SYSTEM_HINT = (
    "You are reviewing frames from a surveillance or drone camera, in time order. "
    "Answer with one word, yes or no, then a very short reason."
)
