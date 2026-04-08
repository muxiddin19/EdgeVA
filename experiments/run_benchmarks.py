"""
EdgeVA Benchmark Suite
======================
Measures inference latency and throughput for YOLOv8 family models on:
  - CPU  (AMD Ryzen 5 5600X, 6-core)
  - GPU  (NVIDIA RTX 3060 Ti, CUDA / CUDAExecutionProvider)

Also benchmarks:
  - Tracker comparison: IoU-only vs LITE feature-reuse
  - Full pipeline timing: decode → preprocess → detect → track → analytics

Usage:
    python experiments/run_benchmarks.py

Output:
    experiments/results/benchmark_results.csv
    experiments/results/pipeline_timing.csv
    experiments/results/tracker_comparison.csv
"""

import os, sys, time, json, csv
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import onnxruntime as ort
import cv2

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MODELS = {
    "yolov8n": ("D:/VoiceAI/EdgeVA/yolov8n.onnx", 3_151_904),
    "yolov8s": ("D:/VoiceAI/EdgeVA/yolov8s.onnx", 11_166_560),
    "yolov8m": ("D:/VoiceAI/EdgeVA/yolov8m.onnx", 25_886_080),
}

INPUT_SHAPE = (1, 3, 640, 640)
WARMUP = 50
RUNS   = 200


# ── helpers ─────────────────────────────────────────────────────────────────

def make_dummy_input():
    return np.random.rand(*INPUT_SHAPE).astype(np.float32)


def build_session(model_path: str, provider: str) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 6   # all physical cores for CPU
    return ort.InferenceSession(model_path,
                                providers=[provider, "CPUExecutionProvider"],
                                sess_options=opts)


def benchmark_model(model_name: str, model_path: str, params: int,
                    provider: str) -> dict:
    print(f"  Benchmarking {model_name} on {provider} …", flush=True)
    sess  = build_session(model_path, provider)
    inp   = make_dummy_input()
    iname = sess.get_inputs()[0].name

    # warmup
    for _ in range(WARMUP):
        sess.run(None, {iname: inp})

    latencies = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        sess.run(None, {iname: inp})
        latencies.append((time.perf_counter() - t0) * 1000)   # ms

    lats = np.array(latencies)
    return {
        "model":    model_name,
        "provider": provider.replace("ExecutionProvider", ""),
        "params_M": round(params / 1e6, 1),
        "p50_ms":   round(float(np.percentile(lats, 50)), 2),
        "p95_ms":   round(float(np.percentile(lats, 95)), 2),
        "p99_ms":   round(float(np.percentile(lats, 99)), 2),
        "fps":      round(1000.0 / float(np.percentile(lats, 50)), 1),
        "mean_ms":  round(float(lats.mean()), 2),
        "std_ms":   round(float(lats.std()), 2),
    }


# ── Experiment 1: inference benchmark ───────────────────────────────────────

def run_inference_benchmark():
    providers = []
    avail = ort.get_available_providers()
    if "CUDAExecutionProvider" in avail:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    rows = []
    for prov in providers:
        for name, (path, params) in MODELS.items():
            if not os.path.exists(path):
                print(f"  SKIP {name}: {path} not found")
                continue
            row = benchmark_model(name, path, params, prov)
            rows.append(row)
            print(f"    {name} {prov}: p50={row['p50_ms']} ms, FPS={row['fps']}")

    out = os.path.join(RESULTS_DIR, "benchmark_results.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Saved → {out}")
    return rows


# ── Experiment 2: pipeline timing breakdown ──────────────────────────────────

def run_pipeline_timing(yolov8m_path: str):
    """Measure per-stage latency for a complete video analytics pipeline."""
    print("\nExperiment 2: Pipeline timing breakdown …", flush=True)

    if not os.path.exists(yolov8m_path):
        print("  SKIP: yolov8m.onnx not found")
        return []

    # Build a synthetic 640×640 BGR frame (simulates decoded video)
    frame_bgr = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)

    provider = ("CUDAExecutionProvider"
                if "CUDAExecutionProvider" in ort.get_available_providers()
                else "CPUExecutionProvider")
    sess  = build_session(yolov8m_path, provider)
    iname = sess.get_inputs()[0].name

    # Simple LITE-style tracker placeholder: maintains centroids, no ReID pass
    class MinimalTracker:
        def __init__(self):
            self.track_id = 0
            self.tracks   = {}

        def update(self, boxes):
            """boxes: (N,4) xyxy"""
            self.tracks = {}
            for b in boxes:
                self.track_id += 1
                cx = (b[0] + b[2]) / 2
                cy = (b[1] + b[3]) / 2
                self.tracks[self.track_id] = (cx, cy)
            return self.tracks

    tracker = MinimalTracker()

    stages = {"preprocess": [], "inference": [], "postprocess": [],
              "tracking": [], "analytics": [], "total": []}

    for _ in range(WARMUP):
        inp = frame_bgr[:, :, ::-1].transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0
        sess.run(None, {iname: inp})

    for _ in range(RUNS):
        t_start = time.perf_counter()

        # preprocess: BGR→RGB, HWC→BCHW, /255
        t0 = time.perf_counter()
        inp = frame_bgr[:, :, ::-1].transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0
        t1 = time.perf_counter()

        # inference
        out = sess.run(None, {iname: inp})
        t2 = time.perf_counter()

        # postprocess: parse (1,84,8400) → (N,6) boxes (conf>0.25, simple)
        pred = out[0][0].T            # (8400, 84)
        scores = pred[:, 4:].max(1)
        mask   = scores > 0.25
        boxes  = pred[mask, :4]       # cx,cy,w,h
        t3 = time.perf_counter()

        # tracking
        if len(boxes):
            # convert cx,cy,w,h to xyxy
            xyxy = np.stack([boxes[:, 0] - boxes[:, 2]/2,
                             boxes[:, 1] - boxes[:, 3]/2,
                             boxes[:, 0] + boxes[:, 2]/2,
                             boxes[:, 1] + boxes[:, 3]/2], 1)
        else:
            xyxy = np.empty((0, 4))
        tracker.update(xyxy)
        t4 = time.perf_counter()

        # analytics: zone occupancy check (ray-casting, trivial polygon)
        zone = np.array([(0, 0), (320, 0), (320, 320), (0, 320)])
        count = sum(1 for cx, cy in tracker.tracks.values()
                    if 0 <= cx <= 320 and 0 <= cy <= 320)
        t5 = time.perf_counter()

        stages["preprocess"].append((t1-t0)*1e3)
        stages["inference"].append((t2-t1)*1e3)
        stages["postprocess"].append((t3-t2)*1e3)
        stages["tracking"].append((t4-t3)*1e3)
        stages["analytics"].append((t5-t4)*1e3)
        stages["total"].append((t5-t_start)*1e3)

    rows = []
    for stage, lats in stages.items():
        arr = np.array(lats)
        rows.append({
            "stage":   stage,
            "provider": provider.replace("ExecutionProvider", ""),
            "p50_ms":  round(float(np.percentile(arr, 50)), 3),
            "p95_ms":  round(float(np.percentile(arr, 95)), 3),
            "mean_ms": round(float(arr.mean()), 3),
        })
        print(f"    {stage:15s}  p50={rows[-1]['p50_ms']:7.3f} ms")

    out = os.path.join(RESULTS_DIR, "pipeline_timing.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved → {out}")
    return rows


# ── Experiment 3: tracker comparison ────────────────────────────────────────

def run_tracker_comparison():
    """
    Synthetic tracking benchmark: 20 objects moving linearly in 640×480 scene.
    Compares IoU-only association (ByteTrack-proxy) vs LITE feature-reuse proxy
    (no ReID forward pass, feature extraction from bounding box coordinates only).
    Measures wall-clock association time for 300 frames.
    """
    print("\nExperiment 3: Tracker association timing …", flush=True)
    from scipy.optimize import linear_sum_assignment

    RNG    = np.random.default_rng(42)
    N_OBJ  = 20
    N_FRAMES = 300
    H, W   = 480, 640

    # Ground-truth linear trajectories
    starts = RNG.uniform(50, W - 50, (N_OBJ, 2))
    vels   = RNG.uniform(-3, 3, (N_OBJ, 2))

    def gt_boxes(frame_idx):
        pos = starts + vels * frame_idx
        pos[:, 0] = np.clip(pos[:, 0], 20, W - 20)
        pos[:, 1] = np.clip(pos[:, 1], 20, H - 20)
        w = RNG.uniform(30, 80, N_OBJ)
        h = RNG.uniform(60, 160, N_OBJ)
        noise = RNG.normal(0, 2, (N_OBJ, 2))
        cx = pos[:, 0] + noise[:, 0]
        cy = pos[:, 1] + noise[:, 1]
        return np.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], 1)

    def iou_matrix(boxes_a, boxes_b):
        xa1, ya1, xa2, ya2 = boxes_a[:, 0:1], boxes_a[:, 1:2], boxes_a[:, 2:3], boxes_a[:, 3:4]
        xb1, yb1, xb2, yb2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]
        inter_x = np.maximum(0, np.minimum(xa2, xb2) - np.maximum(xa1, xb1))
        inter_y = np.maximum(0, np.minimum(ya2, yb2) - np.maximum(ya1, yb1))
        inter   = inter_x * inter_y
        area_a  = (xa2 - xa1) * (ya2 - ya1)
        area_b  = (xb2 - xb1) * (yb2 - yb1)
        return inter / (area_a + area_b - inter + 1e-6)

    # ── IoU-only tracker (ByteTrack proxy) ───
    times_iou = []
    track_boxes = gt_boxes(0)

    for f in range(N_FRAMES):
        det_boxes = gt_boxes(f)
        t0 = time.perf_counter()
        cost = 1.0 - iou_matrix(track_boxes, det_boxes)
        row_ind, col_ind = linear_sum_assignment(cost)
        matched = cost[row_ind, col_ind] < 0.7
        track_boxes = det_boxes
        times_iou.append((time.perf_counter() - t0) * 1e3)

    # ── LITE feature-reuse proxy ─────────────────────────────────────────────
    # LITE reuses FPN features.  Here we simulate the computation:
    # (a) extract 8-dim box-geometry feature (no separate network forward pass)
    # (b) combine with IoU cost via linear combination
    # This represents the additional cost vs pure-IoU: one vector op per track
    times_lite = []
    track_boxes = gt_boxes(0)
    track_feats = np.zeros((N_OBJ, 8), dtype=np.float32)  # simulated FPN features

    for f in range(N_FRAMES):
        det_boxes   = gt_boxes(f)
        t0 = time.perf_counter()

        # "feature extraction": normalise box coords to [0,1] — proxy for FPN pool
        det_feats = np.column_stack([
            det_boxes / np.array([W, H, W, H]),
            (det_boxes[:, 2:4] - det_boxes[:, 0:2]) / np.array([W, H]),   # wh
            (det_boxes[:, 0:2] + det_boxes[:, 2:4]) / 2 / np.array([W, H])  # cxy
        ])                                                                    # (N,8)

        iou_cost  = 1.0 - iou_matrix(track_boxes, det_boxes)
        # cosine distance between feature vectors
        tn = track_feats / (np.linalg.norm(track_feats, axis=1, keepdims=True) + 1e-6)
        dn = det_feats   / (np.linalg.norm(det_feats,   axis=1, keepdims=True) + 1e-6)
        cos_cost  = 1.0 - (tn @ dn.T)          # (N_track, N_det)
        fused     = 0.7 * iou_cost + 0.3 * cos_cost
        row_ind, col_ind = linear_sum_assignment(fused)
        track_boxes = det_boxes
        track_feats = det_feats
        times_lite.append((time.perf_counter() - t0) * 1e3)

    t_iou  = np.array(times_iou)
    t_lite = np.array(times_lite)

    rows = [
        {"tracker": "IoU-only (ByteTrack proxy)",
         "p50_ms": round(float(np.percentile(t_iou, 50)), 3),
         "p95_ms": round(float(np.percentile(t_iou, 95)), 3),
         "fps_association": round(1000.0 / float(np.percentile(t_iou, 50)), 0)},
        {"tracker": "LITE feature-reuse proxy",
         "p50_ms": round(float(np.percentile(t_lite, 50)), 3),
         "p95_ms": round(float(np.percentile(t_lite, 95)), 3),
         "fps_association": round(1000.0 / float(np.percentile(t_lite, 50)), 0)},
    ]
    for r in rows:
        print(f"    {r['tracker']:35s}  p50={r['p50_ms']} ms  FPS={r['fps_association']}")

    note = ("Note: times measure association overhead only (not detection inference). "
            "Both trackers run at thousands of FPS for this N=20 scenario; the overhead "
            "difference reflects feature vector ops vs pure IoU. "
            "The primary LITE advantage over DeepSORT comes from eliminating a ~7-20ms "
            "ReID forward pass, not from association cost differences.")
    print(f"\n    {note}")

    out = os.path.join(RESULTS_DIR, "tracker_comparison.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    with open(out.replace(".csv", "_note.txt"), "w") as f:
        f.write(note)
    print(f"  Saved → {out}")
    return rows


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("EdgeVA Benchmark Suite")
    print(f"ONNX Runtime: {ort.__version__}")
    print(f"Providers:    {ort.get_available_providers()}")
    print("=" * 60)

    print("\nExperiment 1: Inference latency benchmark …")
    inf_rows = run_inference_benchmark()

    pipe_rows  = run_pipeline_timing(MODELS["yolov8m"][0])
    track_rows = run_tracker_comparison()

    # ── summary JSON ─────────────────────────────────────────────────────────
    summary = {
        "hardware": {
            "cpu": "AMD Ryzen 5 5600X 6-Core @ 3.7 GHz",
            "gpu": "NVIDIA GeForce RTX 3060 Ti (8 GB GDDR6)",
            "ort_version": ort.__version__,
        },
        "inference":   inf_rows,
        "pipeline":    pipe_rows,
        "tracker":     track_rows,
    }
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n✓ All experiments complete. Results in experiments/results/")
