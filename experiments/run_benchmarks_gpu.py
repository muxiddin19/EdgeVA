"""
EdgeVA GPU Benchmark Suite — RTX 4090 (Remote Server)
======================================================
Runs on remote Ubuntu server with dual NVIDIA RTX 4090 GPUs.

Experiments:
  1. YOLOv8 family inference: CPU vs CUDA latency & throughput
  2. Full pipeline timing breakdown (CUDA provider)
  3. Tracker association comparison: IoU-only vs LITE feature-reuse

Usage:
    python run_benchmarks_gpu.py

Output:
    results/gpu_benchmark_results.csv
    results/gpu_pipeline_timing.csv
    results/gpu_tracker_comparison.csv
    results/gpu_summary.json
"""

import os, sys, time, json, csv, platform, subprocess
import numpy as np
import onnxruntime as ort

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

HOME = os.path.expanduser("~")
MODELS = {
    "yolov8n": (os.path.join(HOME, "EdgeVA", "yolov8n.onnx"), 3_151_904),
    "yolov8s": (os.path.join(HOME, "EdgeVA", "yolov8s.onnx"), 11_166_560),
    "yolov8m": (os.path.join(HOME, "EdgeVA", "yolov8m.onnx"), 25_886_080),
}

INPUT_SHAPE = (1, 3, 640, 640)
WARMUP = 100   # more warmup for GPU
RUNS   = 500   # more runs for stability


# ── helpers ──────────────────────────────────────────────────────────────────

def make_dummy_input():
    return np.random.rand(*INPUT_SHAPE).astype(np.float32)


def build_session(model_path: str, provider: str) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 8
    providers = [provider, "CPUExecutionProvider"]
    if provider == "CUDAExecutionProvider":
        # Pin to GPU 0
        providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    return ort.InferenceSession(model_path, providers=providers, sess_options=opts)


def get_gpu_info():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
             "--format=csv,noheader,nounits"],
            text=True).strip()
        lines = [l.strip() for l in out.split("\n") if l.strip()]
        return lines
    except Exception:
        return ["unknown"]


def benchmark_model(model_name: str, model_path: str, params: int,
                    provider: str) -> dict:
    print(f"  [{provider.replace('ExecutionProvider','')}] {model_name} …", flush=True)
    if not os.path.exists(model_path):
        print(f"    SKIP: {model_path} not found")
        return None
    sess  = build_session(model_path, provider)
    inp   = make_dummy_input()
    iname = sess.get_inputs()[0].name

    # check which provider actually ran
    actual_prov = sess.get_providers()[0]

    for _ in range(WARMUP):
        sess.run(None, {iname: inp})

    latencies = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        sess.run(None, {iname: inp})
        latencies.append((time.perf_counter() - t0) * 1000)

    lats = np.array(latencies)
    row = {
        "model":         model_name,
        "requested_prov": provider.replace("ExecutionProvider", ""),
        "actual_prov":   actual_prov.replace("ExecutionProvider", ""),
        "params_M":      round(params / 1e6, 1),
        "p50_ms":        round(float(np.percentile(lats, 50)), 3),
        "p95_ms":        round(float(np.percentile(lats, 95)), 3),
        "p99_ms":        round(float(np.percentile(lats, 99)), 3),
        "mean_ms":       round(float(lats.mean()), 3),
        "std_ms":        round(float(lats.std()), 3),
        "fps":           round(1000.0 / float(np.percentile(lats, 50)), 1),
    }
    print(f"    p50={row['p50_ms']} ms  p99={row['p99_ms']} ms  "
          f"FPS={row['fps']}  (actual: {row['actual_prov']})")
    return row


# ── Experiment 1: inference benchmark ────────────────────────────────────────

def run_inference_benchmark():
    print("\nExperiment 1: Inference latency benchmark …", flush=True)
    avail = ort.get_available_providers()
    print(f"  Available providers: {avail}")

    providers = []
    if "CUDAExecutionProvider" in avail:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    rows = []
    for prov in providers:
        for name, (path, params) in MODELS.items():
            row = benchmark_model(name, path, params, prov)
            if row:
                rows.append(row)

    if not rows:
        print("  No models found — run export_models.py first")
        return []

    out = os.path.join(RESULTS_DIR, "gpu_benchmark_results.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Saved → {out}")
    return rows


# ── Experiment 2: pipeline timing breakdown ───────────────────────────────────

def run_pipeline_timing():
    print("\nExperiment 2: Pipeline timing breakdown (YOLOv8m) …", flush=True)
    yolov8m_path = MODELS["yolov8m"][0]
    if not os.path.exists(yolov8m_path):
        print("  SKIP: yolov8m.onnx not found")
        return []

    avail = ort.get_available_providers()
    provider = ("CUDAExecutionProvider" if "CUDAExecutionProvider" in avail
                else "CPUExecutionProvider")
    sess  = build_session(yolov8m_path, provider)
    iname = sess.get_inputs()[0].name
    frame_bgr = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)

    class MinimalTracker:
        def __init__(self):
            self.track_id = 0
            self.tracks   = {}
        def update(self, boxes):
            self.tracks = {}
            for b in boxes:
                self.track_id += 1
                self.tracks[self.track_id] = ((b[0]+b[2])/2, (b[1]+b[3])/2)
            return self.tracks

    tracker = MinimalTracker()
    stages = {s: [] for s in ["preprocess","inference","postprocess","tracking","analytics","total"]}

    for _ in range(WARMUP):
        inp = frame_bgr[:,:,::-1].transpose(2,0,1)[np.newaxis].astype(np.float32)/255.
        sess.run(None, {iname: inp})

    for _ in range(RUNS):
        t_start = time.perf_counter()

        t0 = time.perf_counter()
        inp = frame_bgr[:,:,::-1].transpose(2,0,1)[np.newaxis].astype(np.float32)/255.
        t1 = time.perf_counter()

        out = sess.run(None, {iname: inp})
        t2 = time.perf_counter()

        pred   = out[0][0].T
        scores = pred[:, 4:].max(1)
        boxes  = pred[scores > 0.25, :4]
        t3 = time.perf_counter()

        if len(boxes):
            xyxy = np.stack([boxes[:,0]-boxes[:,2]/2, boxes[:,1]-boxes[:,3]/2,
                             boxes[:,0]+boxes[:,2]/2, boxes[:,1]+boxes[:,3]/2], 1)
        else:
            xyxy = np.empty((0,4))
        tracker.update(xyxy)
        t4 = time.perf_counter()

        count = sum(1 for cx,cy in tracker.tracks.values() if 0<=cx<=320 and 0<=cy<=320)
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
            "stage":    stage,
            "provider": provider.replace("ExecutionProvider",""),
            "p50_ms":   round(float(np.percentile(arr,50)),3),
            "p95_ms":   round(float(np.percentile(arr,95)),3),
            "mean_ms":  round(float(arr.mean()),3),
        })
        print(f"    {stage:15s}  p50={rows[-1]['p50_ms']:8.3f} ms")

    out = os.path.join(RESULTS_DIR, "gpu_pipeline_timing.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved → {out}")
    return rows


# ── Experiment 3: tracker comparison ─────────────────────────────────────────

def run_tracker_comparison():
    print("\nExperiment 3: Tracker association timing …", flush=True)
    from scipy.optimize import linear_sum_assignment

    RNG = np.random.default_rng(42)
    N_OBJ, N_FRAMES, H, W = 20, 300, 480, 640
    starts = RNG.uniform(50, W-50, (N_OBJ,2))
    vels   = RNG.uniform(-3, 3, (N_OBJ,2))

    def gt_boxes(f):
        pos = starts + vels*f
        pos[:,0] = np.clip(pos[:,0], 20, W-20)
        pos[:,1] = np.clip(pos[:,1], 20, H-20)
        ww = RNG.uniform(30,80,N_OBJ); hh = RNG.uniform(60,160,N_OBJ)
        n  = RNG.normal(0,2,(N_OBJ,2))
        cx = pos[:,0]+n[:,0]; cy = pos[:,1]+n[:,1]
        return np.stack([cx-ww/2, cy-hh/2, cx+ww/2, cy+hh/2],1)

    def iou_matrix(a, b):
        xa1,ya1,xa2,ya2 = a[:,0:1],a[:,1:2],a[:,2:3],a[:,3:4]
        xb1,yb1,xb2,yb2 = b[:,0],b[:,1],b[:,2],b[:,3]
        ix = np.maximum(0, np.minimum(xa2,xb2)-np.maximum(xa1,xb1))
        iy = np.maximum(0, np.minimum(ya2,yb2)-np.maximum(ya1,yb1))
        inter = ix*iy
        return inter / ((xa2-xa1)*(ya2-ya1) + (xb2-xb1)*(yb2-yb1) - inter + 1e-6)

    times_iou, times_lite = [], []
    track_boxes = gt_boxes(0)
    for f in range(N_FRAMES):
        det = gt_boxes(f)
        t0 = time.perf_counter()
        cost = 1. - iou_matrix(track_boxes, det)
        r,c = linear_sum_assignment(cost)
        track_boxes = det
        times_iou.append((time.perf_counter()-t0)*1e3)

    track_boxes = gt_boxes(0)
    track_feats = np.zeros((N_OBJ,8), np.float32)
    for f in range(N_FRAMES):
        det = gt_boxes(f)
        t0 = time.perf_counter()
        det_feats = np.column_stack([det/[W,H,W,H],
                                     (det[:,2:4]-det[:,0:2])/[W,H],
                                     (det[:,0:2]+det[:,2:4])/2/[W,H]])
        iou_cost = 1. - iou_matrix(track_boxes, det)
        tn = track_feats/(np.linalg.norm(track_feats,axis=1,keepdims=True)+1e-6)
        dn = det_feats  /(np.linalg.norm(det_feats,  axis=1,keepdims=True)+1e-6)
        cos_cost = 1. - (tn @ dn.T)
        r,c = linear_sum_assignment(0.7*iou_cost + 0.3*cos_cost)
        track_boxes = det; track_feats = det_feats
        times_lite.append((time.perf_counter()-t0)*1e3)

    t_iou  = np.array(times_iou)
    t_lite = np.array(times_lite)
    rows = [
        {"tracker":"IoU-only (ByteTrack proxy)",
         "p50_ms":round(float(np.percentile(t_iou,50)),3),
         "p95_ms":round(float(np.percentile(t_iou,95)),3),
         "fps_association":round(1000./float(np.percentile(t_iou,50)),0)},
        {"tracker":"LITE feature-reuse proxy",
         "p50_ms":round(float(np.percentile(t_lite,50)),3),
         "p95_ms":round(float(np.percentile(t_lite,95)),3),
         "fps_association":round(1000./float(np.percentile(t_lite,50)),0)},
    ]
    for r in rows:
        print(f"    {r['tracker']:40s}  p50={r['p50_ms']} ms  FPS={r['fps_association']}")

    out = os.path.join(RESULTS_DIR, "gpu_tracker_comparison.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved → {out}")
    return rows


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("EdgeVA GPU Benchmark Suite")
    print(f"ONNX Runtime : {ort.__version__}")
    print(f"Providers    : {ort.get_available_providers()}")
    print(f"Platform     : {platform.node()} / {platform.system()}")
    gpu_info = get_gpu_info()
    for i, g in enumerate(gpu_info):
        print(f"GPU {i}        : {g}")
    print("=" * 60)

    inf_rows   = run_inference_benchmark()
    pipe_rows  = run_pipeline_timing()
    track_rows = run_tracker_comparison()

    summary = {
        "hardware": {
            "gpus": gpu_info,
            "ort_version": ort.__version__,
            "platform": platform.node(),
            "cuda_available": "CUDAExecutionProvider" in ort.get_available_providers(),
        },
        "inference": inf_rows,
        "pipeline":  pipe_rows,
        "tracker":   track_rows,
    }
    out = os.path.join(RESULTS_DIR, "gpu_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary JSON → {out}")
    print("\nAll experiments complete.")
