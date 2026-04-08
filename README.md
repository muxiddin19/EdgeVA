# EdgeVA — Edge-Intelligent Video Analytics

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-Survey-red)](https://arxiv.org/)

Official implementation and benchmark toolkit accompanying the survey paper:

> **"Edge-Intelligent Video Analytics: A Systematic Survey of Architectures,
> Applications, and Deployment Strategies Across Smart City Verticals"**
> *ACM Computing Surveys (under review)*

---

## Overview

**EdgeVA** provides a unified, hardware-agnostic framework for edge video analytics,
covering detection, multi-object tracking, recognition, and five application verticals:

| Component | Description | Key References |
|---|---|---|
| `edgeva.detection` | YOLO v8–v12, RT-DETR via ONNX/TensorRT | [YOLOv8], [YOLOv10], [RT-DETR] |
| `edgeva.tracking` | Feature-reuse tracker (LITE paradigm), ByteTrack, OC-SORT | [LITE], [ByteTrack], [OC-SORT] |
| `edgeva.analytics` | People counting, PPE detection, traffic, occupancy | Survey §7 |
| `edgeva.hardware` | Cross-platform benchmarking (Jetson, Hailo, Coral, CPU/GPU) | Survey §8 |
| `edgeva.utils` | HOTA, MOTA, IDF1 evaluation metrics | [HOTA] |

---

## Hardware Support

| Platform | Compute | TDP | Backend |
|---|---|---|---|
| NVIDIA Jetson AGX Orin 64GB | 275 TOPS | 60W | TensorRT / ONNX CUDA |
| NVIDIA Jetson Orin NX 16GB | 100 TOPS | 25W | TensorRT / ONNX CUDA |
| NVIDIA Jetson Orin Nano 8GB | 40 TOPS | 15W | TensorRT / ONNX CUDA |
| Hailo-10H | 40 TOPS | 5W | HailoRT |
| Hailo-8 | 26 TOPS | 2.5W | HailoRT |
| Google Coral Edge TPU | 4 TOPS | <2W | TFLite / PyCoral |
| x86 CPU / GPU | — | — | ONNX Runtime |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/muxiddin19/EdgeVA.git
cd EdgeVA

# Install (CPU baseline — no hardware-specific dependencies)
pip install -e ".[cpu]"

# Install with CUDA/TensorRT support
pip install -e ".[cuda]"

# Install with all extras (development + evaluation)
pip install -e ".[all]"
```

### Hardware-specific setup

**NVIDIA Jetson Orin (TensorRT)**
```bash
# TensorRT is pre-installed on JetPack; install Python bindings:
pip install tensorrt pycuda
pip install onnxruntime-gpu  # use the Jetson-optimised wheel
```

**Hailo-8/10H**
```bash
pip install hailo-platform   # requires HailoRT driver installation
```

**Google Coral Edge TPU**
```bash
pip install pycoral tflite-runtime
```

---

## Quick Start

### Detection

```python
from edgeva.detection import YOLODetector

det = YOLODetector("yolov8n.onnx", device="cuda", conf_thresh=0.3)
dets = det.detect(frame)          # frame: HWC uint8 numpy array
for d in dets:
    print(d)
# Detection(cls='person', score=0.91, box=[120,45,380,620])
```

### Feature-Reuse Tracking (LITE Paradigm)

```python
from edgeva.tracking import FeatureReuseTracker

tracker = FeatureReuseTracker(
    track_thresh=0.5,
    fpn_level="P4",      # P3 | P4 | P5
    use_byte=True,       # ByteTrack low-confidence second round
)

for frame in video_stream:
    dets = det.detect(frame)
    boxes  = np.array([d.bbox_xyxy for d in dets])
    scores = np.array([d.score     for d in dets])

    # Pass FPN features for appearance-based association (optional)
    fpn_feats = det.get_fpn_features(frame)   # {"P3":..., "P4":..., "P5":...}

    tracks = tracker.update(boxes, scores, fpn_features=fpn_feats)
    for t in tracks:
        print(f"Track {t.track_id}: {t.xyxy}")
```

### People Counting (Retail Entrance)

```python
from edgeva.analytics import LineCrossingCounter, DwellTimeAnalyser

# Define entrance line (image coordinates)
counter = LineCrossingCounter(line_p1=(640, 0), line_p2=(640, 720), name="entrance")

for frame in video_stream:
    tracks = tracker.update(...)
    for t in tracks:
        cx = (t.xyxy[0] + t.xyxy[2]) / 2
        cy = (t.xyxy[1] + t.xyxy[3]) / 2
        counter.update(t.track_id, (cx, cy))

print(f"In: {counter.count_in}  Out: {counter.count_out}  Net: {counter.net_count()}")
```

### Hardware Benchmarking

```python
from edgeva.hardware import HardwareBenchmark, BenchmarkSuite
import onnxruntime as ort

session = ort.InferenceSession("yolov8n.onnx")
infer_fn = lambda x: session.run(None, {"images": x})

bench = HardwareBenchmark(
    model_name="YOLOv8n",
    infer_fn=infer_fn,
    input_shape=(1, 3, 640, 640),
    precision="fp32",
    backend="onnx_cpu",
    n_warmup=20,
    n_runs=200,
)
result = bench.run()
print(result)
# [x86_cpu/onnx_cpu/fp32] YOLOv8n (640, 640) mean=12.3ms p95=13.1ms FPS=81.3
```

---

## MOT Evaluation

```bash
# Evaluate on MOT17 (requires dataset at data/MOT17/)
python scripts/eval_mot17.py \
    --model yolov8n.onnx \
    --tracker feature_reuse \
    --split val \
    --fpn-level P4
```

---

## Project Structure

```
EdgeVA/
├── edgeva/
│   ├── detection/          # YOLO / RT-DETR detectors (ONNX, TensorRT)
│   ├── tracking/           # Feature-reuse (LITE), ByteTrack, Kalman, matching
│   ├── analytics/          # People counting, PPE, traffic, occupancy
│   ├── recognition/        # Face recognition (ArcFace), ALPR pipeline
│   ├── hardware/           # Cross-platform benchmarking suite
│   └── utils/              # HOTA / MOTA / IDF1 metrics, visualisation
├── scripts/                # Demo and evaluation scripts
├── configs/                # YAML configs per application vertical
├── tests/                  # Unit tests
└── .github/workflows/      # CI (GitHub Actions)
```

---

## Citing This Work

If you use EdgeVA in your research, please cite:

```bibtex
@article{toshpulatov2026edgeva,
  author    = {Toshpulatov, Mukhiddin and Lee, Wookey and
               Cho, Jinsoo and Iskandarova, Dilafruz},
  title     = {Edge-Intelligent Video Analytics: A Systematic Survey
               of Architectures, Applications, and Deployment Strategies
               Across Smart City Verticals},
  journal   = {ACM Computing Surveys},
  year      = {2026},
  note      = {Under review}
}

@misc{alikhanov2024lite,
  author    = {Alikhanov, Jumabek and others},
  title     = {{LITE}: A Paradigm Shift in Multi-Object Tracking
               with Efficient {ReID} Feature Extraction},
  year      = {2024},
  eprint    = {arXiv:2024.xxxxx}
}
```

---

## Key References

| Method | Venue | Description |
|---|---|---|
| YOLOv8 | Ultralytics 2023 | Anchor-free detector baseline |
| YOLOv10 | NeurIPS 2024 | NMS-free end-to-end detection |
| YOLOv12 | arXiv 2025 | Area attention detector |
| RT-DETR | CVPR 2024 | Transformer detector, edge-viable |
| ByteTrack | ECCV 2022 | Motion tracker, low-conf association |
| OC-SORT | CVPR 2023 | Observation-centric Kalman tracker |
| LITE | arXiv 2024 | Feature-reuse tracker (this repo's core) |
| HOTA | IJCV 2021 | Primary tracking evaluation metric |

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Acknowledgements

This work was supported by the IITP–ITRC grant (XVoice, RS-2022-II220641)
and the National Research Foundation of Korea (NRF, RS-2025-24534935).
