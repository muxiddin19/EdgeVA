"""
YOLO detector with ONNX Runtime and TensorRT backends.

Supports YOLOv8, YOLOv9, YOLOv10, YOLOv11, YOLOv12, and RT-DETR
exported to ONNX format. TensorRT engine files (.engine/.trt) are
supported on NVIDIA Jetson and desktop GPUs via the optional
tensorrt binding.

References
----------
[1] Jocher et al., "Ultralytics YOLOv8", 2023.
[2] Wang et al., "YOLOv10: Real-Time End-to-End Object Detection",
    NeurIPS 2024.
[3] Tian et al., "YOLOv12: Attention-Centric Real-Time Object
    Detectors", arXiv 2025.
[4] Zhao et al., "DETRs Beat YOLOs on Real-Time Object Detection
    (RT-DETR)", CVPR 2024.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Detection result dataclass
# ---------------------------------------------------------------------------

class Detection:
    """Single-object detection result."""
    __slots__ = ("bbox_xyxy", "score", "class_id", "class_name", "features")

    def __init__(
        self,
        bbox_xyxy: np.ndarray,
        score: float,
        class_id: int,
        class_name: str = "",
        features: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.bbox_xyxy  = np.asarray(bbox_xyxy, dtype=np.float32)
        self.score      = float(score)
        self.class_id   = int(class_id)
        self.class_name = class_name
        self.features   = features or {}   # {"P3": ..., "P4": ..., "P5": ...}

    def __repr__(self) -> str:
        x1, y1, x2, y2 = self.bbox_xyxy
        return (f"Detection(cls={self.class_name!r}, score={self.score:.2f}, "
                f"box=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}])")


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def letterbox(
    image: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    stride: int = 32,
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize image with letterboxing to preserve aspect ratio.

    Returns
    -------
    padded_image : ndarray, shape (new_shape[0], new_shape[1], 3)
    scale        : float — ratio of new size to original size
    pad_wh       : (pad_w, pad_h) in pixels
    """
    h, w = image.shape[:2]
    new_h, new_w = new_shape
    scale = min(new_w / w, new_h / h)
    scaled_w = int(round(w * scale))
    scaled_h = int(round(h * scale))

    try:
        import cv2
        resized = cv2.resize(image, (scaled_w, scaled_h),
                             interpolation=cv2.INTER_LINEAR)
    except ImportError:
        # Fallback: nearest-neighbour via numpy (slower)
        y_idx = (np.arange(scaled_h) * (h / scaled_h)).astype(int)
        x_idx = (np.arange(scaled_w) * (w / scaled_w)).astype(int)
        resized = image[np.ix_(y_idx, x_idx)]

    pad_w = new_w - scaled_w
    pad_h = new_h - scaled_h
    pad_w_half = pad_w // 2
    pad_h_half = pad_h // 2

    padded = np.full((new_h, new_w, 3), color, dtype=np.uint8)
    padded[pad_h_half: pad_h_half + scaled_h,
           pad_w_half: pad_w_half + scaled_w] = resized

    return padded, scale, (pad_w_half, pad_h_half)


def preprocess_image(
    image: np.ndarray,
    input_size: Tuple[int, int] = (640, 640),
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Convert HWC uint8 BGR/RGB image to NCHW float32 blob for ONNX inference.

    Returns
    -------
    blob     : ndarray, shape (1, 3, H, W), float32, range [0, 1]
    scale    : letterbox scale factor
    pad_wh   : (pad_w, pad_h)
    """
    padded, scale, pad_wh = letterbox(image, new_shape=input_size)
    blob = padded.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[None]  # HWC → NCHW
    return blob, scale, pad_wh


def postprocess_yolo(
    outputs: np.ndarray,
    scale: float,
    pad_wh: Tuple[int, int],
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    input_size: Tuple[int, int] = (640, 640),
    num_classes: int = 80,
    class_names: Optional[List[str]] = None,
    is_nms_free: bool = False,
) -> List[Detection]:
    """
    Postprocess raw YOLO output tensor to Detection objects.

    Handles both NMS-based (YOLOv8/v9/v11/v12) and NMS-free
    (YOLOv10) output formats.

    Parameters
    ----------
    outputs      : ndarray, shape (1, 4+num_classes, num_anchors) for standard YOLO
                   or (1, num_anchors, 6) for NMS-free (YOLOv10)
    scale        : letterbox scale from preprocess_image
    pad_wh       : (pad_w, pad_h) from preprocess_image
    conf_thresh  : minimum detection confidence
    iou_thresh   : NMS IoU threshold
    is_nms_free  : set True for YOLOv10 which performs end-to-end NMS
    class_names  : optional list of class name strings

    Returns
    -------
    detections : list of Detection objects, sorted by score descending
    """
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    pad_w, pad_h = pad_wh

    # --- NMS-free branch (YOLOv10 format: (1, N, 6)) ---
    if is_nms_free:
        preds = outputs[0]                     # (N, 6)
        mask = preds[:, 4] >= conf_thresh
        preds = preds[mask]
        dets = []
        for pred in preds:
            x1, y1, x2, y2, score, cls_id = pred
            # Undo letterbox
            x1 = (x1 - pad_w) / scale
            y1 = (y1 - pad_h) / scale
            x2 = (x2 - pad_w) / scale
            y2 = (y2 - pad_h) / scale
            cls_id = int(cls_id)
            dets.append(Detection(
                bbox_xyxy=np.array([x1, y1, x2, y2]),
                score=float(score),
                class_id=cls_id,
                class_name=class_names[cls_id] if cls_id < len(class_names) else str(cls_id),
            ))
        return sorted(dets, key=lambda d: -d.score)

    # --- Standard YOLO branch (output: (1, 4+C, A)) ---
    preds = outputs[0].T                       # (A, 4+C)
    boxes_cxcywh = preds[:, :4]
    class_scores = preds[:, 4:]

    class_ids  = class_scores.argmax(axis=1)
    confidences = class_scores.max(axis=1)
    mask = confidences >= conf_thresh

    boxes_cxcywh = boxes_cxcywh[mask]
    confidences  = confidences[mask]
    class_ids    = class_ids[mask]

    if len(boxes_cxcywh) == 0:
        return []

    # cxcywh → xyxy
    cx, cy, w, h = boxes_cxcywh.T
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # NMS
    keep = _nms(boxes_xyxy, confidences, iou_thresh)
    boxes_xyxy  = boxes_xyxy[keep]
    confidences = confidences[keep]
    class_ids   = class_ids[keep]

    dets = []
    for i in range(len(boxes_xyxy)):
        # Undo letterbox
        bx1 = (boxes_xyxy[i, 0] - pad_w) / scale
        by1 = (boxes_xyxy[i, 1] - pad_h) / scale
        bx2 = (boxes_xyxy[i, 2] - pad_w) / scale
        by2 = (boxes_xyxy[i, 3] - pad_h) / scale
        cls_id = int(class_ids[i])
        dets.append(Detection(
            bbox_xyxy=np.array([bx1, by1, bx2, by2]),
            score=float(confidences[i]),
            class_id=cls_id,
            class_name=class_names[cls_id] if cls_id < len(class_names) else str(cls_id),
        ))
    return sorted(dets, key=lambda d: -d.score)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    """Greedy NMS — returns indices of kept boxes sorted by score."""
    order = scores.argsort()[::-1]
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_j = ((boxes[order[1:], 2] - boxes[order[1:], 0]) *
                  (boxes[order[1:], 3] - boxes[order[1:], 1]))
        iou = inter / np.maximum(area_i + area_j - inter, 1e-6)
        order = order[1:][iou <= iou_thresh]
    return np.array(keep, dtype=int)


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------

class YOLODetector:
    """
    Unified YOLO detector supporting ONNX Runtime and TensorRT backends.

    Parameters
    ----------
    model_path    : path to .onnx or .engine file
    input_size    : (H, W) inference resolution (default 640×640)
    conf_thresh   : confidence threshold (default 0.25)
    iou_thresh    : NMS IoU threshold (default 0.45)
    class_names   : list of class name strings (COCO-80 if None)
    device        : "cpu" | "cuda" | "tensorrt"
    is_nms_free   : True for YOLOv10 which embeds NMS in the graph
    fp16          : enable FP16 inference (TensorRT / CUDA EP only)

    Examples
    --------
    >>> det = YOLODetector("yolov8n.onnx", device="cuda")
    >>> dets = det.detect(frame)
    >>> for d in dets:
    ...     print(d)
    """

    COCO_CLASSES = [
        "person","bicycle","car","motorcycle","airplane","bus","train","truck",
        "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
        "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
        "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
        "skis","snowboard","sports ball","kite","baseball bat","baseball glove",
        "skateboard","surfboard","tennis racket","bottle","wine glass","cup",
        "fork","knife","spoon","bowl","banana","apple","sandwich","orange",
        "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
        "potted plant","bed","dining table","toilet","tv","laptop","mouse",
        "remote","keyboard","cell phone","microwave","oven","toaster","sink",
        "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
        "toothbrush",
    ]

    def __init__(
        self,
        model_path: str | Path,
        input_size: Tuple[int, int] = (640, 640),
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        class_names: Optional[List[str]] = None,
        device: str = "cpu",
        is_nms_free: bool = False,
        fp16: bool = False,
    ):
        self.model_path  = Path(model_path)
        self.input_size  = input_size
        self.conf_thresh = conf_thresh
        self.iou_thresh  = iou_thresh
        self.class_names = class_names or self.COCO_CLASSES
        self.device      = device
        self.is_nms_free = is_nms_free
        self.fp16        = fp16

        self._session = None
        self._trt_engine = None
        self._load_model()

    # ------------------------------------------------------------------
    def _load_model(self):
        suffix = self.model_path.suffix.lower()
        if suffix == ".onnx":
            self._load_onnx()
        elif suffix in (".engine", ".trt"):
            self._load_tensorrt()
        else:
            raise ValueError(f"Unsupported model format: {suffix}. "
                             f"Use .onnx or .engine/.trt")

    def _load_onnx(self):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime is required: pip install onnxruntime-gpu")

        providers = []
        if self.device == "cuda":
            providers = [
                ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),
                "CPUExecutionProvider",
            ]
        elif self.device == "tensorrt":
            providers = [
                ("TensorrtExecutionProvider", {
                    "trt_fp16_enable": self.fp16,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": str(self.model_path.parent),
                }),
                "CPUExecutionProvider",
            ]
        else:
            providers = ["CPUExecutionProvider"]

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(self.model_path), sess_options=opts, providers=providers
        )
        self._input_name = self._session.get_inputs()[0].name

    def _load_tensorrt(self):
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401
        except ImportError:
            raise ImportError(
                "TensorRT Python bindings required. "
                "Install via: pip install tensorrt pycuda"
            )
        logger = trt.Logger(trt.Logger.WARNING)
        with open(self.model_path, "rb") as f, trt.Runtime(logger) as runtime:
            self._trt_engine = runtime.deserialize_cuda_engine(f.read())
        self._trt_context = self._trt_engine.create_execution_context()

    # ------------------------------------------------------------------
    def detect(
        self,
        image: np.ndarray,
        target_classes: Optional[List[int]] = None,
    ) -> List[Detection]:
        """
        Run detection on a single BGR/RGB image.

        Parameters
        ----------
        image          : ndarray, shape (H, W, 3), uint8
        target_classes : optional list of class IDs to keep (e.g. [0] for person)

        Returns
        -------
        detections : list of Detection, sorted by confidence descending
        """
        blob, scale, pad_wh = preprocess_image(image, self.input_size)

        if self._session is not None:
            outputs = self._session.run(None, {self._input_name: blob})
            raw = outputs[0]
        else:
            raw = self._infer_tensorrt(blob)

        dets = postprocess_yolo(
            raw, scale, pad_wh,
            conf_thresh=self.conf_thresh,
            iou_thresh=self.iou_thresh,
            input_size=self.input_size,
            num_classes=len(self.class_names),
            class_names=self.class_names,
            is_nms_free=self.is_nms_free,
        )

        if target_classes is not None:
            dets = [d for d in dets if d.class_id in target_classes]
        return dets

    def _infer_tensorrt(self, blob: np.ndarray) -> np.ndarray:
        import pycuda.driver as cuda
        ctx = self._trt_context
        engine = self._trt_engine

        # Allocate I/O buffers
        bindings = []
        host_outputs = []
        for i in range(engine.num_bindings):
            shape = engine.get_binding_shape(i)
            dtype = np.float16 if self.fp16 else np.float32
            buf = np.empty(shape, dtype=dtype)
            d_buf = cuda.mem_alloc(buf.nbytes)
            bindings.append(int(d_buf))
            if engine.binding_is_input(i):
                cuda.memcpy_htod(d_buf, blob.astype(dtype))
            else:
                host_outputs.append((buf, d_buf))

        ctx.execute_v2(bindings)
        for buf, d_buf in host_outputs:
            cuda.memcpy_dtoh(buf, d_buf)

        return host_outputs[0][0].astype(np.float32)

    # ------------------------------------------------------------------
    def benchmark(
        self,
        image: np.ndarray,
        n_warmup: int = 10,
        n_runs: int = 100,
    ) -> Dict[str, float]:
        """
        Measure inference latency and throughput.

        Returns dict with keys: mean_ms, std_ms, min_ms, max_ms, fps.
        """
        blob, _, _ = preprocess_image(image, self.input_size)

        # Warmup
        for _ in range(n_warmup):
            if self._session:
                self._session.run(None, {self._input_name: blob})

        latencies = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            if self._session:
                self._session.run(None, {self._input_name: blob})
            else:
                self._infer_tensorrt(blob)
            latencies.append((time.perf_counter() - t0) * 1000)

        arr = np.array(latencies)
        return {
            "mean_ms": float(arr.mean()),
            "std_ms":  float(arr.std()),
            "min_ms":  float(arr.min()),
            "max_ms":  float(arr.max()),
            "fps":     float(1000.0 / arr.mean()),
        }
