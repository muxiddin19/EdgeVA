"""
Cross-platform hardware benchmarking for edge AI inference.

Measures inference latency, throughput, and power draw across
NVIDIA Jetson Orin series, Hailo-8/10H, Google Coral, and
standard CPU/GPU platforms.

References
----------
[1] NVIDIA, "Jetson Orin Series Module Datasheet", 2024.
[2] Hailo, "Hailo-10H AI Processor Datasheet", 2024.
[3] Google, "Coral Edge TPU", 2019.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

def detect_platform() -> str:
    """
    Detect the current edge hardware platform.

    Returns one of:
        "jetson_agx_orin", "jetson_orin_nx", "jetson_orin_nano",
        "hailo8", "hailo10h", "coral", "x86_gpu", "x86_cpu", "unknown"
    """
    system = platform.system()
    machine = platform.machine()

    # Check for Jetson via device-tree compatible string
    dt_path = Path("/proc/device-tree/compatible")
    if dt_path.exists():
        compat = dt_path.read_text(errors="ignore").lower()
        if "agx-orin" in compat:
            return "jetson_agx_orin"
        elif "orin-nx" in compat:
            return "jetson_orin_nx"
        elif "orin-nano" in compat:
            return "jetson_orin_nano"
        elif "xavier" in compat:
            return "jetson_xavier"

    # Check for Hailo via PCIe device enumeration
    try:
        lspci = subprocess.check_output(["lspci"], stderr=subprocess.DEVNULL,
                                        text=True)
        if "Hailo" in lspci:
            if "Hailo-10" in lspci:
                return "hailo10h"
            return "hailo8"
    except (FileNotFoundError, subprocess.SubprocessError):
        pass

    # Check for Coral via USB or PCIe
    try:
        lsusb = subprocess.check_output(["lsusb"], stderr=subprocess.DEVNULL,
                                        text=True)
        if "Global Unichip Corp" in lsusb or "18d1:9302" in lsusb:
            return "coral"
    except (FileNotFoundError, subprocess.SubprocessError):
        pass

    # x86 GPU / CPU
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers or "TensorrtExecutionProvider" in providers:
            return "x86_gpu"
    except ImportError:
        pass

    return "x86_cpu"


# ---------------------------------------------------------------------------
# Jetson power measurement
# ---------------------------------------------------------------------------

def read_jetson_power_mw() -> Optional[float]:
    """Read instantaneous power draw in mW from Jetson INA3221 sensors."""
    # Typical paths on Jetson Orin
    power_paths = [
        "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon*/in1_input",
        "/sys/bus/i2c/devices/1-0040/iio:device0/in_power0_input",
    ]
    import glob
    for pattern in power_paths:
        matches = glob.glob(pattern)
        if matches:
            try:
                val = int(Path(matches[0]).read_text().strip())
                return float(val)   # already in mW for INA3221
            except (ValueError, PermissionError):
                pass
    return None


# ---------------------------------------------------------------------------
# Benchmark result
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Container for a single model benchmark run."""
    model_name: str
    platform: str
    backend: str                   # "onnx_cpu" | "onnx_cuda" | "tensorrt" | "hailo" | "coral"
    input_size: tuple
    precision: str                 # "fp32" | "fp16" | "int8"
    batch_size: int = 1

    # Latency (ms)
    mean_latency_ms: float = 0.0
    std_latency_ms: float  = 0.0
    p50_latency_ms: float  = 0.0
    p95_latency_ms: float  = 0.0
    p99_latency_ms: float  = 0.0
    min_latency_ms: float  = 0.0
    max_latency_ms: float  = 0.0

    # Throughput
    throughput_fps: float = 0.0

    # Power (optional)
    mean_power_mw: Optional[float] = None
    efficiency_fps_per_w: Optional[float] = None

    # Metadata
    n_warmup: int = 0
    n_runs: int   = 0
    notes: str    = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        power_str = ""
        if self.mean_power_mw:
            power_str = f"  Power: {self.mean_power_mw/1000:.1f}W  Eff: {self.efficiency_fps_per_w:.1f} FPS/W"
        return (
            f"[{self.platform}/{self.backend}/{self.precision}] "
            f"{self.model_name} {self.input_size}  "
            f"mean={self.mean_latency_ms:.1f}ms  "
            f"p95={self.p95_latency_ms:.1f}ms  "
            f"FPS={self.throughput_fps:.1f}"
            f"{power_str}"
        )


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

class HardwareBenchmark:
    """
    Hardware-agnostic latency and throughput benchmarking tool.

    Parameters
    ----------
    model_name  : display name for this model variant
    infer_fn    : callable(input_array) → any; the inference function to time
    input_shape : shape of the input array (e.g. (1, 3, 640, 640))
    precision   : "fp32" | "fp16" | "int8"
    backend     : backend identifier string
    n_warmup    : number of warmup iterations
    n_runs      : number of timed iterations
    measure_power : attempt to read Jetson power sensors during inference

    Examples
    --------
    >>> bench = HardwareBenchmark("YOLOv8n", session.run, (1,3,640,640))
    >>> result = bench.run()
    >>> print(result)
    """

    def __init__(
        self,
        model_name: str,
        infer_fn: Callable,
        input_shape: tuple = (1, 3, 640, 640),
        precision: str = "fp32",
        backend: str = "onnx_cpu",
        n_warmup: int = 20,
        n_runs: int = 200,
        measure_power: bool = True,
    ):
        self.model_name  = model_name
        self.infer_fn    = infer_fn
        self.input_shape = input_shape
        self.precision   = precision
        self.backend     = backend
        self.n_warmup    = n_warmup
        self.n_runs      = n_runs
        self.measure_power = measure_power
        self._platform   = detect_platform()

    def run(self, dummy_input: Optional[np.ndarray] = None) -> BenchmarkResult:
        """
        Execute the benchmark and return a BenchmarkResult.

        Parameters
        ----------
        dummy_input : pre-allocated input array; generated if None
        """
        dtype = np.float16 if self.precision == "fp16" else np.float32
        inp = (dummy_input if dummy_input is not None
               else np.random.randn(*self.input_shape).astype(dtype))

        # Warmup
        for _ in range(self.n_warmup):
            self.infer_fn(inp)

        # Timed runs
        latencies_ms = []
        power_readings = []

        for _ in range(self.n_runs):
            if self.measure_power:
                pw = read_jetson_power_mw()
                if pw is not None:
                    power_readings.append(pw)

            t0 = time.perf_counter()
            self.infer_fn(inp)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        arr = np.array(latencies_ms)
        mean_lat = float(arr.mean())
        fps = 1000.0 / mean_lat

        mean_power = float(np.mean(power_readings)) if power_readings else None
        efficiency = (fps / (mean_power / 1000.0)) if mean_power else None

        return BenchmarkResult(
            model_name=self.model_name,
            platform=self._platform,
            backend=self.backend,
            input_size=self.input_shape[2:],
            precision=self.precision,
            batch_size=self.input_shape[0],
            mean_latency_ms=mean_lat,
            std_latency_ms=float(arr.std()),
            p50_latency_ms=float(np.percentile(arr, 50)),
            p95_latency_ms=float(np.percentile(arr, 95)),
            p99_latency_ms=float(np.percentile(arr, 99)),
            min_latency_ms=float(arr.min()),
            max_latency_ms=float(arr.max()),
            throughput_fps=fps,
            mean_power_mw=mean_power,
            efficiency_fps_per_w=efficiency,
            n_warmup=self.n_warmup,
            n_runs=self.n_runs,
        )


# ---------------------------------------------------------------------------
# Multi-model benchmark suite
# ---------------------------------------------------------------------------

class BenchmarkSuite:
    """
    Run and save a battery of benchmarks across multiple models.

    Produces results compatible with the hardware comparison tables
    in the companion paper.
    """

    def __init__(self, output_dir: str | Path = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._results: List[BenchmarkResult] = []

    def add(self, result: BenchmarkResult):
        self._results.append(result)
        return self

    def summary(self) -> str:
        lines = [
            f"{'Model':<20} {'Platform':<18} {'Backend':<14} "
            f"{'Prec':<6} {'mean ms':>8} {'p95 ms':>8} {'FPS':>8} {'FPS/W':>8}",
            "-" * 90,
        ]
        for r in self._results:
            eff = f"{r.efficiency_fps_per_w:.1f}" if r.efficiency_fps_per_w else "N/A"
            lines.append(
                f"{r.model_name:<20} {r.platform:<18} {r.backend:<14} "
                f"{r.precision:<6} {r.mean_latency_ms:>8.1f} "
                f"{r.p95_latency_ms:>8.1f} {r.throughput_fps:>8.1f} {eff:>8}"
            )
        return "\n".join(lines)

    def save_json(self, filename: str = "results.json"):
        path = self.output_dir / filename
        with open(path, "w") as f:
            json.dump([r.to_dict() for r in self._results], f, indent=2)
        return path

    def save_csv(self, filename: str = "results.csv"):
        import csv
        path = self.output_dir / filename
        if not self._results:
            return path
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=asdict(self._results[0]).keys())
            writer.writeheader()
            writer.writerows(asdict(r) for r in self._results)
        return path
