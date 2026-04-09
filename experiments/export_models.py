"""
Export YOLOv8 models to ONNX format for benchmarking.
Run this once on the remote server before run_benchmarks_gpu.py
"""
import os
HOME = os.path.expanduser("~")
OUT_DIR = os.path.join(HOME, "EdgeVA")
os.makedirs(OUT_DIR, exist_ok=True)

from ultralytics import YOLO

for variant in ["yolov8n", "yolov8s", "yolov8m"]:
    out_path = os.path.join(OUT_DIR, f"{variant}.onnx")
    if os.path.exists(out_path):
        print(f"  {variant}.onnx already exists — skipping")
        continue
    print(f"  Exporting {variant} …", flush=True)
    model = YOLO(f"{variant}.pt")
    model.export(format="onnx", imgsz=640, dynamic=False, simplify=True)
    # ultralytics saves to {variant}.onnx in cwd; move to target dir
    src = f"{variant}.onnx"
    if os.path.exists(src):
        os.rename(src, out_path)
    print(f"  Saved → {out_path}")

print("Done.")
