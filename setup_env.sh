#!/bin/bash
# Create isolated EdgeVA conda environment on remote server
# This avoids polluting the shared base environment

set -e
source ~/miniconda3/etc/profile.d/conda.sh

echo "==> Creating edgeva conda environment (Python 3.10) ..."
conda create -n edgeva python=3.10 -y

echo "==> Activating edgeva ..."
conda activate edgeva

echo "==> Installing packages ..."
pip install -q onnxruntime-gpu ultralytics scipy numpy opencv-python-headless onnx onnxslim

echo "==> Verifying ..."
python -c "
import onnxruntime as ort
import ultralytics
import scipy, numpy, cv2
print('onnxruntime:', ort.__version__)
print('providers:  ', ort.get_available_providers())
print('ultralytics:', ultralytics.__version__)
print('scipy:      ', scipy.__version__)
print('numpy:      ', numpy.__version__)
"

echo ""
echo "==> edgeva environment is ready."
echo "    Activate with: conda activate edgeva"
echo "    Run benchmarks: cd ~/EdgeVA && python run_benchmarks_gpu.py"
