"""EdgeVA package setup."""
from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="edgeva",
    version="0.1.0",
    description="Edge-Intelligent Video Analytics — detection, tracking, and analytics for edge hardware",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Mukhiddin Toshpulatov, Wookey Lee, Jinsoo Cho, Dilafruz Iskandarova",
    author_email="muhiddin@gachon.ac.kr",
    url="https://github.com/muxiddin19/EdgeVA",
    license="MIT",
    packages=find_packages(exclude=["tests*", "scripts*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "pyyaml>=6.0",
    ],
    extras_require={
        "cpu": [
            "onnxruntime>=1.17",
            "opencv-python>=4.8",
        ],
        "cuda": [
            "onnxruntime-gpu>=1.17",
            "opencv-python>=4.8",
        ],
        "eval": [
            "pandas>=2.0",
            "matplotlib>=3.7",
            "tqdm>=4.65",
        ],
        "all": [
            "onnxruntime-gpu>=1.17",
            "opencv-python>=4.8",
            "pandas>=2.0",
            "matplotlib>=3.7",
            "tqdm>=4.65",
            "faiss-cpu>=1.7",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
            "ruff",
            "mypy",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    keywords=[
        "edge-ai", "video-analytics", "object-detection", "multi-object-tracking",
        "yolo", "feature-reuse", "lite-tracker", "jetson", "hailo", "smart-city",
        "retail-analytics", "industrial-safety", "ppe-detection",
    ],
)
