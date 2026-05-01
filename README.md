# Prompt-Calibrated SAM 3 for Open-Vocabulary Remote Sensing Semantic Segmentation

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-%3E%3D2.2-orange.svg)](https://pytorch.org/)

## Overview

**QwSAM3-pgrf** (Prompt-Guided Robust Fusion) enhances SAM3 with multimodal prompting from Qwen-VL. It combines vision‑language understanding and advanced segmentation to achieve state‑of‑the‑art open‑vocabulary semantic segmentation on remote sensing images.

## Key Features

- **Multimodal prompting** – text + optional visual prompts
- **Robust fusion** of multiple prompt expansions
- **Training‑free** – works off‑the‑shelf with SAM3 + Qwen-VL
- **High accuracy** – outperforms previous methods across 8 datasets


## Datasets

Evaluated on eight diverse remote sensing benchmarks:

- LoveDA, Potsdam, Vaihingen, iSAID  
- OEM, UDD5, VDD, UAVid

## Results

The following table reports **mIoU** (%) for open‑vocabulary semantic segmentation.  
Our method (using SAM3 backbone) achieves the highest average performance across all datasets.

| Method | Backbone | LoveDA | Potsdam | Vaihingen | iSAID | OEM | UDD5 | VDD | UAVid | Avg. |
|--------|----------|--------|---------|-----------|-------|-----|------|-----|-------|------|
| **Training‑Required Methods** |
| CAT-Seg (CVPR'24) | CLIP | 28.6 | 35.8 | 42.3 | 53.3 | – | – | 40.2 | 39.1 | 25.7 |
| RSKR-Seg (AAAI'26) | CLIP+RemoteCLIP+DINO | 33.2 | 38.4 | 42.7 | 54.3 | – | – | 42.1 | 39.7 | 25.7 |
| GSNet (AAAI'25) | CLIP | 32.5 | 37.9 | 44.1 | 53.7 | – | – | 40.9 | 37.3 | 24.2 |
| SkySense-O (CVPR'25) | CLIP | 38.3 | 54.1 | 51.6 | 43.9 | 40.8 | – | – | – | – |
| **Training‑Free Methods**  |
| GEM (CVPR'24) | CLIP | 31.6 | 39.1 | 36.4 | 17.7 | 33.9 | 41.2 | 39.5 | 33.4 | 34.1 |
| ClearCLIP (ECCV'24) | CLIP | 32.4 | 42.0 | 36.2 | 18.2 | 31.0 | 41.8 | 39.3 | 36.2 | 34.6 |
| SCLIP (ECCV'24) | CLIP | 30.4 | 39.6 | 35.9 | 16.1 | 29.3 | 38.7 | 37.9 | 31.4 | 32.4 |
| ProxyCLIP (ECCV'24) | CLIP+DINOv2 | 34.3 | 49.0 | 47.5 | 21.8 | 38.9 | 40.8 | 47.8 | 35.8 | 39.5 |
| CorrCLIP (ICCV'25) | CLIP+DINO+SAM2 | 36.9 | 51.9 | 47.0 | 25.5 | 32.9 | 46.1 | 47.3 | 38.3 | 40.7 |
| SegEarth-OV (CVPR'25) | CLIP | 36.9 | 48.5 | 40.0 | 21.7 | 40.3 | 50.6 | 45.3 | 42.5 | 40.7 |
| SegEarth-OV3* (arXiv'25) | SAM3 | 41.6 | 57.2 | 60.8 | 26.0 | 41.0 | 71.6 | 64.4 | 54.7 | 52.2 |
| **Ours** | **SAM3** | **46.1** | **58.8** | **64.5** | **29.5** | **48.9** | **73.4** | **67.5** | **60.0** | **56.1** |



## Installation

1. Clone the repository:
```bash
git clone https://github.com/YanghuiSong/QwSAM3_pgrf.git
cd QwSAM3_pgrf
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download models:
   - [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
   - SAM3 checkpoint (from official Meta repository)

4. Update paths in `config.py` to point to your local models.

## Usage

### Basic segmentation
```python
from sam3_segmentor import SegEarthOV3Segmentation

segmentor = SegEarthOV3Segmentation(
    classname_path="path/to/classnames.txt",
    device="cuda"
)
result = segmentor.segment(image_path="image.jpg", prompt="segment all buildings")
```

### Run evaluation on a dataset
```bash
python eval.py --config configs/cfg_*_pgrf_max.py
```


## Acknowledgments

We thank the SAM3 and Qwen-VL teams for their foundational models, and the open‑source community for their tools.

