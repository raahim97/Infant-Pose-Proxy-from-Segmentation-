# Infant Pose Proxy from Segmentation Masks

A web application and training pipeline for infant body part segmentation from images and videos.

![Demo Screenshot](screenshots/demo.png)
*Screenshot placeholder - add actual screenshots after deployment*

## ⚠️ Important Disclaimer

**This project is for RESEARCH and EDUCATIONAL purposes only.**

- NOT intended for medical diagnosis or clinical use
- NOT a substitute for professional medical advice
- Results should NOT be used to make health-related decisions
- Always consult qualified healthcare professionals for medical concerns

---

## Overview

This project provides:

1. **Training Pipeline**: Fine-tune DeepLabV3 (MobileNetV3 backbone) on the Youtube-Infant-Body-Parsing dataset
2. **FastAPI Web App**: Upload images/videos and get:
   - Colorized segmentation mask overlay
   - Bounding boxes per body part
   - Centroid keypoints per body part
   - Video: centroid tracks over time

### Body Parts Detected

| Class ID | Body Part | Color |
|----------|-----------|-------|
| 0 | Background | Black |
| 1 | Head | Green |
| 2 | Hand/Arm | Blue |
| 3 | Body/Torso | Orange |
| 4 | Foot/Leg | Red |

---

## Quick Start

### Prerequisites

- Python 3.10+
- The dataset files:
  - `all_data/` directory with JPG frames
  - `train_label.json`, `test_label.json`

### Installation

```bash
# Clone/navigate to repository
cd infant-pose-proxy

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Dataset Preparation

### Step 1: Prepare Dataset

Convert JSON labels to training-ready format:

```bash
python scripts/prepare_dataset.py
```

This will:
- Read `train_label.json` and `test_label.json`
- Find corresponding images in `all_data/`
- Decode base64 labels to class-ID masks
- Create organized folders:
  - `data/images/` - Input images
  - `data/masks/` - Single-channel class masks
  - `data/splits/` - train.txt and val.txt

### Step 2: Verify Dataset

```bash
python scripts/verify_dataset.py
```

Checks:
- Image/mask counts and 1:1 correspondence
- Mask value ranges (classes 0-4)
- Split file validity

---

## Training

### Train the Model

```bash
python train/train.py \
    --images_dir data/images \
    --masks_dir data/masks \
    --train_split data/splits/train.txt \
    --val_split data/splits/val.txt \
    --num_classes 5 \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-4 \
    --image_size 256 \
    --out models/segmentation.pt
```

#### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--images_dir` | `data/images` | Input images directory |
| `--masks_dir` | `data/masks` | Mask images directory |
| `--train_split` | `data/splits/train.txt` | Training split file |
| `--val_split` | `data/splits/val.txt` | Validation split file |
| `--num_classes` | `5` | Number of segmentation classes |
| `--epochs` | `50` | Training epochs |
| `--batch_size` | `8` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--image_size` | `256` | Input image size |
| `--out` | `models/segmentation.pt` | Output checkpoint path |
| `--device` | `auto` | Device (auto, cpu, cuda, mps) |

### Export to ONNX

```bash
python train/export.py \
    --checkpoint models/segmentation.pt \
    --output models/segmentation.onnx \
    --opset 17
```

---

## Running the Web App

### Local Development

```bash
# Set environment variables (optional - has defaults)
export MODEL_PATH=models/segmentation.onnx
export NUM_CLASSES=5
export IMAGE_SIZE=256

# Run with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 in your browser.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `models/segmentation.onnx` | Path to ONNX model |
| `NUM_CLASSES` | `5` | Number of segmentation classes |
| `IMAGE_SIZE` | `256` | Model input size |
| `VIDEO_MAX_FRAMES` | `30` | Max frames to sample from video |
| `VIDEO_FPS` | `2` | Frame sampling rate for videos |

---

## API Endpoints

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "model_loaded": true,
  "message": "Model loaded successfully",
  "config": {
    "num_classes": 5,
    "image_size": 256,
    "video_max_frames": 30,
    "video_fps": 2.0,
    "model_path": "models/segmentation.onnx"
  }
}
```

### Analyze Image

```bash
curl -X POST http://localhost:8000/analyze_image \
    -F "file=@test_image.jpg" \
    -o response.json
```

Response:
```json
{
  "num_classes": 5,
  "image_size": 256,
  "runtime_ms": 45.23,
  "mask_png_base64": "...",
  "overlay_png_base64": "...",
  "detections": [
    {
      "class_id": 1,
      "class_name": "head",
      "box": {"xmin": 100, "ymin": 50, "xmax": 180, "ymax": 130},
      "keypoint": {"cx": 140.5, "cy": 90.2},
      "area_px": 4820
    }
  ]
}
```

### Analyze Video

```bash
curl -X POST http://localhost:8000/analyze_video \
    -F "file=@test_video.mp4" \
    -o response.json
```

Response:
```json
{
  "num_classes": 5,
  "image_size": 256,
  "sampled_frames": 15,
  "frame_timestamps_s": [0.0, 0.5, 1.0, ...],
  "runtime_ms_total": 1234.56,
  "first_overlay_png_base64": "...",
  "tracks": {
    "1": [[0.0, 140.5, 90.2], [0.5, 142.1, 89.8], ...],
    "2": [[0.0, 80.3, 150.1], ...],
    "3": [[0.0, 128.0, 180.5], ...],
    "4": [[0.0, 130.2, 220.8], ...]
  }
}
```

---

## Docker Deployment

### Build and Run Locally

```bash
# Build image
docker build -t infant-pose-proxy .

# Run container
docker run -p 7860:7860 infant-pose-proxy
```

### Deploy to Hugging Face Spaces

1. Create a new Space on [Hugging Face](https://huggingface.co/spaces)
2. Select "Docker" as the SDK
3. Upload these files:
   - `app/` directory
   - `static/` directory
   - `models/segmentation.onnx`
   - `requirements.txt`
   - `Dockerfile`

The Space will build and deploy automatically.

---

## Project Structure

```
infant-pose-proxy/
├── app/
│   ├── main.py          # FastAPI application
│   ├── inference.py     # ONNX model inference
│   ├── features.py      # Bounding box/centroid extraction
│   ├── viz.py           # Mask colorization and overlay
│   ├── schemas.py       # Pydantic models
│   └── utils.py         # Utility functions
├── train/
│   ├── dataset.py       # PyTorch dataset
│   ├── train.py         # Training script
│   └── export.py        # ONNX export
├── scripts/
│   ├── prepare_dataset.py   # Dataset preparation
│   └── verify_dataset.py    # Dataset verification
├── static/
│   └── index.html       # Web UI
├── models/
│   └── segmentation.onnx    # Trained model (after export)
├── data/                # Generated by prepare_dataset.py
│   ├── images/
│   ├── masks/
│   └── splits/
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

---

## Dataset Attribution

This project uses the [Youtube-Infant-Body-Parsing Dataset](https://github.com/nihaomiao/MIA23_InfantSeg):

```bibtex
@article{ni2023semi,
  title={Semi-supervised body parsing and pose estimation for enhancing infant general movement assessment},
  author={Ni, Haomiao and Xue, Yuan and Ma, Liya and Zhang, Qian and Li, Xiaoye and Huang, Sharon X},
  journal={Medical Image Analysis},
  volume={83},
  pages={102654},
  year={2023},
  publisher={Elsevier}
}

@article{chambers2020computer,
  title={Computer vision to automatically assess infant neuromotor risk},
  author={Chambers, Claire and others},
  journal={IEEE TNSRE},
  year={2020}
}
```

---

## Limitations

- Model trained on limited dataset (~2000 labeled frames)
- Performance varies with lighting, camera angle, and occlusion
- Video analysis may be slow for long videos (samples at 2 FPS)
- Not validated for clinical use

---

## License

This code is provided for research and educational purposes.
The dataset has its own license - see the original repository.

---

## Support

For issues with:
- **This code**: Open a GitHub issue
- **Dataset**: Contact the original authors at homerhm.ni@gmail.com
