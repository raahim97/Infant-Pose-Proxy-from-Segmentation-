"""
FastAPI application for infant body part segmentation.

Provides endpoints for:
- Image segmentation with overlay and detections
- Video analysis with centroid tracking

DISCLAIMER: This is a research/educational demo only.
Not intended for medical diagnosis or clinical use.
"""

import os
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .inference import get_model
from .features import compute_all_features, get_centroid_for_class
from .viz import colorize_mask, overlay
from .utils import pil_to_base64, temp_video_file, validate_image_file, validate_video_file, bytes_to_mb
from .schemas import (
    HealthResponse,
    ImageAnalysisResponse,
    VideoAnalysisResponse,
    ErrorResponse,
    CLASS_NAMES
)


# Configuration from environment
NUM_CLASSES = int(os.environ.get('NUM_CLASSES', '5'))
IMAGE_SIZE = int(os.environ.get('IMAGE_SIZE', '256'))
VIDEO_MAX_FRAMES = int(os.environ.get('VIDEO_MAX_FRAMES', '30'))
VIDEO_FPS = float(os.environ.get('VIDEO_FPS', '2'))
MAX_UPLOAD_SIZE_MB = 25


# Create FastAPI app
app = FastAPI(
    title="Infant Pose Proxy from Segmentation Masks",
    description=(
        "Body part segmentation for infant pose estimation. "
        "**DISCLAIMER**: This is a research/educational demo only. "
        "Not intended for medical diagnosis or clinical use."
    ),
    version="1.0.0"
)


# Mount static files
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main UI."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(), status_code=200)
    return HTMLResponse(
        content="<h1>Infant Pose Proxy</h1><p>Static files not found. API is running.</p>",
        status_code=200
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    model = get_model()
    
    return HealthResponse(
        model_loaded=model.model_loaded,
        message="Model loaded successfully" if model.model_loaded else f"Model not loaded: {model.load_error}",
        config={
            "num_classes": NUM_CLASSES,
            "image_size": IMAGE_SIZE,
            "video_max_frames": VIDEO_MAX_FRAMES,
            "video_fps": VIDEO_FPS,
            "model_path": model.model_path
        }
    )


@app.post("/analyze_image", response_model=ImageAnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze a single image for body part segmentation.
    
    Returns segmentation mask, overlay, and body part detections.
    """
    # Validate file type
    if not file.filename or not validate_image_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an image (jpg, png, etc.)"
        )
    
    # Check file size
    content = await file.read()
    if bytes_to_mb(len(content)) > MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE_MB}MB"
        )
    
    # Check model
    model = get_model()
    if not model.model_loaded:
        raise HTTPException(
            status_code=503,
            detail=f"Model not available: {model.load_error}"
        )
    
    # Load image
    try:
        image = Image.open(BytesIO(content)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image: {str(e)}")
    
    # Run inference
    start_time = time.time()
    mask = model.predict(image)
    runtime_ms = (time.time() - start_time) * 1000
    
    # Compute features
    detections = compute_all_features(mask, NUM_CLASSES)
    
    # Create visualizations
    mask_rgb = colorize_mask(mask, NUM_CLASSES)
    
    # Resize original image to match mask size for overlay
    image_resized = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    overlay_img = overlay(np.array(image_resized), mask_rgb, alpha=0.5)
    
    # Encode to base64
    mask_pil = Image.fromarray(mask_rgb)
    overlay_pil = Image.fromarray(overlay_img)
    
    return ImageAnalysisResponse(
        num_classes=NUM_CLASSES,
        image_size=IMAGE_SIZE,
        runtime_ms=round(runtime_ms, 2),
        mask_png_base64=pil_to_base64(mask_pil),
        overlay_png_base64=pil_to_base64(overlay_pil),
        detections=detections
    )


@app.post("/analyze_video", response_model=VideoAnalysisResponse)
async def analyze_video(file: UploadFile = File(...)):
    """
    Analyze a video for body part tracking over time.
    
    Samples frames at VIDEO_FPS and tracks body part centroids.
    """
    # Validate file type
    if not file.filename or not validate_video_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a video (mp4, avi, mov, etc.)"
        )
    
    # Check file size
    content = await file.read()
    if bytes_to_mb(len(content)) > MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE_MB}MB"
        )
    
    # Check model
    model = get_model()
    if not model.model_loaded:
        raise HTTPException(
            status_code=503,
            detail=f"Model not available: {model.load_error}"
        )
    
    start_time = time.time()
    
    # Save video to temp file and process with OpenCV
    with temp_video_file(suffix=Path(file.filename).suffix) as video_path:
        video_path.write_bytes(content)
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Failed to open video file")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration_s = total_frames / video_fps
        
        # Calculate sampling
        sample_interval = video_fps / VIDEO_FPS  # frames between samples
        max_samples = min(VIDEO_MAX_FRAMES, int(duration_s * VIDEO_FPS))
        
        # Initialize tracking data
        tracks: dict[str, list[list[float]]] = {str(i): [] for i in range(1, NUM_CLASSES)}
        frame_timestamps: list[float] = []
        first_overlay: Optional[Image.Image] = None
        frames_processed = 0
        
        frame_idx = 0
        next_sample_frame = 0
        
        while cap.isOpened() and frames_processed < max_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx >= next_sample_frame:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                
                # Run segmentation
                mask = model.predict(image)
                
                # Record timestamp
                timestamp_s = frame_idx / video_fps
                frame_timestamps.append(round(timestamp_s, 3))
                
                # Extract centroids for each body part
                for class_id in range(1, NUM_CLASSES):
                    centroid = get_centroid_for_class(mask, class_id)
                    if centroid is not None:
                        cx, cy = centroid
                        tracks[str(class_id)].append([
                            round(timestamp_s, 3),
                            round(cx, 2),
                            round(cy, 2)
                        ])
                
                # Save first frame overlay
                if first_overlay is None:
                    mask_rgb = colorize_mask(mask, NUM_CLASSES)
                    image_resized = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
                    overlay_img = overlay(np.array(image_resized), mask_rgb, alpha=0.5)
                    first_overlay = Image.fromarray(overlay_img)
                
                frames_processed += 1
                next_sample_frame += sample_interval
            
            frame_idx += 1
        
        cap.release()
    
    runtime_ms = (time.time() - start_time) * 1000
    
    # Ensure we have at least a placeholder for first overlay
    if first_overlay is None:
        first_overlay = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (128, 128, 128))
    
    return VideoAnalysisResponse(
        num_classes=NUM_CLASSES,
        image_size=IMAGE_SIZE,
        sampled_frames=frames_processed,
        frame_timestamps_s=frame_timestamps,
        runtime_ms_total=round(runtime_ms, 2),
        first_overlay_png_base64=pil_to_base64(first_overlay),
        tracks=tracks
    )


# Import BytesIO at module level
from io import BytesIO


# Custom exception handler for file size
@app.exception_handler(413)
async def file_too_large_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=413,
        content={"error": "File too large", "detail": f"Maximum upload size is {MAX_UPLOAD_SIZE_MB}MB"}
    )


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    model = get_model()
    if model.model_loaded:
        print(f"Model loaded successfully from {model.model_path}")
    else:
        print(f"WARNING: Model not loaded - {model.load_error}")
        print("The API will start but prediction endpoints will return errors.")
