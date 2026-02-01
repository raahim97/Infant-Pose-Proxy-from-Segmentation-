"""
Pydantic schemas for API request/response models.
"""

from typing import Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    message: str = Field(..., description="Status message")
    config: dict = Field(..., description="Current configuration")


class BoundingBox(BaseModel):
    """Bounding box coordinates."""
    xmin: int
    ymin: int
    xmax: int
    ymax: int


class Keypoint(BaseModel):
    """Centroid keypoint coordinates."""
    cx: float
    cy: float


class Detection(BaseModel):
    """Single body part detection."""
    class_id: int = Field(..., description="Class ID (1-4)")
    class_name: str = Field(..., description="Class name")
    box: Optional[BoundingBox] = Field(None, description="Bounding box")
    keypoint: Optional[Keypoint] = Field(None, description="Centroid")
    area_px: int = Field(..., description="Area in pixels")


class ImageAnalysisResponse(BaseModel):
    """Response for image analysis endpoint."""
    num_classes: int = Field(..., description="Number of segmentation classes")
    image_size: int = Field(..., description="Processing image size")
    runtime_ms: float = Field(..., description="Inference runtime in milliseconds")
    mask_png_base64: str = Field(..., description="Colorized mask as base64 PNG")
    overlay_png_base64: str = Field(..., description="Overlay image as base64 PNG")
    detections: list[Detection] = Field(..., description="Detected body parts")


class TrackPoint(BaseModel):
    """Single point in a centroid track."""
    t: float = Field(..., description="Timestamp in seconds")
    cx: float = Field(..., description="X coordinate of centroid")
    cy: float = Field(..., description="Y coordinate of centroid")


class VideoAnalysisResponse(BaseModel):
    """Response for video analysis endpoint."""
    num_classes: int = Field(..., description="Number of segmentation classes")
    image_size: int = Field(..., description="Processing image size")
    sampled_frames: int = Field(..., description="Number of frames analyzed")
    frame_timestamps_s: list[float] = Field(..., description="Timestamps of sampled frames")
    runtime_ms_total: float = Field(..., description="Total runtime in milliseconds")
    first_overlay_png_base64: str = Field(..., description="First frame overlay as base64 PNG")
    tracks: dict[str, list[list[float]]] = Field(
        ..., 
        description="Centroid tracks per class: {class_id: [[t, cx, cy], ...]}"
    )


class ErrorResponse(BaseModel):
    """Error response."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")


# Class name mapping
CLASS_NAMES = {
    0: "background",
    1: "head",
    2: "hand/arm",
    3: "body/torso",
    4: "foot/leg"
}
