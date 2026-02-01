"""
ONNX inference engine for segmentation model.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# ONNX Runtime - imported conditionally
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class SegmentationModel:
    """
    ONNX-based segmentation model for inference.
    
    Handles model loading, preprocessing, and prediction.
    App can start even if model file is missing.
    """
    
    # ImageNet normalization constants
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        num_classes: int = 5,
        image_size: int = 256
    ):
        """
        Initialize the segmentation model.
        
        Args:
            model_path: Path to ONNX model file
            num_classes: Number of segmentation classes
            image_size: Input image size for the model
        """
        self.model_path = model_path or os.environ.get('MODEL_PATH', 'models/segmentation.onnx')
        self.num_classes = num_classes
        self.image_size = image_size
        
        self.session: Optional[ort.InferenceSession] = None
        self.model_loaded = False
        self.load_error: Optional[str] = None
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Attempt to load the ONNX model."""
        if not ONNX_AVAILABLE:
            self.load_error = "ONNX Runtime not installed"
            return
        
        model_path = Path(self.model_path)
        
        if not model_path.exists():
            self.load_error = f"Model file not found: {model_path}"
            return
        
        try:
            # Use CPU provider only for consistent behavior
            self.session = ort.InferenceSession(
                str(model_path),
                providers=['CPUExecutionProvider']
            )
            self.model_loaded = True
            self.load_error = None
        except Exception as e:
            self.load_error = f"Failed to load model: {str(e)}"
            self.session = None
    
    def preprocess(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess image for model input.
        
        Args:
            image: PIL Image (any mode, any size)
        
        Returns:
            NCHW float32 numpy array, ImageNet normalized
        """
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        
        # Convert to numpy float32 [0, 1]
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        # ImageNet normalization
        img_array = (img_array - self.MEAN) / self.STD
        
        # HWC -> CHW
        img_array = img_array.transpose(2, 0, 1)
        
        # Add batch dimension: CHW -> NCHW
        img_array = img_array[np.newaxis, ...]
        
        return img_array
    
    def predict(self, image: Image.Image) -> np.ndarray:
        """
        Run segmentation prediction on an image.
        
        Args:
            image: PIL Image
        
        Returns:
            HxW uint8 numpy array with class IDs
        
        Raises:
            RuntimeError: If model is not loaded
        """
        if not self.model_loaded or self.session is None:
            raise RuntimeError(
                f"Model not loaded: {self.load_error or 'Unknown error'}"
            )
        
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Run inference
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_tensor})
        
        # outputs[0] shape: (1, num_classes, H, W) - logits
        logits = outputs[0]
        
        # Argmax to get class predictions
        mask = np.argmax(logits[0], axis=0).astype(np.uint8)
        
        return mask
    
    def predict_batch(self, images: list[Image.Image]) -> list[np.ndarray]:
        """
        Run segmentation prediction on a batch of images.
        
        Args:
            images: List of PIL Images
        
        Returns:
            List of HxW uint8 numpy arrays with class IDs
        """
        if not self.model_loaded or self.session is None:
            raise RuntimeError(
                f"Model not loaded: {self.load_error or 'Unknown error'}"
            )
        
        # Preprocess all images
        input_batch = np.concatenate(
            [self.preprocess(img) for img in images],
            axis=0
        )
        
        # Run inference
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_batch})
        
        # Process each output
        logits = outputs[0]  # (N, num_classes, H, W)
        masks = []
        
        for i in range(logits.shape[0]):
            mask = np.argmax(logits[i], axis=0).astype(np.uint8)
            masks.append(mask)
        
        return masks
    
    @property
    def status(self) -> dict:
        """Get model status information."""
        return {
            'model_loaded': self.model_loaded,
            'model_path': str(self.model_path),
            'num_classes': self.num_classes,
            'image_size': self.image_size,
            'error': self.load_error
        }


# Global model instance (lazy loaded)
_model_instance: Optional[SegmentationModel] = None


def get_model() -> SegmentationModel:
    """
    Get or create the global model instance.
    
    Returns:
        SegmentationModel instance
    """
    global _model_instance
    
    if _model_instance is None:
        model_path = os.environ.get('MODEL_PATH', 'models/segmentation.onnx')
        num_classes = int(os.environ.get('NUM_CLASSES', '5'))
        image_size = int(os.environ.get('IMAGE_SIZE', '256'))
        
        _model_instance = SegmentationModel(
            model_path=model_path,
            num_classes=num_classes,
            image_size=image_size
        )
    
    return _model_instance


def reset_model() -> None:
    """Reset the global model instance (useful for testing)."""
    global _model_instance
    _model_instance = None
