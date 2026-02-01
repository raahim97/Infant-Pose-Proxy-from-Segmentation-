"""
Feature extraction from segmentation masks.

Computes bounding boxes and centroids for each body part.
"""

from typing import Optional
import numpy as np

from .schemas import BoundingBox, Keypoint, Detection, CLASS_NAMES


def compute_bounding_box(mask: np.ndarray, class_id: int) -> Optional[tuple[BoundingBox, int]]:
    """
    Compute bounding box for a specific class in the mask.
    
    Args:
        mask: HxW numpy array with class IDs
        class_id: Class ID to compute box for
    
    Returns:
        Tuple of (BoundingBox, area_px) or None if class not present
    """
    class_mask = (mask == class_id)
    
    if not np.any(class_mask):
        return None
    
    # Find bounding box coordinates
    rows = np.any(class_mask, axis=1)
    cols = np.any(class_mask, axis=0)
    
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    
    area_px = int(np.sum(class_mask))
    
    box = BoundingBox(
        xmin=int(xmin),
        ymin=int(ymin),
        xmax=int(xmax),
        ymax=int(ymax)
    )
    
    return box, area_px


def compute_centroid(mask: np.ndarray, class_id: int) -> Optional[tuple[Keypoint, int]]:
    """
    Compute centroid for a specific class in the mask.
    
    Args:
        mask: HxW numpy array with class IDs
        class_id: Class ID to compute centroid for
    
    Returns:
        Tuple of (Keypoint, area_px) or None if class not present
    """
    class_mask = (mask == class_id)
    
    if not np.any(class_mask):
        return None
    
    # Compute centroid
    y_coords, x_coords = np.where(class_mask)
    cx = float(np.mean(x_coords))
    cy = float(np.mean(y_coords))
    
    area_px = int(np.sum(class_mask))
    
    keypoint = Keypoint(cx=cx, cy=cy)
    
    return keypoint, area_px


def compute_bounding_boxes(mask: np.ndarray, num_classes: int = 5) -> list[Detection]:
    """
    Compute bounding boxes for all body part classes in the mask.
    
    Args:
        mask: HxW numpy array with class IDs
        num_classes: Total number of classes
    
    Returns:
        List of Detection objects with bounding boxes
    """
    detections = []
    
    # Skip background (class 0)
    for class_id in range(1, num_classes):
        result = compute_bounding_box(mask, class_id)
        
        if result is not None:
            box, area_px = result
            detection = Detection(
                class_id=class_id,
                class_name=CLASS_NAMES.get(class_id, f"class_{class_id}"),
                box=box,
                keypoint=None,
                area_px=area_px
            )
            detections.append(detection)
    
    return detections


def compute_centroids(mask: np.ndarray, num_classes: int = 5) -> list[Detection]:
    """
    Compute centroids for all body part classes in the mask.
    
    Args:
        mask: HxW numpy array with class IDs
        num_classes: Total number of classes
    
    Returns:
        List of Detection objects with keypoints
    """
    detections = []
    
    # Skip background (class 0)
    for class_id in range(1, num_classes):
        result = compute_centroid(mask, class_id)
        
        if result is not None:
            keypoint, area_px = result
            detection = Detection(
                class_id=class_id,
                class_name=CLASS_NAMES.get(class_id, f"class_{class_id}"),
                box=None,
                keypoint=keypoint,
                area_px=area_px
            )
            detections.append(detection)
    
    return detections


def compute_all_features(mask: np.ndarray, num_classes: int = 5) -> list[Detection]:
    """
    Compute both bounding boxes and centroids for all classes.
    
    Args:
        mask: HxW numpy array with class IDs
        num_classes: Total number of classes
    
    Returns:
        List of Detection objects with both boxes and keypoints
    """
    detections = []
    
    # Skip background (class 0)
    for class_id in range(1, num_classes):
        class_mask = (mask == class_id)
        
        if not np.any(class_mask):
            continue
        
        # Compute area
        area_px = int(np.sum(class_mask))
        
        # Compute bounding box
        rows = np.any(class_mask, axis=1)
        cols = np.any(class_mask, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        
        box = BoundingBox(
            xmin=int(xmin),
            ymin=int(ymin),
            xmax=int(xmax),
            ymax=int(ymax)
        )
        
        # Compute centroid
        y_coords, x_coords = np.where(class_mask)
        keypoint = Keypoint(
            cx=float(np.mean(x_coords)),
            cy=float(np.mean(y_coords))
        )
        
        detection = Detection(
            class_id=class_id,
            class_name=CLASS_NAMES.get(class_id, f"class_{class_id}"),
            box=box,
            keypoint=keypoint,
            area_px=area_px
        )
        detections.append(detection)
    
    return detections


def get_centroid_for_class(mask: np.ndarray, class_id: int) -> Optional[tuple[float, float]]:
    """
    Get simple (cx, cy) centroid for a class.
    
    Args:
        mask: HxW numpy array with class IDs
        class_id: Class ID
    
    Returns:
        (cx, cy) tuple or None
    """
    class_mask = (mask == class_id)
    
    if not np.any(class_mask):
        return None
    
    y_coords, x_coords = np.where(class_mask)
    return (float(np.mean(x_coords)), float(np.mean(y_coords)))
