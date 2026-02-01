"""
Visualization utilities for segmentation masks.
"""

import numpy as np
from PIL import Image


# Fixed color palette for body parts (RGB)
# Matches the original dataset colors
CLASS_COLORS = {
    0: (0, 0, 0),         # background - black
    1: (44, 160, 44),     # head - green
    2: (31, 119, 180),    # hand/arm - blue
    3: (255, 127, 14),    # body/torso - orange
    4: (214, 39, 40),     # foot/leg - red
}


def colorize_mask(mask: np.ndarray, num_classes: int = 5) -> np.ndarray:
    """
    Convert single-channel class ID mask to RGB color mask.
    
    Args:
        mask: HxW numpy array with class IDs (uint8)
        num_classes: Number of classes
    
    Returns:
        HxWx3 numpy array (uint8) with RGB colors
    """
    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id in range(num_classes):
        color = CLASS_COLORS.get(class_id, (128, 128, 128))
        rgb_mask[mask == class_id] = color
    
    return rgb_mask


def overlay(
    image: np.ndarray | Image.Image,
    mask_rgb: np.ndarray | Image.Image,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Blend segmentation mask over original image.
    
    Args:
        image: Original image (HxWx3 numpy array or PIL Image)
        mask_rgb: Colorized mask (HxWx3 numpy array or PIL Image)
        alpha: Blend factor for mask (0=only image, 1=only mask)
    
    Returns:
        HxWx3 numpy array (uint8) with blended result
    """
    # Convert to numpy if PIL
    if isinstance(image, Image.Image):
        image = np.array(image)
    if isinstance(mask_rgb, Image.Image):
        mask_rgb = np.array(mask_rgb)
    
    # Ensure same size
    if image.shape[:2] != mask_rgb.shape[:2]:
        mask_pil = Image.fromarray(mask_rgb)
        mask_pil = mask_pil.resize((image.shape[1], image.shape[0]), Image.NEAREST)
        mask_rgb = np.array(mask_pil)
    
    # Blend where mask is not background
    result = image.copy().astype(np.float32)
    mask_float = mask_rgb.astype(np.float32)
    
    # Only blend non-background regions
    non_bg_mask = np.any(mask_rgb > 0, axis=2)
    
    for c in range(3):
        result[:, :, c] = np.where(
            non_bg_mask,
            (1 - alpha) * image[:, :, c] + alpha * mask_float[:, :, c],
            result[:, :, c]
        )
    
    return result.astype(np.uint8)


def create_legend(num_classes: int = 5, cell_size: int = 30) -> Image.Image:
    """
    Create a legend image showing class colors and names.
    
    Args:
        num_classes: Number of classes
        cell_size: Size of each color cell
    
    Returns:
        PIL Image with legend
    """
    from PIL import ImageDraw, ImageFont
    
    class_names = {
        0: "background",
        1: "head",
        2: "hand/arm",
        3: "body/torso",
        4: "foot/leg"
    }
    
    # Create image
    width = 200
    height = num_classes * cell_size
    legend = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(legend)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()
    
    for class_id in range(num_classes):
        y = class_id * cell_size
        color = CLASS_COLORS.get(class_id, (128, 128, 128))
        name = class_names.get(class_id, f"class_{class_id}")
        
        # Draw color box
        draw.rectangle([5, y + 5, 5 + cell_size - 10, y + cell_size - 5], fill=color)
        
        # Draw text
        draw.text((cell_size + 5, y + 8), name, fill=(0, 0, 0), font=font)
    
    return legend


def get_class_color(class_id: int) -> tuple[int, int, int]:
    """
    Get RGB color for a class ID.
    
    Args:
        class_id: Class ID
    
    Returns:
        RGB tuple
    """
    return CLASS_COLORS.get(class_id, (128, 128, 128))
