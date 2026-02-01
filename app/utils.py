"""
Utility functions for the API.
"""

import base64
import io
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from PIL import Image


def pil_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """
    Encode a PIL Image to base64 string.
    
    Args:
        image: PIL Image to encode
        format: Image format (PNG, JPEG, etc.)
    
    Returns:
        Base64-encoded string
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


def base64_to_pil(base64_str: str) -> Image.Image:
    """
    Decode a base64 string to PIL Image.
    
    Args:
        base64_str: Base64-encoded image string
    
    Returns:
        PIL Image
    """
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data))


@contextmanager
def temp_video_file(suffix: str = ".mp4") -> Generator[Path, None, None]:
    """
    Context manager for creating a temporary video file.
    
    Yields the path to a temporary file that will be deleted on exit.
    
    Args:
        suffix: File extension for the temp file
    
    Yields:
        Path to temporary file
    """
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        os.close(fd)
        yield Path(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def get_file_extension(filename: str) -> str:
    """
    Get lowercase file extension from filename.
    
    Args:
        filename: Original filename
    
    Returns:
        Lowercase extension including dot (e.g., ".mp4")
    """
    return Path(filename).suffix.lower()


def validate_image_file(filename: str) -> bool:
    """
    Check if filename has valid image extension.
    
    Args:
        filename: Original filename
    
    Returns:
        True if valid image extension
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    return get_file_extension(filename) in valid_extensions


def validate_video_file(filename: str) -> bool:
    """
    Check if filename has valid video extension.
    
    Args:
        filename: Original filename
    
    Returns:
        True if valid video extension
    """
    valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}
    return get_file_extension(filename) in valid_extensions


def bytes_to_mb(size_bytes: int) -> float:
    """
    Convert bytes to megabytes.
    
    Args:
        size_bytes: Size in bytes
    
    Returns:
        Size in megabytes
    """
    return size_bytes / (1024 * 1024)
