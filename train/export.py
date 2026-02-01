#!/usr/bin/env python3
"""
Export trained PyTorch model to ONNX format.

Exports the segmentation model with:
- Opset version 17
- Dynamic batch size
- Logits output (before argmax)
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large


def create_model(num_classes: int) -> nn.Module:
    """Create DeepLabV3 model architecture."""
    model = deeplabv3_mobilenet_v3_large(weights=None)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(40, num_classes, kernel_size=1)
    return model


class SegmentationWrapper(nn.Module):
    """Wrapper to output only the main logits tensor."""
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)['out']


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument('--checkpoint', type=str, default='models/segmentation.pt',
                        help='Path to PyTorch checkpoint')
    parser.add_argument('--output', type=str, default='models/segmentation.onnx',
                        help='Output ONNX path')
    parser.add_argument('--num_classes', type=int, default=None,
                        help='Number of classes (auto-detected from checkpoint)')
    parser.add_argument('--image_size', type=int, default=None,
                        help='Image size (auto-detected from checkpoint)')
    parser.add_argument('--opset', type=int, default=17,
                        help='ONNX opset version')
    args = parser.parse_args()
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Get hyperparameters from checkpoint or args
    num_classes = args.num_classes or checkpoint.get('num_classes', 5)
    image_size = args.image_size or checkpoint.get('image_size', 256)
    
    print(f"  Num classes: {num_classes}")
    print(f"  Image size: {image_size}")
    print(f"  Best mIoU: {checkpoint.get('best_miou', 'N/A')}")
    
    # Create model and load weights
    print("\nCreating model...")
    model = create_model(num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Wrap model to output only logits
    wrapped_model = SegmentationWrapper(model)
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, image_size, image_size)
    
    # Export to ONNX
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nExporting to ONNX (opset {args.opset})...")
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        str(output_path),
        opset_version=args.opset,
        input_names=['input'],
        output_names=['logits'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        },
        do_constant_folding=True
    )
    
    # Verify export
    print("\nVerifying ONNX model...")
    import onnx
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    
    # Print model info
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nExport successful!")
    print(f"  Output: {output_path}")
    print(f"  Size: {file_size_mb:.2f} MB")
    print(f"  Input shape: (batch, 3, {image_size}, {image_size})")
    print(f"  Output shape: (batch, {num_classes}, {image_size}, {image_size})")
    
    # Quick inference test
    print("\nTesting inference with ONNX Runtime...")
    import onnxruntime as ort
    
    session = ort.InferenceSession(str(output_path), providers=['CPUExecutionProvider'])
    test_input = dummy_input.numpy()
    outputs = session.run(None, {'input': test_input})
    
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {outputs[0].shape}")
    print(f"  Output dtype: {outputs[0].dtype}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
