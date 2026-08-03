"""
CPU-only preprocessing untuk inference ONNX Runtime.
Tidak memakai dependency training berat supaya bisa deploy di CPU container SnapDeploy.
"""

from PIL import Image
import numpy as np

from src.config import IMG_SIZE

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
PRECHECK_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
PRECHECK_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)


def _pil_to_chw_float(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0  # HWC, range 0..1
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    return arr


def precheck_to_numpy(image: Image.Image) -> np.ndarray:
    """Return tensor numpy NCHW float32 untuk model precheck ONNX."""
    arr = _pil_to_chw_float(image)
    arr = (arr - PRECHECK_MEAN) / PRECHECK_STD
    return np.expand_dims(arr, axis=0).astype(np.float32)


def val_to_numpy(image: Image.Image) -> np.ndarray:
    """Return tensor numpy NCHW float32 untuk model hybrid ONNX."""
    arr = _pil_to_chw_float(image)
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return np.expand_dims(arr, axis=0).astype(np.float32)


# Alias lama agar import tidak rusak jika ada file lain yang masih mengacu.
precheck_transforms = precheck_to_numpy
val_transforms = val_to_numpy


def get_transforms():
    return None, val_to_numpy
