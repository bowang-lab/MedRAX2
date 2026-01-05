"""
Device Utilities for PyTorch Models

Provides automatic device detection for CPU/GPU/MPS (Apple Silicon).
"""

import torch
from typing import Optional


def get_optimal_device(preferred_device: Optional[str] = None) -> torch.device:
    """
    Get the optimal PyTorch device for model inference.
    
    Priority:
    1. Preferred device (if specified and available)
    2. CUDA (NVIDIA GPU)
    3. MPS (Apple Silicon GPU)
    4. CPU (fallback)
    
    Args:
        preferred_device: Optional device string ('cuda', 'mps', 'cpu')
        
    Returns:
        torch.device: The optimal device for this system
    """
    # If a specific device is requested, try to use it
    if preferred_device:
        try:
            device = torch.device(preferred_device)
            # Test if device is actually available
            if preferred_device == "cuda" and not torch.cuda.is_available():
                print(f"⚠️  CUDA requested but not available, falling back to auto-detect")
            elif preferred_device == "mps" and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                print(f"⚠️  MPS requested but not available, falling back to auto-detect")
            else:
                print(f"✅ Using requested device: {device}")
                return device
        except Exception as e:
            print(f"⚠️  Error with requested device '{preferred_device}': {e}, falling back to auto-detect")
    
    # Auto-detect best available device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"✅ Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print(f"ℹ️  Using CPU (no GPU available)")
    
    return device


def get_device_info() -> dict:
    """
    Get detailed information about available devices.
    
    Returns:
        dict: Device information including CUDA, MPS, and CPU details
    """
    info = {
        "cpu": True,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        "recommended_device": None,
        "details": {}
    }
    
    if info["cuda_available"]:
        info["recommended_device"] = "cuda"
        info["details"]["cuda"] = {
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None,
            "cuda_version": torch.version.cuda
        }
    elif info["mps_available"]:
        info["recommended_device"] = "mps"
        info["details"]["mps"] = {
            "backend": "Metal Performance Shaders",
            "platform": "Apple Silicon"
        }
    else:
        info["recommended_device"] = "cpu"
        info["details"]["cpu"] = {
            "processor": "CPU only (no GPU acceleration)"
        }
    
    return info


def is_gpu_available() -> bool:
    """
    Check if any GPU (CUDA or MPS) is available.
    
    Returns:
        bool: True if GPU is available, False otherwise
    """
    return torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())

