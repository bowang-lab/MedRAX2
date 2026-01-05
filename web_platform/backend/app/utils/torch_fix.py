"""
PyTorch 2.6+ Compatibility Fix

This module provides a context manager for loading older PyTorch models
that are not compatible with PyTorch 2.6+ weights_only=True default.

Usage:
    with torch_safe_load():
        model = torch.load('model.pth')
"""

import torch
from contextlib import contextmanager
from typing import Any


@contextmanager
def torch_safe_load():
    """
    Context manager to temporarily allow torch.load with weights_only=False.
    
    This is needed for loading models from trusted sources that use custom classes.
    Use this only when loading models from:
    - HuggingFace official repos
    - TorchXRayVision models
    - Other verified sources
    
    Example:
        with torch_safe_load():
            model = torch.load('trusted_model.pth')
    """
    original_load = torch.load
    
    def safe_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    
    torch.load = safe_load
    try:
        yield
    finally:
        torch.load = original_load


def apply_torch_safe_loading():
    """
    Apply safe loading globally for the session.
    
    This should be called once at startup if you trust all model sources.
    For production, prefer using the context manager approach.
    """
    original_load = torch.load
    
    def safe_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    
    torch.load = safe_load
    return original_load


# For backwards compatibility
apply_torch_load_patch = apply_torch_safe_loading

