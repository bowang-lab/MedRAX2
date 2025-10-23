"""
Device Utility for Medical Imaging Tools

Handles device detection and configuration for PyTorch-based tools.
Ensures tools can run on CPU when CUDA is not available.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def get_device(preferred_device: Optional[str] = None, force_cpu: bool = False) -> str:
    """
    Get the appropriate device for PyTorch models with proper fallback.
    
    Args:
        preferred_device: Preferred device ("cuda", "cpu", "auto", or None)
        force_cpu: Force CPU usage even if CUDA is available
        
    Returns:
        str: Device string ("cuda" or "cpu")
        
    Environment Variables:
        CUDA: Set to "FALSE" or "false" to disable CUDA (force CPU mode)
        FORCE_CPU: Alternative to CUDA=FALSE (set to "true" to force CPU)
        DEVICE: Preferred device ("cuda", "cpu", or "auto")
        
    Examples:
        >>> get_device("auto")  # Auto-detect
        "cuda" or "cpu"
        
        >>> get_device("cuda")  # Try CUDA, fallback to CPU if not available
        "cuda" or "cpu"
        
        >>> get_device(force_cpu=True)  # Force CPU
        "cpu"
        
        >>> # With CUDA=FALSE environment variable
        >>> os.environ["CUDA"] = "FALSE"
        >>> get_device()  # Returns "cpu"
        "cpu"
    """
    # Force CPU if requested via parameter
    if force_cpu:
        logger.info("Device: CPU (forced by parameter)")
        return "cpu"
    
    # Check CUDA environment variable (primary way to disable CUDA)
    env_cuda = os.getenv("CUDA", "true").lower()
    if env_cuda in ("false", "0", "no", "off"):
        logger.info("Device: CPU (CUDA disabled by CUDA environment variable)")
        return "cpu"
    
    # Check legacy FORCE_CPU environment variable for backward compatibility
    env_force_cpu = os.getenv("FORCE_CPU", "false").lower() in ("true", "1", "yes", "on")
    if env_force_cpu:
        logger.info("Device: CPU (forced by FORCE_CPU environment variable)")
        return "cpu"
    
    # Check DEVICE environment variable
    env_device = os.getenv("DEVICE", "auto").lower()
    
    # Determine device priority: parameter > env_device > auto
    device = preferred_device or env_device
    
    if device == "auto":
        device = _auto_detect_device()
    elif device == "cuda":
        if not _is_cuda_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
    elif device == "cpu":
        pass  # Use CPU as requested
    else:
        logger.warning(f"Unknown device '{device}'. Using auto-detection.")
        device = _auto_detect_device()
    
    logger.info(f"Device: {device}")
    return device


def _is_cuda_available() -> bool:
    """
    Check if CUDA is available.
    
    Returns:
        bool: True if CUDA is available and functional
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
    except Exception as e:
        logger.warning(f"Error checking CUDA availability: {e}")
        return False


def _auto_detect_device() -> str:
    """
    Auto-detect the best available device.
    
    Returns:
        str: "cuda" if available, otherwise "cpu"
    """
    if _is_cuda_available():
        try:
            import torch
            # Test if CUDA is actually functional
            torch.cuda.current_device()
            logger.info("CUDA detected and functional")
            return "cuda"
        except Exception as e:
            logger.warning(f"CUDA detected but not functional: {e}. Using CPU.")
            return "cpu"
    else:
        logger.info("CUDA not available, using CPU")
        return "cpu"


def get_torch_device(preferred_device: Optional[str] = None, force_cpu: bool = False):
    """
    Get PyTorch device object with proper fallback.
    
    Args:
        preferred_device: Preferred device ("cuda", "cpu", "auto", or None)
        force_cpu: Force CPU usage even if CUDA is available
        
    Returns:
        torch.device: PyTorch device object
    """
    try:
        import torch
        device_str = get_device(preferred_device, force_cpu)
        return torch.device(device_str)
    except ImportError:
        raise ImportError("PyTorch is required. Install with: pip install torch")


def get_device_map(preferred_device: Optional[str] = None, force_cpu: bool = False) -> str:
    """
    Get device_map string for HuggingFace models with proper fallback.
    
    Args:
        preferred_device: Preferred device ("cuda", "cpu", "auto", or None)
        force_cpu: Force CPU usage even if CUDA is available
        
    Returns:
        str: Device map for HuggingFace models ("cuda", "cpu", or "auto")
    """
    device_str = get_device(preferred_device, force_cpu)
    
    # For HuggingFace models, we can use "auto" which intelligently distributes
    # the model across available devices
    if device_str == "cuda":
        return "auto"  # Let HuggingFace auto-distribute on GPU(s)
    else:
        return "cpu"


def check_gpu_availability() -> dict:
    """
    Check GPU availability and return detailed information.
    
    Returns:
        dict: GPU information including availability, count, names, and memory
    """
    info = {
        "cuda_available": False,
        "cuda_version": None,
        "device_count": 0,
        "devices": [],
    }
    
    try:
        import torch
        
        info["cuda_available"] = torch.cuda.is_available()
        
        if info["cuda_available"]:
            info["cuda_version"] = torch.version.cuda
            info["device_count"] = torch.cuda.device_count()
            
            for i in range(info["device_count"]):
                device_info = {
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "total_memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2),
                }
                info["devices"].append(device_info)
                
    except ImportError:
        logger.warning("PyTorch not installed. Cannot check GPU availability.")
    except Exception as e:
        logger.error(f"Error checking GPU availability: {e}")
    
    return info


def log_device_info():
    """Log detailed device information for debugging."""
    info = check_gpu_availability()
    
    logger.info("=" * 60)
    logger.info("DEVICE INFORMATION")
    logger.info("=" * 60)
    logger.info(f"CUDA Available: {info['cuda_available']}")
    
    if info['cuda_available']:
        logger.info(f"CUDA Version: {info['cuda_version']}")
        logger.info(f"GPU Count: {info['device_count']}")
        for device in info['devices']:
            logger.info(f"  GPU {device['id']}: {device['name']} ({device['total_memory_gb']} GB)")
    else:
        logger.info("Running on CPU only")
    
    logger.info("=" * 60)

