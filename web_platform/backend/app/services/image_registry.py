"""
Image Registry Service

Production-ready image path management system that prevents LLM transcription errors
by using simple indices instead of complex paths.
"""

import threading
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ImageRegistry:
    """
    Thread-safe registry for mapping image indices to file paths.
    
    This solves the LLM path corruption issue by allowing tools to reference
    images by simple indices (image_1, image_2, etc.) instead of complex paths.
    """
    
    def __init__(self):
        self._registry: Dict[str, Dict[str, str]] = {}  # request_id -> {index -> path}
        self._lock = threading.Lock()
    
    def register_images(self, request_id: str, image_paths: List[str]) -> Dict[str, str]:
        """
        Register images for a request and return index mapping.
        
        Args:
            request_id: Unique request identifier
            image_paths: List of image file paths
            
        Returns:
            Dictionary mapping indices to paths
        """
        with self._lock:
            mapping = {}
            for i, path in enumerate(image_paths, 1):
                index = f"image_{i}"
                mapping[index] = path
            
            self._registry[request_id] = mapping
            logger.info(f"Registered {len(mapping)} images for request {request_id[:8]}")
            return mapping
    
    def resolve_image(self, request_id: str, image_ref: str) -> Optional[str]:
        """
        Resolve an image reference to its actual path.
        
        Args:
            request_id: Request identifier
            image_ref: Either an index (image_1) or a path
            
        Returns:
            Actual file path or None if not found
        """
        with self._lock:
            # First check if it's an index reference
            if request_id in self._registry:
                if image_ref in self._registry[request_id]:
                    resolved = self._registry[request_id][image_ref]
                    logger.debug(f"Resolved {image_ref} -> {resolved}")
                    return resolved
                
                # Check if any registered path matches (for backward compatibility)
                for index, path in self._registry[request_id].items():
                    if path == image_ref or Path(path).name == Path(image_ref).name:
                        logger.debug(f"Resolved by path match: {image_ref} -> {path}")
                        return path
            
            # If not found in registry, check if it's a valid path
            if Path(image_ref).exists():
                logger.debug(f"Using direct path: {image_ref}")
                return image_ref
            
            # Try to find in any request (fallback for cross-request references)
            for req_id, mapping in self._registry.items():
                for index, path in mapping.items():
                    if Path(path).name == Path(image_ref).name:
                        logger.warning(f"Cross-request resolution: {image_ref} -> {path}")
                        return path
            
            logger.warning(f"Failed to resolve image reference: {image_ref}")
            return None
    
    def get_image_list(self, request_id: str) -> List[Tuple[str, str]]:
        """
        Get list of (index, path) tuples for a request.
        
        Args:
            request_id: Request identifier
            
        Returns:
            List of (index, path) tuples
        """
        with self._lock:
            if request_id in self._registry:
                return list(self._registry[request_id].items())
            return []
    
    def cleanup_request(self, request_id: str):
        """
        Remove a request's images from the registry.
        
        Args:
            request_id: Request identifier to clean up
        """
        with self._lock:
            if request_id in self._registry:
                count = len(self._registry[request_id])
                del self._registry[request_id]
                logger.debug(f"Cleaned up {count} images for request {request_id[:8]}")
    
    def clear_all(self):
        """Clear all registered images."""
        with self._lock:
            count = sum(len(mapping) for mapping in self._registry.values())
            self._registry.clear()
            logger.info(f"Cleared {count} images from registry")


# Global registry instance
image_registry = ImageRegistry()
