"""
MedGemma Direct Tool - Integrated into MedRAX Backend

This is the DIRECT integration of MedGemma that loads in the main MedRAX backend,
eliminating the need for a separate API server.

Use this instead of medgemma_client.py for single-server deployments.
"""

from typing import Dict, List, Optional, Tuple, Type, Any
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from PIL import Image
import torch
from transformers import (
    AutoProcessor,
    AutoModelForVisionText2Text,
)
import logging

# Try to import BitsAndBytesConfig, but don't fail if unavailable
try:
    from transformers import BitsAndBytesConfig
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False
    logging.warning("BitsAndBytesConfig not available, 4-bit quantization disabled")

logger = logging.getLogger(__name__)


class MedGemmaVQAInput(BaseModel):
    """Input schema for the MedGemma VQA Tool."""
    
    image_path: str = Field(
        default="",
        description="Path to medical image file to analyze, only supports JPG or PNG images"
    )
    image_paths: Optional[List[str]] = Field(
        default=None,
        description="Optional list of image paths (legacy format); first valid path will be used",
        json_schema_extra={"type": "array", "items": {"type": "string"}},
    )
    prompt: str = Field(..., description="Question or instruction about the medical image")
    system_prompt: Optional[str] = Field(
        "You are an expert radiologist.",
        description="System prompt to set the context for the model",
    )
    max_new_tokens: int = Field(
        300, description="Maximum number of tokens to generate in the response"
    )


class MedGemmaTool(BaseTool):
    """Medical visual question answering tool using Google's MedGemma 4B model.
    
    This is the DIRECT integration version that runs in the same process as MedRAX,
    eliminating the need for a separate API server.
    
    MedGemma is a specialized multimodal AI model trained on medical images and text.
    It provides expert-level analysis for chest X-rays, dermatology images,
    ophthalmology images, and histopathology slides.
    
    Key capabilities:
    - Medical image classification and analysis across multiple modalities
    - Visual question answering for radiology, dermatology, pathology, ophthalmology
    - Clinical reasoning and medical knowledge integration
    - Multi-modal medical understanding (text + images)
    - Support for up to 128K context length
    
    Performance:
    - Full precision (bfloat16): ~8GB VRAM, recommended for medical applications
    - 4-bit quantization: ~2GB VRAM, faster but may affect quality
    
    Resource Requirements:
    - Minimum: 8GB VRAM (GPU) or 16GB RAM (CPU)
    - Recommended: NVIDIA GPU with 8GB+ VRAM
    """
    
    name: str = "medgemma_medical_vqa"
    description: str = (
        "Advanced medical visual question answering tool using Google's MedGemma 4B instruction-tuned model. "
        "Specialized for comprehensive medical image analysis across multiple modalities including chest X-rays, "
        "dermatology images, ophthalmology images, and histopathology slides. Provides expert-level medical "
        "reasoning, diagnosis assistance, and detailed image interpretation with radiologist-level expertise. "
        "Input: List of medical image paths and medical question/prompt with optional custom system prompt. "
        "Output: Comprehensive medical analysis and answers based on visual content with detailed reasoning. "
        "Supports multi-image analysis, comparative studies, and complex medical reasoning tasks. "
        "Model handles images up to 896x896 resolution and supports context up to 128K tokens."
    )
    args_schema: Type[BaseModel] = MedGemmaVQAInput
    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())
    return_direct: bool = True
    
    # Model components
    pipe: Any = None  # kept for compatibility but not used after refactor
    processor: Any = None
    model: Any = None
    device: str = "cuda"
    model_name: str = "google/medgemma-4b-it"
    
    def __init__(
        self,
        model_name: str = "google/medgemma-4b-it",
        device: Optional[str] = None,
        use_4bit: bool = False,
        cache_dir: Optional[str] = None,
        **kwargs
    ):
        """Initialize the MedGemma tool.
        
        Args:
            model_name: Hugging Face model identifier
            device: Device to use ('cuda' or 'cpu', auto-detect if None)
            use_4bit: Use 4-bit quantization (saves VRAM, may affect quality)
            cache_dir: Directory for caching model files
            **kwargs: Additional arguments passed to BaseTool
        """
        super().__init__(**kwargs)
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model_name = model_name
        self._use_4bit = use_4bit
        self._cache_dir = cache_dir
        
        logger.info(f"MedGemma tool initialized (model will load on first use)")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  4-bit quantization: {use_4bit}")
        logger.info(f"  Cache dir: {cache_dir or 'default'}")
    
    def _ensure_model_loaded(self):
        """Lazy load processor + model (direct integration, no pipeline)."""
        if self.processor is not None and self.model is not None:
            return  # Already loaded
        
        logger.info(f"Loading MedGemma model (direct): {self.model_name}")
        logger.info(f"  This may take 1-2 minutes on first load (downloading ~8GB)...")
        
        try:
            # Configure quantization if requested
            quantization_config = None
            if self._use_4bit:
                if HAS_BITSANDBYTES:
                    logger.info("  Using 4-bit quantization (saves VRAM)")
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                else:
                    logger.warning("  4-bit quantization requested but BitsAndBytes not available, using full precision")
                    self._use_4bit = False
            
            # Determine dtype based on device
            if self.device == "cuda":
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32
                logger.warning("Using CPU - inference will be slow!")
            
            model_kwargs = {
                "torch_dtype": torch_dtype,
                "device_map": "auto" if self.device == "cuda" else None,
            }
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            if self._cache_dir:
                model_kwargs["cache_dir"] = self._cache_dir
            
            # Load processor and model directly (no pipeline) to keep control of image tokens
            self.processor = AutoProcessor.from_pretrained(self.model_name, **({"cache_dir": self._cache_dir} if self._cache_dir else {}))
            self.model = AutoModelForVisionText2Text.from_pretrained(self.model_name, **model_kwargs)
            
            logger.info(f"✅ MedGemma model loaded successfully (direct)")
            
            # Print GPU memory usage
            if self.device == "cuda" and torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"  GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
        
        except Exception as e:
            logger.error(f"Failed to load MedGemma model: {e}", exc_info=True)
            raise
    
    def _resolve_image_path(self, image_path: str, image_paths: Optional[List[str]]) -> str:
        """
        Support both single-path (current) and list-of-paths (legacy) inputs.
        Picks the first non-empty path.
        """
        if image_path:
            return image_path
        if image_paths:
            for p in image_paths:
                if p:
                    return p
        raise ValueError("No image path provided. Supply image_path or image_paths[0].")

    def _generate_direct(self, img: Image.Image, full_prompt: str, max_new_tokens: int):
        """
        Direct generation via processor + model (no pipeline). Ensures image tokens are inserted.
        """
        if self.processor is None or self.model is None:
            raise RuntimeError("MedGemma processor/model not loaded")

        # Preferred path: processor handles both text and images
        inputs = self.processor(text=[full_prompt], images=[img], return_tensors="pt")

        # Move to device
        device = self.model.device
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        # Strip num_crops if present (defensive)
        inputs.pop("num_crops", None)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        decoded = self.processor.batch_decode(outputs, skip_special_tokens=True)
        text = decoded[0] if decoded else ""
        return [{"generated_text": text}]

    def _run(
        self,
        image_path: str,
        prompt: str,
        system_prompt: str = "You are an expert radiologist.",
        max_new_tokens: int = 300,
        image_paths: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Execute medical visual question answering.
        
        Args:
            image_path: Path to medical image (current)
            image_paths: Optional list of images (legacy); first valid will be used
            prompt: Question or instruction about the image
            system_prompt: System context for the model
            max_new_tokens: Maximum number of tokens to generate
            run_manager: Optional callback manager
            
        Returns:
            Tuple of output dictionary and metadata
        """
        try:
            # Ensure model is loaded
            self._ensure_model_loaded()
            
            # Resolve and validate image path (support single path or list)
            resolved_path = self._resolve_image_path(image_path, image_paths)
            path_obj = Path(resolved_path)
            if not path_obj.exists():
                raise FileNotFoundError(f"Image file not found: {resolved_path}")
            if not path_obj.is_file():
                raise ValueError(f"Path is not a file: {resolved_path}")
            
            # Load image
            try:
                img = Image.open(resolved_path).convert("RGB")
            except Exception as e:
                raise ValueError(f"Failed to load image {resolved_path}: {e}")
            
            # Prepare prompt
            full_prompt = f"{system_prompt}\n\n{prompt}"
            
            # Generate response
            logger.info(f"Generating response for image: {resolved_path}")
            
            result = self._generate_direct(img, full_prompt, max_new_tokens)
            
            # Extract response
            response_text = result[0]["generated_text"] if result else ""
            
            output = {
                "response": response_text,
            }
            
            metadata = {
                "image_path": resolved_path,
                "prompt": prompt,
                "system_prompt": system_prompt,
                "max_new_tokens": max_new_tokens,
                "analysis_status": "completed",
                "model": self.model_name,
                "device": self.device,
            }
            
            logger.info(f"Response generated successfully ({len(response_text)} chars)")
            
            return output, metadata
        
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            return {"error": str(e)}, {
                "image_path": image_path,
                "prompt": prompt,
                "analysis_status": "failed",
                "error_type": "FileNotFoundError",
                "error_details": str(e),
            }
        
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU out of memory: {e}")
            return {"error": "GPU memory exhausted. Try reducing image resolution or max_new_tokens."}, {
                "image_path": image_path,
                "prompt": prompt,
                "analysis_status": "failed",
                "error_type": "OutOfMemoryError",
                "error_details": str(e),
            }
        
        except Exception as e:
            logger.error(f"MedGemma analysis failed: {e}", exc_info=True)
            return {"error": str(e)}, {
                "image_path": image_path,
                "prompt": prompt,
                "analysis_status": "failed",
                "error_type": type(e).__name__,
                "error_details": str(e),
            }
    
    async def _arun(
        self,
        image_path: str,
        prompt: str,
        system_prompt: str = "You are an expert radiologist.",
        max_new_tokens: int = 300,
        image_paths: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Async version of _run (currently calls sync version)."""
        return self._run(image_path, prompt, system_prompt, max_new_tokens, image_paths, run_manager)
    
    def cleanup(self):
        """Cleanup method called when tool is unloaded."""
        if self.pipe is not None:
            logger.info("Unloading MedGemma model...")
            del self.pipe
            self.pipe = None
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("CUDA cache cleared")
            
            logger.info("MedGemma model unloaded")

