from typing import Dict, List, Optional, Tuple, Type, Any
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
import logging

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool

from medrax.utils.device import get_device, get_device_map

logger = logging.getLogger(__name__)


class XRayVQAToolInput(BaseModel):
    """Input schema for the CheXagent Tool."""

    image_paths: List[str] = Field(
        ..., 
        description="List of paths to chest X-ray images to analyze",
        json_schema_extra={"items": {"type": "string"}}  # Explicit for Gemini compatibility
    )
    prompt: str = Field(..., description="Question or instruction about the chest X-ray images")
    max_new_tokens: int = Field(512, description="Maximum number of tokens to generate in the response")


class CheXagentXRayVQATool(BaseTool):
    """Tool that leverages CheXagent for comprehensive chest X-ray analysis."""

    name: str = "chexagent_xray_vqa"
    description: str = (
        "A versatile tool for analyzing chest X-rays. "
        "Can perform multiple tasks including: visual question answering, report generation, "
        "abnormality detection, comparative analysis, anatomical description, "
        "and clinical interpretation. Input should be paths to X-ray images "
        "and a natural language prompt describing the analysis needed."
    )
    args_schema: Type[BaseModel] = XRayVQAToolInput
    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())
    return_direct: bool = True
    cache_dir: Optional[str] = None
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    tokenizer: Optional[AutoTokenizer] = None
    model: Optional[AutoModelForCausalLM] = None

    def __init__(
        self,
        model_name: str = "StanfordAIMI/CheXagent-2-3b",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        cache_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the CheXagentXRayVQATool.

        Args:
            model_name: Name of the CheXagent model to use
            device: Device to run model on (cuda/cpu/auto). If None, uses environment config.
            dtype: Data type for model weights
            cache_dir: Directory to cache downloaded models
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)

        self.device = get_device(device)
        # Choose dtype per accelerator: CUDA=bfloat16 (fast, widely supported), MPS=float16 (bfloat16 unsupported), CPU=float32
        if self.device == "cuda":
            self.dtype = dtype if dtype is not None else torch.bfloat16
        elif self.device == "mps":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        self.cache_dir = cache_dir
        
        logger.info(f"Initializing CheXagent VQA on device: {self.device}")
        
        # Check if model will work on CPU
        if self.device == "cpu":
            logger.warning("CheXagent VQA running on CPU. This will be significantly slower than GPU.")
            logger.warning("For better performance, consider using a system with CUDA support.")

        try:
            # Some remote CheXagent code enforces transformers==4.40.0.
            # Spoof version string during model/tokenizer load to pass checks, then restore.
            import transformers as _tf_mod
            _original_tf_version = getattr(_tf_mod, "__version__", None)
            try:
                _tf_mod.__version__ = "4.40.0"
            except Exception:
                pass

            # Load tokenizer
            logger.info(f"Loading tokenizer from {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=cache_dir,
            )
            
            # Load model with appropriate device mapping
            logger.info(f"Loading model from {model_name}...")
            device_map = get_device_map(self.device)
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                trust_remote_code=True,
                cache_dir=cache_dir,
                torch_dtype=self.dtype,
            )
            
            # Model is already initialized with appropriate dtype
            
            self.model.eval()
            
            logger.info("CheXagent VQA model loaded successfully")
            
            # Restore version string
            try:
                if _original_tf_version is not None:
                    _tf_mod.__version__ = _original_tf_version
            except Exception:
                pass

        except Exception as e:
            logger.error(f"Failed to initialize CheXagent VQA: {e}")
            raise

    def _generate_response(self, image_paths: List[str], prompt: str, max_new_tokens: int) -> str:
        """Generate response using CheXagent model.

        Args:
            image_paths: List of paths to chest X-ray images
            prompt: Question or instruction about the images
            max_new_tokens: Maximum number of tokens to generate
        Returns:
            str: Model's response
        """
        query = self.tokenizer.from_list_format([*[{"image": path} for path in image_paths], {"text": prompt}])
        conv = [
            {"from": "system", "value": "You are a helpful assistant."},
            {"from": "human", "value": query},
        ]
        # transformers 4.43 has chat templating; ensure tokenizer exposes it
        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise RuntimeError("This transformers version lacks chat templating; please upgrade to >=4.43.0 or set a chat_template.")
        input_ids = self.tokenizer.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt").to(
            device=self.device
        )

        # Run inference
        with torch.inference_mode():
            output = self.model.generate(
                input_ids,
                do_sample=False,
                num_beams=1,
                temperature=1.0,
                top_p=1.0,
                use_cache=True,
                max_new_tokens=max_new_tokens,
            )[0]
            response = self.tokenizer.decode(output[input_ids.size(1) : -1])

            return response

    def _run(
        self,
        image_paths: List[str],
        prompt: str,
        max_new_tokens: int = 512,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Execute the chest X-ray analysis.

        Args:
            image_paths: List of paths to chest X-ray images
            prompt: Question or instruction about the images
            max_new_tokens: Maximum number of tokens to generate
            run_manager: Optional callback manager

        Returns:
            Tuple[Dict[str, Any], Dict]: Output dictionary and metadata dictionary
        """
        try:
            # Verify image paths
            for path in image_paths:
                if not Path(path).is_file():
                    raise FileNotFoundError(f"Image file not found: {path}")

            response = self._generate_response(image_paths, prompt, max_new_tokens)

            output = {
                "response": response,
            }

            metadata = {
                "image_paths": image_paths,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "analysis_status": "completed",
            }

            return output, metadata

        except Exception as e:
            output = {"error": str(e)}
            metadata = {
                "image_paths": image_paths,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "analysis_status": "failed",
                "error_details": str(e),
            }
            return output, metadata

    async def _arun(
        self,
        image_paths: List[str],
        prompt: str,
        max_new_tokens: int = 512,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Async version of _run."""
        return self._run(image_paths, prompt, max_new_tokens)
