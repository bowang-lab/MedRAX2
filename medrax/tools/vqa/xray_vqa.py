from typing import Dict, List, Optional, Tuple, Type, Any
from pathlib import Path
from pydantic import BaseModel, Field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool


def _patch_transformers_cache_api_for_chexagent() -> None:
    """Bridge old cache API used by CheXagent to newer transformers cache API."""
    try:
        from transformers.cache_utils import Cache
    except Exception:
        return

    if not hasattr(Cache, "get_max_length"):
        def get_max_length(self) -> Optional[int]:
            get_max_cache_shape = getattr(self, "get_max_cache_shape", None)
            if callable(get_max_cache_shape):
                return get_max_cache_shape()
            return None

        Cache.get_max_length = get_max_length  # type: ignore[attr-defined]


class XRayVQAToolInput(BaseModel):
    """Input schema for the CheXagent Tool."""

    image_paths: List[str] = Field(..., description="List of paths to chest X-ray images to analyze")
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
    return_direct: bool = True
    cache_dir: Optional[str] = None
    device: Optional[str] = None
    dtype: torch.dtype = torch.bfloat16
    tokenizer: Optional[AutoTokenizer] = None
    model: Optional[AutoModelForCausalLM] = None
    load_in_8bit: bool = True

    def __init__(
        self,
        model_name: str = "StanfordAIMI/CheXagent-2-3b",
        device: Optional[str] = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cache_dir: Optional[str] = None,
        load_in_8bit: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the CheXagentXRayVQATool.

        Args:
            model_name: Name of the CheXagent model to use
            device: Device to run model on (cuda/cpu)
            dtype: Data type for model weights
            cache_dir: Directory to cache downloaded models
            load_in_8bit: Whether to load model with bitsandbytes 8-bit quantization
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)

        # Dangerous code, but works for now
        import transformers

        original_transformers_version = transformers.__version__
        transformers.__version__ = "4.40.0"
        _patch_transformers_cache_api_for_chexagent()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.cache_dir = cache_dir
        self.load_in_8bit = load_in_8bit

        quantization_config = BitsAndBytesConfig(load_in_8bit=True) if load_in_8bit else None

        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=cache_dir,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto" if load_in_8bit else self.device,
                trust_remote_code=True,
                cache_dir=cache_dir,
                quantization_config=quantization_config,
            )
            if not load_in_8bit:
                self.model = self.model.to(dtype=self.dtype)
            self.model.eval()
        finally:
            transformers.__version__ = original_transformers_version

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