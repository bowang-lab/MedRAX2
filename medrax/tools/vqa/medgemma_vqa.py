from typing import Dict, List, Optional, Tuple, Type, Any
from pathlib import Path
from pydantic import BaseModel, Field
import warnings

import torch
from PIL import Image
from transformers import pipeline, BitsAndBytesConfig

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool


class MedGemmaVQAInput(BaseModel):
    """Input schema for the MedGemma VQA Tool. Only supports JPG or PNG images."""

    image_paths: List[str] = Field(
        ...,
        description="List of paths to medical image files to analyze, only supports JPG or PNG images",
    )
    prompt: str = Field(..., description="Question or instruction about the medical images")
    system_prompt: Optional[str] = Field(
        "You are an expert radiologist.",
        description="System prompt to set the context for the model",
    )
    max_new_tokens: int = Field(
        300, description="Maximum number of tokens to generate in the response"
    )


class MedGemmaVQATool(BaseTool):
    """Medical visual question answering tool using Google's MedGemma 4B model.

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
    - 4-bit quantization (default): Available but may affect quality on some systems
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
    return_direct: bool = True

    # Model components
    pipe: Optional[Any] = None  # transformers pipeline
    device: Optional[str] = "cuda"
    cache_dir: Optional[str] = None
    dtype: torch.dtype = torch.bfloat16

    def __init__(
        self,
        model_name: str = "google/medgemma-4b-it",
        device: Optional[str] = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cache_dir: Optional[str] = None,
        load_in_4bit: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the MedGemmaVQATool.

        Args:
            model_name: Name of the MedGemma model to use
            device: Device to run model on (cuda/cpu)
            dtype: Data type for model weights (bfloat16 recommended)
            cache_dir: Directory to cache downloaded models
            load_in_4bit: Whether to load model in 4-bit quantization
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.cache_dir = cache_dir

        # Setup model configuration
        model_kwargs = {
            "torch_dtype": self.dtype,
        }

        if cache_dir:
            model_kwargs["cache_dir"] = cache_dir

        # Handle device mapping and quantization
        pipeline_kwargs = {
            "model": model_name,
            "model_kwargs": model_kwargs,
            "trust_remote_code": True,
            "use_cache": True,
        }

        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        model_kwargs["device_map"] = {"": self.device}

        try:
            self.pipe = pipeline("image-text-to-text", **pipeline_kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize MedGemma pipeline: {str(e)}")

    def _prepare_messages(
        self, image_paths: List[str], prompt: str, system_prompt: str
    ) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
        """Prepare chat messages in the format expected by MedGemma.

        Args:
            image_paths: List of paths to medical images
            prompt: User's question or instruction
            system_prompt: System context message

        Returns:
            Tuple of formatted messages and loaded images
        """
        images = []
        for path in image_paths:
            if not Path(path).is_file():
                raise FileNotFoundError(f"Image file not found: {path}")

            image = Image.open(path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            images.append(image)

        # Create messages in chat format
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
                + [{"type": "image", "image": img} for img in images],
            },
        ]

        return messages, images

    def _generate_response(self, messages: List[Dict[str, Any]], max_new_tokens: int) -> str:
        """Generate response using MedGemma pipeline.

        Args:
            messages: Formatted chat messages
            max_new_tokens: Maximum number of tokens to generate

        Returns:
            Generated response text
        """
        # Generate using pipeline
        output = self.pipe(
            text=messages,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        # Extract generated text from pipeline output
        if (
            isinstance(output, list)
            and output
            and isinstance(output[0].get("generated_text"), list)
        ):
            generated_text = output[0]["generated_text"]
            if generated_text:
                return generated_text[-1].get("content", "").strip()

        return "No response generated"

    def _run(
        self,
        image_paths: List[str],
        prompt: str,
        system_prompt: str = "You are an expert radiologist.",
        max_new_tokens: int = 300,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Execute medical visual question answering.

        Args:
            image_paths: List of paths to medical images
            prompt: Question or instruction about the images
            system_prompt: System context for the model
            max_new_tokens: Maximum number of tokens to generate
            run_manager: Optional callback manager

        Returns:
            Tuple of output dictionary and metadata
        """
        try:
            # Prepare messages and load images
            messages, images = self._prepare_messages(image_paths, prompt, system_prompt)

            # Generate response
            response = self._generate_response(messages, max_new_tokens)

            output = {"response": response}

            metadata = {
                "image_paths": image_paths,
                "prompt": prompt,
                "system_prompt": system_prompt,
                "max_new_tokens": max_new_tokens,
                "num_images": len(image_paths),
                "analysis_status": "completed",
            }

            return output, metadata

        except FileNotFoundError as e:
            return self._create_error_response(
                image_paths, prompt, f"Image file not found: {str(e)}", "file_not_found", str(e)
            )

        except torch.cuda.OutOfMemoryError as e:
            return self._create_error_response(
                image_paths,
                prompt,
                "GPU memory exhausted. Try reducing image resolution or max_new_tokens.",
                "memory_error",
                str(e),
            )

        except Exception as e:
            return self._create_error_response(
                image_paths, prompt, f"Analysis failed: {str(e)}", "general_error", str(e)
            )

    def _create_error_response(
        self,
        image_paths: List[str],
        prompt: str,
        error_message: str,
        error_type: str,
        error_details: str,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Create standardized error response.

        Args:
            image_paths: List of image paths
            prompt: User prompt
            error_message: Human-readable error message
            error_type: Type of error
            error_details: Detailed error information

        Returns:
            Tuple of error output and metadata
        """
        output = {"error": error_message}
        metadata = {
            "image_paths": image_paths,
            "prompt": prompt,
            "analysis_status": "failed",
            "error_type": error_type,
            "error_details": error_details,
        }
        return output, metadata

    async def _arun(
        self,
        image_paths: List[str],
        prompt: str,
        system_prompt: str = "You are an expert radiologist.",
        max_new_tokens: int = 300,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Async version of _run."""
        return self._run(image_paths, prompt, system_prompt, max_new_tokens)
