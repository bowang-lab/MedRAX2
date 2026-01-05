from typing import Dict, List, Optional, Tuple, Type, Any
from pathlib import Path
import uuid
import tempfile
import logging
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field, ConfigDict

from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig, AutoConfig
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool

from medrax.utils.device import get_device, get_device_map

logger = logging.getLogger(__name__)


class XRayPhraseGroundingInput(BaseModel):
    """Input schema for the XRay Phrase Grounding Tool. Only supports JPG or PNG images."""

    image_path: str = Field(
        ...,
        description="Path to the frontal chest X-ray image file, only supports JPG or PNG images",
    )
    phrase: str = Field(
        ...,
        description="Medical finding or condition to locate in the image (e.g., 'Pleural effusion')",
    )
    max_new_tokens: int = Field(default=300, description="Maximum number of new tokens to generate")


class XRayPhraseGroundingTool(BaseTool):
    """Tool for grounding medical findings in chest X-ray images using the MAIRA-2 model.

    This tool processes chest X-ray images and locates specific medical findings mentioned
    in the input phrase. It returns both the bounding box coordinates and a visualization
    of the finding's location in the image.
    """

    name: str = "xray_phrase_grounding"
    description: str = (
        "Locates and visualizes specific medical findings in chest X-ray images. "
        "Takes a chest X-ray image and medical phrase to locate (e.g., 'Pleural effusion', 'Cardiomegaly'). "
        "Returns bounding box coordinates in format [x_topleft, y_topleft, x_bottomright, y_bottomright] "
        "where each value is between 0-1 representing relative position in the image, "
        "a visualization of the finding's location, and confidence metadata. "
        "Example input: {'image_path': '/path/to/xray.png', 'phrase': 'Pleural effusion', 'max_new_tokens': 300}"
    )
    args_schema: Type[BaseModel] = XRayPhraseGroundingInput
    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    model: Any = None
    processor: Any = None
    device: str = "cuda"
    temp_dir: Path = None

    def __init__(
        self,
        model_path: str = "microsoft/maira-2",
        cache_dir: Optional[str] = None,
        temp_dir: Optional[str] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        device: Optional[str] = None,
    ):
        """Initialize the XRay Phrase Grounding Tool."""
        super().__init__()
        
        # Patch transformers for MAIRA-2 compatibility
        import transformers
        if not hasattr(transformers, 'BaseImageProcessor'):
            logger.info("Pre-patching BaseImageProcessor for MAIRA-2 compatibility")
            if hasattr(transformers, 'ImageProcessingMixin'):
                transformers.BaseImageProcessor = transformers.ImageProcessingMixin
                logger.info("Using ImageProcessingMixin as BaseImageProcessor")
            elif hasattr(transformers, 'ProcessorMixin'):
                transformers.BaseImageProcessor = transformers.ProcessorMixin
                logger.info("Using ProcessorMixin as BaseImageProcessor")
            else:
                from transformers.processing_utils import ProcessorMixin
                transformers.BaseImageProcessor = ProcessorMixin
                logger.warning("Created minimal BaseImageProcessor for compatibility")
        
        # Patch LlavaProcessor to accept MAIRA-2 specific parameters
        from transformers import LlavaProcessor
        original_llava_init = LlavaProcessor.__init__
        
        def patched_llava_init(self, image_processor=None, tokenizer=None, patch_size=None, vision_feature_select_strategy=None, **kwargs):
            """Patched LlavaProcessor that accepts extra MAIRA-2 parameters and drops unsupported kwargs."""
            # Drop unsupported kwargs that can be supplied by remote processors (e.g., MAIRA-2)
            for key in ("chat_template", "conv_template", "chat_template_content"):
                if key in kwargs:
                    kwargs.pop(key, None)
                    logger.debug(f"Dropping unsupported LlavaProcessor kwarg: {key}")

            original_llava_init(self, image_processor=image_processor, tokenizer=tokenizer, **kwargs)
            if patch_size is not None:
                self.patch_size = patch_size
            if vision_feature_select_strategy is not None:
                self.vision_feature_select_strategy = vision_feature_select_strategy
        
        LlavaProcessor.__init__ = patched_llava_init
        logger.info("Patched LlavaProcessor to accept MAIRA-2 parameters")
        
        device_str = get_device(device)
        self.device = device_str
        
        logger.info(f"Initializing X-Ray Phrase Grounding on device: {device_str}")
        
        if device_str == "cpu":
            logger.warning("X-Ray Phrase Grounding running on CPU. This will be significantly slower than GPU.")

        quantization_config = None
        if device_str == "cuda":
            if load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            elif load_in_8bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
        elif load_in_4bit or load_in_8bit:
            logger.warning("Quantization (4-bit/8-bit) only available on CUDA. Loading full precision model.")

        # Load MAIRA-2 model - use the same approach as in dev branch
        # The key is to use AutoModelForCausalLM with device_map set to the device string directly
        logger.info("Loading MAIRA-2 model with trust_remote_code=True...")
        
        try:
            from transformers import AutoModelForCausalLM
            
            # Use the same loading approach as the dev branch
            # Pass device_str directly as device_map for MAIRA-2
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_str,  # Use device string directly like dev branch
                cache_dir=cache_dir,
                trust_remote_code=True,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16 if device_str == "cuda" else torch.float32,
            )
            logger.info(f"Model loaded successfully: {type(self.model).__name__}")
            
            # Verify the model has generation capabilities
            if not hasattr(self.model, 'generate'):
                raise AttributeError(f"Loaded model {type(self.model).__name__} doesn't have generate method")
                    
        except Exception as e:
            logger.error(f"Failed to load MAIRA-2 model: {e}")
            raise RuntimeError(f"Could not load MAIRA-2 model: {e}")
        
        logger.info("Loading processor...")
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        logger.info("Processor loaded successfully")

        self.model = self.model.eval()

        # Use local temp directory within project instead of system /tmp
        self.temp_dir = Path(temp_dir) if temp_dir else Path("temp/grounding")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("X-Ray Phrase Grounding model loaded successfully")

    def _visualize_bboxes(
        self, image: Image.Image, bboxes: List[Tuple[float, float, float, float]], phrase: str
    ) -> str:
        """Create and save visualization of multiple bounding boxes on the image."""
        plt.figure(figsize=(12, 12))
        plt.imshow(image)

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1

            plt.gca().add_patch(
                plt.Rectangle(
                    (x1 * image.width, y1 * image.height),
                    width * image.width,
                    height * image.height,
                    fill=False,
                    color="red",
                    linewidth=2,
                )
            )

        plt.title(f"Located: {phrase}", pad=20)
        plt.axis("off")

        viz_path = self.temp_dir / f"grounding_{uuid.uuid4().hex[:8]}.png"
        plt.savefig(viz_path, bbox_inches="tight", dpi=150)
        plt.close()

        return str(viz_path)

    def _run(
        self,
        image_path: str,
        phrase: str,
        max_new_tokens: int = 300,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Ground a medical finding phrase in an X-ray image.

        Args:
            image_path: Path to the chest X-ray image file
            phrase: Medical finding to locate in the image
            max_new_tokens: Maximum number of new tokens to generate
            run_manager: Optional callback manager

        Returns:
            Tuple[Dict, Dict]: Output dictionary and metadata dictionary
        """
        try:
            image = Image.open(image_path)
            
            # Properly handle 16-bit grayscale images (common in medical imaging)
            if image.mode == "I;16":
                # Convert 16-bit to 8-bit by normalizing to 0-255 range
                img_array = np.array(image)
                img_normalized = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype(np.uint8)
                image = Image.fromarray(img_normalized, mode='L')
            
            if image.mode != "RGB":
                image = image.convert("RGB")

            inputs = self.processor.format_and_preprocess_phrase_grounding_input(
                frontal_image=image, phrase=phrase, return_tensors="pt"
            )
            
            # Move inputs to device
            # MAIRA-2 model.generate() expects input_ids and attention_mask, but NOT pixel_values
            # The pixel_values are processed internally when you call the model
            device_inputs = {}
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    device_inputs[k] = v.to(self.device)

            # Try with all inputs first, then fallback if needed
            try:
                with torch.no_grad():
                    output = self.model.generate(
                        **device_inputs,
                        max_new_tokens=max_new_tokens,
                        use_cache=True,
                    )
            except (TypeError, AttributeError) as e:
                if "pixel_values" in str(e) or "model" in str(e):
                    # Fallback: remove pixel_values if it causes issues
                    logger.warning(f"Generation failed with all inputs, retrying without pixel_values: {e}")
                    generate_inputs = {k: v for k, v in device_inputs.items() if k != 'pixel_values'}
                    with torch.no_grad():
                        output = self.model.generate(
                            **generate_inputs,
                            max_new_tokens=max_new_tokens,
                            use_cache=True,
                        )
                else:
                    raise

            prompt_length = inputs["input_ids"].shape[-1]
            decoded_text = self.processor.decode(output[0][prompt_length:], skip_special_tokens=True)
            predictions = self.processor.convert_output_to_plaintext_or_grounded_sequence(decoded_text)

            metadata = {
                "image_path": image_path,
                "original_size": image.size,
                "model_input_size": tuple(inputs["pixel_values"].shape[-2:]),
                "device": str(self.device),
                "analysis_status": "completed",
            }

            if not predictions:
                output = {
                    "predictions": [],
                    "visualization_path": None,
                }
                metadata["analysis_status"] = "completed_no_finding"
                return output, metadata

            # Process multiple predictions
            processed_predictions = []
            for pred_phrase, pred_bboxes in predictions:
                if not pred_bboxes:  # Skip if no bounding boxes
                    continue

                # Convert model bboxes to list format and get original image bboxes
                model_bboxes = [list(bbox) for bbox in pred_bboxes]
                
                # Try to adjust boxes to original size, but fallback if processor method fails
                try:
                    original_bboxes = [
                        self.processor.adjust_box_for_original_image_size(bbox, width=image.size[0], height=image.size[1])
                        for bbox in model_bboxes
                    ]
                except Exception as e:
                    logger.warning(f"Failed to adjust box size with processor: {e}. Using model coordinates.")
                    # Fallback: use model coordinates as-is (they're already normalized 0-1)
                    original_bboxes = model_bboxes

                processed_predictions.append(
                    {
                        "phrase": pred_phrase,
                        "bounding_boxes": {
                            "model_coordinates": model_bboxes,
                            "image_coordinates": original_bboxes,
                        },
                    }
                )

            # Create visualization with all bounding boxes
            if processed_predictions:
                all_bboxes = []
                for pred in processed_predictions:
                    all_bboxes.extend(pred["bounding_boxes"]["image_coordinates"])
                viz_path = self._visualize_bboxes(image, all_bboxes, phrase)
            else:
                viz_path = None
                metadata["analysis_status"] = "completed_no_finding"

            output = {
                "predictions": processed_predictions,
                "visualization_path": viz_path,
            }

            return output, metadata

        except Exception as e:
            output = {"error": str(e)}
            metadata = {
                "image_path": image_path,
                "analysis_status": "failed",
                "error_details": str(e),
            }
            return output, metadata

    async def _arun(
        self,
        image_path: str,
        phrase: str,
        max_new_tokens: int = 300,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Asynchronous version of _run."""
        return self._run(image_path, phrase, max_new_tokens, run_manager)
