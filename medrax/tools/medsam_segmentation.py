from typing import Dict, List, Optional, Tuple, Type, Any
from pathlib import Path
import uuid
import tempfile
import cv2

import numpy as np
import torch
import matplotlib.pyplot as plt
import skimage.io
import skimage.measure
import skimage.transform
import traceback
from huggingface_hub import hf_hub_download

from pydantic import BaseModel, Field
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("SAM2 not installed. Please install sam2 to use MedSAM2 functionality.")
    build_sam2 = None
    SAM2ImagePredictor = None


class ChestXRaySegmentationInput(BaseModel):
    """Input schema for the Chest X-ray Segmentation Tool."""

    image_path: str = Field(..., description="Path to the chest X-ray image file to be segmented")
    organs: Optional[List[str]] = Field(
        None,
        description="List of organs to segment. If None, all available organs will be segmented. "
        "Available organs: Left/Right Clavicle, Left/Right Scapula, Left/Right Lung, "
        "Left/Right Hilus Pulmonis, Heart, Aorta, Facies Diaphragmatica, "
        "Mediastinum, Weasand, Spine",
    )
    bounding_boxes: Optional[Dict[str, List[int]]] = Field(
        None,
        description="Optional bounding boxes for organs in format [x1, y1, x2, y2]. "
        "If not provided, default anatomical regions will be used."
    )


class OrganMetrics(BaseModel):
    """Detailed metrics for a segmented organ."""

    # Basic metrics
    area_pixels: int = Field(..., description="Area in pixels")
    area_cm2: float = Field(..., description="Approximate area in cm²")
    centroid: Tuple[float, float] = Field(..., description="(y, x) coordinates of centroid")
    bbox: Tuple[int, int, int, int] = Field(
        ..., description="Bounding box coordinates (min_y, min_x, max_y, max_x)"
    )

    # Size metrics
    width: int = Field(..., description="Width of the organ in pixels")
    height: int = Field(..., description="Height of the organ in pixels")
    aspect_ratio: float = Field(..., description="Height/width ratio")

    # Position metrics
    relative_position: Dict[str, float] = Field(
        ..., description="Position relative to image boundaries (0-1 scale)"
    )

    # Analysis metrics
    mean_intensity: float = Field(..., description="Mean pixel intensity in the organ region")
    std_intensity: float = Field(..., description="Standard deviation of pixel intensity")
    confidence_score: float = Field(..., description="Model confidence score for this organ")


class MedSamChestXRaySegmentationTool(BaseTool):
    """Tool for performing detailed segmentation analysis of chest X-ray images using MedSAM2."""

    name: str = "chest_xray_segmentation"
    description: str = (
        "Segments chest X-ray images to specified anatomical structures using MedSAM2. "
        "Available organs: Left/Right Clavicle (collar bones), Left/Right Scapula (shoulder blades), "
        "Left/Right Lung, Left/Right Hilus Pulmonis (lung roots), Heart, Aorta, "
        "Facies Diaphragmatica (diaphragm), Mediastinum (central cavity), Weasand (esophagus), "
        "and Spine. Returns segmentation visualization and comprehensive metrics. "
        "Let the user know the area is not accurate unless input has been DICOM."
    )
    args_schema: Type[BaseModel] = ChestXRaySegmentationInput

    predictor: Any = None
    device: str = "cuda"
    pixel_spacing_mm: float = 0.2
    temp_dir: Path = Path("temp")
    
    # Default anatomical bounding boxes (relative coordinates 0-1)
    default_organ_boxes: Dict[str, List[float]] = {
        "Left Clavicle": [0.15, 0.05, 0.45, 0.15],
        "Right Clavicle": [0.55, 0.05, 0.85, 0.15],
        "Left Scapula": [0.05, 0.15, 0.25, 0.45],
        "Right Scapula": [0.75, 0.15, 0.95, 0.45],
        "Left Lung": [0.1, 0.2, 0.45, 0.8],
        "Right Lung": [0.55, 0.2, 0.9, 0.8],
        "Left Hilus Pulmonis": [0.35, 0.35, 0.5, 0.55],
        "Right Hilus Pulmonis": [0.5, 0.35, 0.65, 0.55],
        "Heart": [0.35, 0.45, 0.65, 0.75],
        "Aorta": [0.42, 0.25, 0.58, 0.45],
        "Facies Diaphragmatica": [0.2, 0.75, 0.8, 0.85],
        "Mediastinum": [0.4, 0.3, 0.6, 0.7],
        "Weasand": [0.45, 0.1, 0.55, 0.3],
        "Spine": [0.45, 0.1, 0.55, 0.9],
    }

    def __init__(self, device: str = "cuda", temp_dir: Optional[Path] = Path("temp")):
        """Initialize the segmentation tool with MedSAM2 model."""
        super().__init__()
        self.device = device
        self.temp_dir = temp_dir if isinstance(temp_dir, Path) else Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        self._load_model()

    def _load_model(self):
        """Load the MedSAM2 model."""
        try:
            if build_sam2 is None or SAM2ImagePredictor is None:
                raise ImportError("SAM2 modules not available. Please install sam2.")
            
            # Download MedSAM2 checkpoint from Hugging Face
            checkpoint_path = hf_hub_download(repo_id="wanglab/MedSAM2", filename="MedSAM2_latest.pt")
            
            # Use a default SAM2 config - you may need to adjust this path
            # For now, we'll use a relative path that should work with typical SAM2 installations
            config_path = "sam2/configs/sam2_hiera_l.yaml"
            
            # Build SAM2 model with MedSAM2 checkpoint
            sam2_model = build_sam2(config_path, checkpoint_path, device=self.device)
            self.predictor = SAM2ImagePredictor(sam2_model)
            
            print(f"MedSAM2 model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading MedSAM2 model: {e}")
            raise

    def _convert_relative_to_absolute_bbox(self, rel_bbox: List[float], img_shape: Tuple[int, int]) -> List[int]:
        """Convert relative bounding box coordinates to absolute pixel coordinates."""
        h, w = img_shape
        x1, y1, x2, y2 = rel_bbox
        return [int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)]

    def _mask_to_bbox(self, mask: np.ndarray) -> Tuple[List[int], bool]:
        """Convert a mask to bounding box coordinates."""
        if mask.sum() == 0:
            return [0, 0, 0, 0], False
        
        coords = np.where(mask > 0)
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        return [x_min, y_min, x_max, y_max], True

    def _segment_organ_with_set_image(self, bbox: List[int], organ_name: str) -> Tuple[np.ndarray, float]:
        """Segment a single organ using MedSAM2 with bounding box prompt (image already set)."""
        try:
            # Convert bbox to numpy array and ensure correct format [x1, y1, x2, y2]
            bbox_np = np.array(bbox, dtype=np.float32)
            
            # Predict with bounding box prompt
            masks, scores, logits = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=bbox_np,  # SAM2 expects shape [4] for single box
                multimask_output=False,
                return_logits=False,
                normalize_coords=False,  # We're providing pixel coordinates
            )
            
            # Get the mask and its score
            # masks shape is [1, H, W] for multimask_output=False
            mask = masks[0].astype(np.uint8)  # Convert to uint8 (0 or 1)
            score = float(scores[0])
            
            return mask, score
            
        except Exception as e:
            print(f"Error segmenting {organ_name}: {e}")
            return np.zeros(self.predictor._orig_hw[0], dtype=np.uint8), 0.0

    def _segment_organ(self, image: np.ndarray, bbox: List[int], organ_name: str) -> Tuple[np.ndarray, float]:
        """Segment a single organ using MedSAM2 with bounding box prompt."""
        try:
            # Set the image for prediction (only set once per image)
            if not self.predictor._is_image_set:
                self.predictor.set_image(image)
            
            # Convert bbox to numpy array and ensure correct format [x1, y1, x2, y2]
            bbox_np = np.array(bbox, dtype=np.float32)
            
            # Predict with bounding box prompt
            masks, scores, logits = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=bbox_np,  # SAM2 expects shape [4] for single box
                multimask_output=False,
                return_logits=False,
                normalize_coords=False,  # We're providing pixel coordinates
            )
            
            # Get the mask and its score
            # masks shape is [1, H, W] for multimask_output=False
            mask = masks[0].astype(np.uint8)  # Convert to uint8 (0 or 1)
            score = float(scores[0])
            
            return mask, score
            
        except Exception as e:
            print(f"Error segmenting {organ_name}: {e}")
            return np.zeros(image.shape[:2], dtype=np.uint8), 0.0

    def _compute_organ_metrics(
        self, mask: np.ndarray, original_img: np.ndarray, confidence: float
    ) -> Optional[OrganMetrics]:
        """Compute comprehensive metrics for a single organ mask."""
        props = skimage.measure.regionprops(mask.astype(int))
        if not props:
            return None

        props = props[0]
        area_cm2 = mask.sum() * (self.pixel_spacing_mm / 10) ** 2

        img_height, img_width = mask.shape
        cy, cx = props.centroid
        relative_pos = {
            "top": cy / img_height,
            "left": cx / img_width,
            "center_dist": np.sqrt(((cy / img_height - 0.5) ** 2 + (cx / img_width - 0.5) ** 2)),
        }

        organ_pixels = original_img[mask > 0]
        mean_intensity = organ_pixels.mean() if len(organ_pixels) > 0 else 0
        std_intensity = organ_pixels.std() if len(organ_pixels) > 0 else 0

        return OrganMetrics(
            area_pixels=int(mask.sum()),
            area_cm2=float(area_cm2),
            centroid=(float(cy), float(cx)),
            bbox=tuple(map(int, props.bbox)),
            width=int(props.bbox[3] - props.bbox[1]),
            height=int(props.bbox[2] - props.bbox[0]),
            aspect_ratio=float(
                (props.bbox[2] - props.bbox[0]) / max(1, props.bbox[3] - props.bbox[1])
            ),
            relative_position=relative_pos,
            mean_intensity=float(mean_intensity),
            std_intensity=float(std_intensity),
            confidence_score=float(confidence),
        )

    def _save_visualization(
        self, 
        original_img: np.ndarray, 
        organ_masks: Dict[str, np.ndarray], 
        organ_bboxes: Dict[str, List[int]]
    ) -> str:
        """Save visualization of original image with segmentation masks and bounding boxes overlaid."""
        plt.figure(figsize=(12, 10))
        plt.imshow(original_img, cmap="gray")

        # Generate color palette for organs
        organ_names = list(organ_masks.keys())
        colors = plt.cm.rainbow(np.linspace(0, 1, len(organ_names)))

        # Process and overlay each organ mask
        for organ_name, color in zip(organ_names, colors):
            mask = organ_masks[organ_name]
            bbox = organ_bboxes[organ_name]
            
            if mask.sum() > 0:
                # Create a colored overlay with transparency
                colored_mask = np.zeros((*original_img.shape, 4))
                colored_mask[mask > 0] = (*color[:3], 0.4)
                plt.imshow(colored_mask)

                # Draw bounding box
                x1, y1, x2, y2 = bbox
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   fill=False, edgecolor=color, linewidth=2)
                plt.gca().add_patch(rect)

                # Add legend entry for the organ
                plt.plot([], [], color=color, label=organ_name, linewidth=3)

        plt.title("MedSAM2 Chest X-ray Segmentation")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.axis("off")

        save_path = self.temp_dir / f"medsam2_segmentation_{uuid.uuid4().hex[:8]}.png"
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()

        return str(save_path)

    def _run(
        self,
        image_path: str,
        organs: Optional[List[str]] = None,
        bounding_boxes: Optional[Dict[str, List[int]]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Run segmentation analysis for specified organs using MedSAM2."""
        try:
            # Validate organs
            if organs:
                organs = [o.strip() for o in organs]
                invalid_organs = [o for o in organs if o not in self.default_organ_boxes]
                if invalid_organs:
                    raise ValueError(f"Invalid organs specified: {invalid_organs}")
            else:
                organs = list(self.default_organ_boxes.keys())

            # Load and process image
            original_img = skimage.io.imread(image_path)
            if len(original_img.shape) == 3:
                # Convert BGR to RGB if needed
                if original_img.shape[2] == 3:
                    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            elif len(original_img.shape) == 2:
                # Convert grayscale to RGB
                original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)

            # Set the image once for all organ predictions
            self.predictor.set_image(original_img)

            # Segment each requested organ
            organ_masks = {}
            organ_bboxes = {}
            results = {}

            for organ_name in organs:
                # Get bounding box (custom or default)
                if bounding_boxes and organ_name in bounding_boxes:
                    bbox = bounding_boxes[organ_name]
                else:
                    rel_bbox = self.default_organ_boxes[organ_name]
                    bbox = self._convert_relative_to_absolute_bbox(rel_bbox, original_img.shape[:2])

                # Segment the organ (predictor already has the image set)
                mask, confidence = self._segment_organ_with_set_image(bbox, organ_name)
                
                if mask.sum() > 0:
                    organ_masks[organ_name] = mask
                    organ_bboxes[organ_name] = bbox
                    
                    # Compute metrics
                    gray_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
                    metrics = self._compute_organ_metrics(mask, gray_img, confidence)
                    if metrics:
                        results[organ_name] = metrics

            # Save visualization
            viz_path = self._save_visualization(
                cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY),
                organ_masks, 
                organ_bboxes
            )

            output = {
                "segmentation_image_path": viz_path,
                "metrics": {organ: metrics.dict() for organ, metrics in results.items()},
            }

            metadata = {
                "image_path": image_path,
                "segmentation_image_path": viz_path,
                "original_size": original_img.shape,
                "model_type": "MedSAM2",
                "pixel_spacing_mm": self.pixel_spacing_mm,
                "requested_organs": organs,
                "processed_organs": list(results.keys()),
                "analysis_status": "completed",
            }

            return output, metadata

        except Exception as e:
            error_output = {"error": str(e)}
            error_metadata = {
                "image_path": image_path,
                "analysis_status": "failed",
                "error_traceback": traceback.format_exc(),
            }
            return error_output, error_metadata

    async def _arun(
        self,
        image_path: str,
        organs: Optional[List[str]] = None,
        bounding_boxes: Optional[Dict[str, List[int]]] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Async version of _run."""
        return self._run(image_path, organs, bounding_boxes)
