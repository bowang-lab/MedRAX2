from typing import Dict, Optional, Tuple, Type
from pydantic import BaseModel, Field, ConfigDict
import logging

import skimage.io
import torch
import torchvision

# Fix PyTorch 2.6+ weights_only issue BEFORE importing torchxrayvision
# The torchxrayvision library uses torch.load internally
_original_torch_load = torch.load
torch.load = lambda *args, **kwargs: _original_torch_load(*args, **{**kwargs, 'weights_only': kwargs.get('weights_only', False)})

import torchxrayvision as xrv

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool

from medrax.utils.utils import preprocess_medical_image
from medrax.utils.device import get_device

logger = logging.getLogger(__name__)


class TorchXRayVisionInput(BaseModel):
    """Input for TorchXRayVision chest X-ray analysis tools. Only supports JPG or PNG images."""

    image_path: str = Field(..., description="Path to the radiology image file, only supports JPG or PNG images")


class TorchXRayVisionClassifierTool(BaseTool):
    """Tool that classifies chest X-ray images for multiple pathologies.

    This tool uses a pre-trained DenseNet model to analyze chest X-ray images and
    predict the likelihood of various pathologies. The model can classify the following 18 conditions:

    Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema,
    Enlarged Cardiomediastinum, Fibrosis, Fracture, Hernia, Infiltration,
    Lung Lesion, Lung Opacity, Mass, Nodule, Pleural Thickening, Pneumonia, Pneumothorax

    The output values represent the probability (from 0 to 1) of each condition being present in the image.
    A higher value indicates a higher likelihood of the condition being present.
    """

    name: str = "torchxrayvision_classifier"
    description: str = (
        "A tool that analyzes chest X-ray images and classifies them for 18 different pathologies using TorchXRayVision DenseNet. "
        "Input should be the path to a chest X-ray image file. "
        "Output is a dictionary of pathologies and their predicted probabilities (0 to 1). "
        "Pathologies include: Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, "
        "Enlarged Cardiomediastinum, Fibrosis, Fracture, Hernia, Infiltration, Lung Lesion, "
        "Lung Opacity, Mass, Nodule, Pleural Thickening, Pneumonia, and Pneumothorax. "
        "Higher values indicate a higher likelihood of the condition being present."
    )
    args_schema: Type[BaseModel] = TorchXRayVisionInput
    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())
    model: xrv.models.DenseNet = None
    device: str = "cuda"
    image_transform: torchvision.transforms.Compose = None

    def __init__(self, model_name: str = "densenet121-res224-all", device: Optional[str] = None):
        super().__init__()
        

        device_str = get_device(device)
        self.device = torch.device(device_str)
        
        logger.info(f"Initializing TorchXRayVision on device: {device_str}")
        
        if device_str == "cpu":
            logger.warning("TorchXRayVision running on CPU. This will be slower than GPU.")
        
        self.model = xrv.models.DenseNet(weights=model_name)
        self.model.eval()
        
        # Ensure model is in float32 to avoid dtype mismatches
        # torchxrayvision models may load with mixed dtypes
        self.model = self.model.float()
        self.model = self.model.to(self.device)
        
        self.image_transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop()])
        
        logger.info("TorchXRayVision model loaded successfully")

    def _process_image(self, image_path: str) -> torch.Tensor:
        """
        Process the input chest X-ray image for model inference.

        This method loads the image, normalizes it, applies necessary transformations,
        and prepares it as a torch.Tensor for model input.

        Args:
            image_path (str): The file path to the chest X-ray image.

        Returns:
            torch.Tensor: A processed image tensor ready for model inference.

        Raises:
            FileNotFoundError: If the specified image file does not exist.
            ValueError: If the image cannot be properly loaded or processed.
        """
        img = skimage.io.imread(image_path)
        
        # Use robust normalization that handles both 8-bit and 16-bit images
        img = preprocess_medical_image(img, target_range=(-1024.0, 1024.0))

        if len(img.shape) > 2:
            img = img[:, :, 0]

        img = img[None, :, :]
        img = self.image_transform(img)
        img = torch.from_numpy(img).unsqueeze(0)

        # Ensure tensor is float32 to match model dtype
        img = img.float().to(self.device)

        return img

    def _run(
        self,
        image_path: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, float], Dict]:
        """Classify the chest X-ray image for multiple pathologies.

        Args:
            image_path (str): The path to the chest X-ray image file.
            run_manager (Optional[CallbackManagerForToolRun]): The callback manager for the tool run.

        Returns:
            Tuple[Dict[str, float], Dict]: A tuple containing the classification results
                                           (pathologies and their probabilities from 0 to 1)
                                           and any additional metadata.

        Raises:
            Exception: If there's an error processing the image or during classification.
        """
        try:
            img = self._process_image(image_path)

            with torch.inference_mode():
                preds = self.model(img).cpu()[0]

            output = dict(zip(xrv.datasets.default_pathologies, preds.numpy()))
            metadata = {
                "image_path": image_path,
                "analysis_status": "completed",
                "note": "Probabilities range from 0 to 1, with higher values indicating higher likelihood of the condition.",
            }
            return output, metadata
        except Exception as e:
            return {"error": str(e)}, {
                "image_path": image_path,
                "analysis_status": "failed",
            }

    async def _arun(
        self,
        image_path: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, float], Dict]:
        """Asynchronously classify the chest X-ray image for multiple pathologies.

        This method currently calls the synchronous version, as the model inference
        is not inherently asynchronous. For true asynchronous behavior, consider
        using a separate thread or process.

        Args:
            image_path (str): The path to the chest X-ray image file.
            run_manager (Optional[AsyncCallbackManagerForToolRun]): The async callback manager for the tool run.

        Returns:
            Tuple[Dict[str, float], Dict]: A tuple containing the classification results
                                           (pathologies and their probabilities from 0 to 1)
                                           and any additional metadata.

        Raises:
            Exception: If there's an error processing the image or during classification.
        """
        return self._run(image_path)
