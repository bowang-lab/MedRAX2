"""
Tool Testing API Routes

Direct endpoints for testing individual tools with their exact inputs/outputs.
No authentication required - for local development only.

Access the interactive docs at: http://localhost:8000/docs#/tool-testing
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from pathlib import Path
import tempfile
import shutil
import json
import traceback
import numpy as np

from ..services.tool_manager import tool_manager
from ..utils.logging_config import logger

router = APIRouter()

# Get backend directory for resolving relative paths
BACKEND_DIR = Path(__file__).parent.parent.parent

def resolve_image_path(path: str) -> str:
    """Convert relative paths to absolute paths."""
    if not Path(path).is_absolute():
        # Try relative to backend directory first
        abs_path = BACKEND_DIR / path
        if abs_path.exists():
            return str(abs_path)
        # Try relative to current working directory
        cwd_path = Path.cwd() / path
        if cwd_path.exists():
            return str(cwd_path)
    return path


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


# ============================================================================
# CLASSIFICATION TOOLS
# ============================================================================

class TorchXRayVisionInput(BaseModel):
    """Input for TorchXRayVision classifier"""
    image_path: str = Field(..., description="Path to chest X-ray image", example="/uploads/test.jpg")

@router.post("/torchxrayvision", 
    summary="Test TorchXRayVision Classifier",
    description="Classify chest X-ray for 18 pathologies using DenseNet",
    tags=["classification"])
async def test_torchxrayvision(input_data: TorchXRayVisionInput):
    """Test TorchXRayVision classifier tool directly."""
    try:
        tool = tool_manager.get_tool("torchxrayvision")
        if not tool or tool.status != "loaded":
            raise HTTPException(status_code=503, detail="TorchXRayVision tool not loaded")
        
        result = tool.instance._run(resolve_image_path(input_data.image_path))
        # Convert numpy types to native Python types
        converted_result = convert_numpy_types(result[0])
        converted_metadata = convert_numpy_types(result[1])
        return {"success": True, "result": converted_result, "metadata": converted_metadata}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TorchXRayVision test error: {e}\n{traceback.format_exc()}")
        return {"success": False, "error": str(e)}


class ArcPlusInput(BaseModel):
    """Input for ArcPlus classifier"""
    image_path: str = Field(..., description="Path to chest X-ray image")

@router.post("/arcplus",
    summary="Test ArcPlus Classifier", 
    description="Multi-head classifier for 19 diseases using Swin Transformer",
    tags=["classification"])
async def test_arcplus(input_data: ArcPlusInput):
    """Test ArcPlus classifier tool directly."""
    try:
        tool = tool_manager.get_tool("arcplus")
        if not tool or tool.status != "loaded":
            raise HTTPException(status_code=503, detail="ArcPlus tool not loaded")
        
        result = tool.instance._run(resolve_image_path(input_data.image_path))
        # Convert numpy types to native Python types
        converted_result = convert_numpy_types(result[0])
        converted_metadata = convert_numpy_types(result[1])
        return {"success": True, "result": converted_result, "metadata": converted_metadata}
    except Exception as e:
        logger.error(f"ArcPlus test error: {e}\n{traceback.format_exc()}")
        return {"success": False, "error": str(e)}


# ============================================================================
# SEGMENTATION TOOLS
# ============================================================================

class ChestSegmentationInput(BaseModel):
    """Input for Chest X-ray Segmentation"""
    image_path: str = Field(..., description="Path to chest X-ray image")
    organs: List[str] = Field(
        default=["Left Lung", "Right Lung"],
        description="Organs to segment. Options: Left/Right Lung, Heart, etc."
    )
    threshold: float = Field(default=0.3, description="Detection threshold (0.1-0.5)")
    debug: bool = Field(default=False, description="Return debug info including raw predictions")

@router.post("/chest_segmentation",
    summary="Test Chest X-ray Segmentation",
    description="Segment anatomical structures in chest X-rays",
    tags=["segmentation"])
async def test_chest_segmentation(input_data: ChestSegmentationInput):
    """Test chest segmentation tool directly."""
    try:
        tool = tool_manager.get_tool("chest_segmentation")
        if not tool or tool.status != "loaded":
            raise HTTPException(status_code=503, detail="Chest segmentation tool not loaded")
        
        # Set threshold if tool supports it
        if hasattr(tool.instance, 'threshold'):
            tool.instance.threshold = input_data.threshold
            
        result = tool.instance._run(resolve_image_path(input_data.image_path), input_data.organs)
        # Convert numpy types to native Python types
        converted_result = convert_numpy_types(result[0])
        converted_metadata = convert_numpy_types(result[1])
        
        # Add debug info if requested
        if input_data.debug:
            # Run again with very low threshold to see what's being detected
            import torch
            import skimage.io
            import numpy as np
            
            # Get raw predictions
            original_img = skimage.io.imread(input_data.image_path)
            if len(original_img.shape) > 2:
                original_img = original_img[:, :, 0]
            
            from medrax.utils.utils import preprocess_medical_image
            img = preprocess_medical_image(original_img)
            img = img[None, ...]
            img = tool.instance.image_transform(img)
            img = torch.from_numpy(img).float().to(tool.instance.device)
            
            with torch.no_grad():
                pred = tool.instance.model(img)
            pred_probs = torch.sigmoid(pred)
            
            # Get probabilities for requested organs
            debug_info = {}
            for organ in input_data.organs:
                if organ in tool.instance.organ_map:
                    idx = tool.instance.organ_map[organ]
                    max_prob = float(pred_probs[0, idx].max().cpu())
                    mean_prob = float(pred_probs[0, idx].mean().cpu())
                    debug_info[organ] = {
                        "max_probability": max_prob,
                        "mean_probability": mean_prob,
                        "detected_at_threshold": max_prob > input_data.threshold
                    }
            
            converted_result["debug"] = debug_info
        
        return {"success": True, "result": converted_result, "metadata": converted_metadata}
    except Exception as e:
        logger.error(f"Chest segmentation test error: {e}\n{traceback.format_exc()}")
        return {"success": False, "error": str(e)}


class MedSAM2Input(BaseModel):
    """Input for MedSAM2 segmentation"""
    image_path: str = Field(..., description="Path to medical image")
    prompt_type: str = Field("box", description="Type: 'box', 'point', or 'auto'")
    prompt_coords: List[int] = Field(
        default=[100, 100, 200, 200],
        description="[x1,y1,x2,y2] for box or [x,y] for point"
    )

@router.post("/medsam2",
    summary="Test MedSAM2 Segmentation",
    description="Advanced medical image segmentation using SAM2",
    tags=["segmentation"])
async def test_medsam2(input_data: MedSAM2Input):
    """Test MedSAM2 segmentation tool directly."""
    try:
        tool = tool_manager.get_tool("medsam2")
        if not tool or tool.status != "loaded":
            raise HTTPException(status_code=503, detail="MedSAM2 tool not loaded")
        
        result = tool.instance._run(
            resolve_image_path(input_data.image_path),
            input_data.prompt_type,
            input_data.prompt_coords
        )
        # Convert numpy types to native Python types
        converted_result = convert_numpy_types(result[0])
        converted_metadata = convert_numpy_types(result[1])
        return {"success": True, "result": converted_result, "metadata": converted_metadata}
    except Exception as e:
        logger.error(f"MedSAM2 test error: {e}\n{traceback.format_exc()}")
        return {"success": False, "error": str(e)}


@router.post("/chest_segmentation_scan",
    summary="Scan All Organs",
    description="Test what organs are actually detected in an image",
    tags=["segmentation"])
async def scan_chest_organs(image_path: str = Form(...)):
    """Scan all organs to see what's detected."""
    try:
        tool = tool_manager.get_tool("chest_segmentation")
        if not tool or tool.status != "loaded":
            raise HTTPException(status_code=503, detail="Chest segmentation tool not loaded")
        
        import torch
        import skimage.io
        
        # Load and process image
        original_img = skimage.io.imread(image_path)
        if len(original_img.shape) > 2:
            original_img = original_img[:, :, 0]
        
        from medrax.utils.utils import preprocess_medical_image
        img = preprocess_medical_image(original_img)
        img = img[None, ...]
        img = tool.instance.image_transform(img)
        img = torch.from_numpy(img).float().to(tool.instance.device)
        
        # Get predictions
        with torch.no_grad():
            pred = tool.instance.model(img)
        pred_probs = torch.sigmoid(pred)
        
        # Check all organs
        organ_scores = {}
        for organ_name, organ_idx in tool.instance.organ_map.items():
            mask_probs = pred_probs[0, organ_idx].cpu()
            max_prob = float(mask_probs.max())
            mean_prob = float(mask_probs.mean())
            pixels_above_thresholds = {
                "0.1": int((mask_probs > 0.1).sum()),
                "0.2": int((mask_probs > 0.2).sum()),
                "0.3": int((mask_probs > 0.3).sum()),
                "0.4": int((mask_probs > 0.4).sum()),
                "0.5": int((mask_probs > 0.5).sum()),
            }
            
            organ_scores[organ_name] = {
                "max_probability": max_prob,
                "mean_probability": mean_prob,
                "pixels_above_thresholds": pixels_above_thresholds
            }
        
        # Sort by max probability
        sorted_organs = sorted(organ_scores.items(), key=lambda x: x[1]["max_probability"], reverse=True)
        
        return {
            "success": True,
            "image_shape": list(original_img.shape),
            "organs_by_confidence": dict(sorted_organs),
            "recommendation": f"Best detected organs: {', '.join([o[0] for o in sorted_organs[:3]])}"
        }
    except Exception as e:
        logger.error(f"Organ scan error: {e}\n{traceback.format_exc()}")
        return {"success": False, "error": str(e)}


@router.post("/chest_segmentation_diagnose",
    summary="Diagnose Segmentation Issues",
    description="Diagnose why segmentation isn't working on an image",
    tags=["segmentation"])
async def diagnose_segmentation(file: UploadFile = File(...)):
    """Diagnose segmentation issues with detailed analysis."""
    try:
        tool = tool_manager.get_tool("chest_segmentation")
        if not tool or tool.status != "loaded":
            raise HTTPException(status_code=503, detail="Chest segmentation tool not loaded")
        
        # Save uploaded file
        temp_path = Path("temp/test_uploads") / f"diagnose_{file.filename}"
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        import torch
        import skimage.io
        import numpy as np
        from PIL import Image
        
        # Load image multiple ways to check
        original_img = skimage.io.imread(str(temp_path))
        pil_img = Image.open(str(temp_path))
        
        # Get image stats
        img_stats = {
            "shape": original_img.shape,
            "dtype": str(original_img.dtype),
            "min_value": float(original_img.min()),
            "max_value": float(original_img.max()),
            "mean_value": float(original_img.mean()),
            "std_value": float(original_img.std()),
            "pil_mode": pil_img.mode,
            "pil_size": pil_img.size,
        }
        
        # Process through the pipeline
        if len(original_img.shape) > 2:
            original_img = original_img[:, :, 0]
            
        from medrax.utils.utils import preprocess_medical_image
        preprocessed = preprocess_medical_image(original_img)
        
        # Check preprocessed stats
        preprocess_stats = {
            "shape": preprocessed.shape,
            "dtype": str(preprocessed.dtype),
            "min_value": float(preprocessed.min()),
            "max_value": float(preprocessed.max()),
            "mean_value": float(preprocessed.mean()),
            "std_value": float(preprocessed.std()),
        }
        
        # Transform and check
        img = preprocessed[None, ...]
        transformed = tool.instance.image_transform(img)
        
        transform_stats = {
            "shape": transformed.shape,
            "dtype": str(transformed.dtype),
            "min_value": float(transformed.min()),
            "max_value": float(transformed.max()),
            "mean_value": float(transformed.mean()),
            "std_value": float(transformed.std()),
        }
        
        # Run through model
        img_tensor = torch.from_numpy(transformed).float().to(tool.instance.device)
        
        with torch.no_grad():
            pred = tool.instance.model(img_tensor)
        
        # Check raw predictions (before sigmoid)
        raw_pred_stats = {
            "shape": list(pred.shape),
            "min_value": float(pred.min().cpu()),
            "max_value": float(pred.max().cpu()),
            "mean_value": float(pred.mean().cpu()),
            "std_value": float(pred.std().cpu()),
        }
        
        # Apply sigmoid
        pred_probs = torch.sigmoid(pred)
        
        # Get stats per organ
        organ_analysis = {}
        for organ_name, organ_idx in tool.instance.organ_map.items():
            raw_vals = pred[0, organ_idx].cpu()
            prob_vals = pred_probs[0, organ_idx].cpu()
            
            organ_analysis[organ_name] = {
                "raw_min": float(raw_vals.min()),
                "raw_max": float(raw_vals.max()),
                "raw_mean": float(raw_vals.mean()),
                "prob_max": float(prob_vals.max()),
                "prob_mean": float(prob_vals.mean()),
            }
        
        # Recommendations
        recommendations = []
        
        if img_stats["max_value"] > 255:
            recommendations.append("Image appears to be 16-bit. Consider normalizing to 0-255 range.")
        
        if img_stats["mean_value"] < 10 or img_stats["mean_value"] > 245:
            recommendations.append("Image has extreme brightness. Check if it's inverted or needs adjustment.")
        
        if raw_pred_stats["mean_value"] < -5:
            recommendations.append("Model outputs are very negative. This suggests the model isn't recognizing the image format.")
        
        if all(organ_analysis[o]["prob_max"] < 0.1 for o in ["Left Lung", "Right Lung"]):
            recommendations.append("Lung detection is failing. Try MedSAM2 instead, or check if image is a standard PA/AP chest X-ray.")
        
        # Clean up temp file
        temp_path.unlink()
        
        return {
            "success": True,
            "diagnosis": {
                "original_image": img_stats,
                "after_preprocessing": preprocess_stats,
                "after_transform": transform_stats,
                "raw_predictions": raw_pred_stats,
                "organ_analysis": organ_analysis,
            },
            "recommendations": recommendations,
            "suggestion": "The PSPNet model seems incompatible with this image. Try MedSAM2 for better results."
        }
        
    except Exception as e:
        logger.error(f"Diagnosis error: {e}\n{traceback.format_exc()}")
        return {"success": False, "error": str(e)}


# ============================================================================
# VQA TOOLS
# ============================================================================

class VQAInput(BaseModel):
    """Input for VQA tools"""
    image_path: str = Field(..., description="Path to medical image")
    question: str = Field(..., description="Question about the image", 
                         example="What abnormalities are visible?")

@router.post("/chexagent",
    summary="Test CheXagent VQA",
    description="Visual question answering using CheXagent-2-3b",
    tags=["vqa"])
async def test_chexagent(input_data: VQAInput):
    """Test CheXagent VQA tool directly."""
    try:
        tool = tool_manager.get_tool("chexagent")
        if not tool or tool.status != "loaded":
            raise HTTPException(status_code=503, detail="CheXagent tool not loaded")
        
        # Convert to absolute path if relative
        image_path = resolve_image_path(input_data.image_path)
        
        # CheXagent now uses single image path like other tools
        result = tool.instance._run(
            image_path=image_path,
            prompt=input_data.question,
            max_new_tokens=512
        )
        # Convert numpy types to native Python types
        converted_result = convert_numpy_types(result[0])
        converted_metadata = convert_numpy_types(result[1])
        return {"success": True, "result": converted_result, "metadata": converted_metadata}
    except Exception as e:
        logger.error(f"CheXagent test error: {e}\n{traceback.format_exc()}")
        return {"success": False, "error": str(e)}


@router.post("/medgemma",
    summary="Test MedGemma VQA",
    description="Medical VQA using MedGemma 4B model",
    tags=["vqa"])
async def test_medgemma(input_data: VQAInput):
    """Test MedGemma VQA tool directly."""
    try:
        tool = tool_manager.get_tool("medgemma")
        if not tool or tool.status != "loaded":
            raise HTTPException(status_code=503, detail="MedGemma tool not loaded")
        
        result = tool.instance._run(resolve_image_path(input_data.image_path), input_data.question)
        # Convert numpy types to native Python types
        converted_result = convert_numpy_types(result[0])
        converted_metadata = convert_numpy_types(result[1])
        return {"success": True, "result": converted_result, "metadata": converted_metadata}
    except Exception as e:
        logger.error(f"MedGemma test error: {e}\n{traceback.format_exc()}")
        return {"success": False, "error": str(e)}


# ============================================================================
# REPORT GENERATION
# ============================================================================

class ReportGeneratorInput(BaseModel):
    """Input for report generation"""
    image_path: str = Field(..., description="Path to chest X-ray image")

@router.post("/report_generator",
    summary="Test Report Generator",
    description="Generate radiology reports from chest X-rays",
    tags=["generation"])
async def test_report_generator(input_data: ReportGeneratorInput):
    """Test report generator tool directly."""
    try:
        tool = tool_manager.get_tool("report_generator")
        if not tool or tool.status != "loaded":
            raise HTTPException(status_code=503, detail="Report generator tool not loaded")
        
        result = tool.instance._run(resolve_image_path(input_data.image_path))
        # Convert numpy types to native Python types
        converted_result = convert_numpy_types(result[0])
        converted_metadata = convert_numpy_types(result[1])
        return {"success": True, "result": converted_result, "metadata": converted_metadata}
    except Exception as e:
        logger.error(f"Report generator test error: {e}\n{traceback.format_exc()}")
        return {"success": False, "error": str(e)}


# ============================================================================
# GROUNDING
# ============================================================================

class GroundingInput(BaseModel):
    """Input for phrase grounding"""
    image_path: str = Field(..., description="Path to chest X-ray image")
    phrase: str = Field(..., description="Medical finding to locate", 
                       example="enlarged heart")

@router.post("/phrase_grounding",
    summary="Test Phrase Grounding",
    description="Locate medical findings in X-rays using MAIRA-2",
    tags=["grounding"])
async def test_phrase_grounding(input_data: GroundingInput):
    """Test phrase grounding tool directly."""
    try:
        tool = tool_manager.get_tool("phrase_grounding")
        if not tool or tool.status != "loaded":
            raise HTTPException(status_code=503, detail="Phrase grounding tool not loaded")
        
        result = tool.instance._run(resolve_image_path(input_data.image_path), input_data.phrase)
        # Convert numpy types to native Python types
        converted_result = convert_numpy_types(result[0])
        converted_metadata = convert_numpy_types(result[1])
        return {"success": True, "result": converted_result, "metadata": converted_metadata}
    except Exception as e:
        logger.error(f"Phrase grounding test error: {e}\n{traceback.format_exc()}")
        return {"success": False, "error": str(e)}


# ============================================================================
# IMAGE GENERATION
# ============================================================================

class XRayGeneratorInput(BaseModel):
    """Input for X-ray generation"""
    prompt: str = Field(..., description="Text description of pathology",
                       example="chest x-ray showing pneumonia in right lung")

@router.post("/xray_generator",
    summary="Test X-Ray Generator",
    description="Generate synthetic chest X-rays from text",
    tags=["generation"])
async def test_xray_generator(input_data: XRayGeneratorInput):
    """Test X-ray generator tool directly."""
    try:
        tool = tool_manager.get_tool("xray_generator")
        if not tool or tool.status != "loaded":
            raise HTTPException(status_code=503, detail="X-ray generator tool not loaded")
        
        result = tool.instance._run(input_data.prompt)
        # Convert numpy types to native Python types
        converted_result = convert_numpy_types(result[0])
        converted_metadata = convert_numpy_types(result[1])
        return {"success": True, "result": converted_result, "metadata": converted_metadata}
    except Exception as e:
        logger.error(f"X-ray generator test error: {e}\n{traceback.format_exc()}")
        return {"success": False, "error": str(e)}


# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@router.post("/upload",
    summary="Upload Test Image",
    description="Upload an image for testing (returns path to use in other endpoints)",
    tags=["utility"])
async def upload_test_image(file: UploadFile = File(...)):
    """Upload a test image and get its path."""
    try:
        # Save to temp directory
        temp_dir = Path("temp/test_uploads")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = temp_dir / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {
            "success": True,
            "file_path": str(file_path),
            "message": f"File uploaded to {file_path}"
        }
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/status",
    summary="Tool Status",
    description="Get status of all tools",
    tags=["utility"])
async def get_tool_status():
    """Get current status of all tools."""
    tools_status = {}
    for tool_id, tool in tool_manager.tools.items():
        tools_status[tool_id] = {
            "name": tool.name,
            "status": tool.status,
            "category": tool.category,
            "requires_gpu": tool.requires_gpu,
            "error": tool.error_message
        }
    
    loaded_count = sum(1 for t in tool_manager.tools.values() if t.status == "loaded")
    
    return {
        "total_tools": len(tool_manager.tools),
        "loaded_tools": loaded_count,
        "tools": tools_status
    }


@router.get("/download/{file_path:path}",
    summary="Download Result",
    description="Download generated images or files",
    tags=["utility"])
async def download_result(file_path: str):
    """Download a generated file (e.g., segmentation mask)."""
    try:
        file = Path(file_path)
        if not file.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            path=str(file),
            filename=file.name,
            media_type="application/octet-stream"
        )
    except Exception as e:
        logger.error(f"Download error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# BATCH TESTING
# ============================================================================

class BatchTestInput(BaseModel):
    """Input for batch testing multiple tools"""
    image_path: str = Field(..., description="Path to test image")
    tools: List[str] = Field(..., description="Tool IDs to test",
                            example=["torchxrayvision", "chest_segmentation"])

@router.post("/batch",
    summary="Batch Test Tools",
    description="Test multiple tools on the same image",
    tags=["utility"])
async def batch_test_tools(input_data: BatchTestInput):
    """Test multiple tools on the same image."""
    results = {}
    
    for tool_id in input_data.tools:
        try:
            tool = tool_manager.get_tool(tool_id)
            if not tool or tool.status != "loaded":
                results[tool_id] = {"error": f"Tool {tool_id} not loaded"}
                continue
            
            # Call appropriate method based on tool type
            image_path = resolve_image_path(input_data.image_path)
            if tool_id in ["torchxrayvision", "arcplus", "report_generator"]:
                result = tool.instance._run(image_path)
            elif tool_id == "chest_segmentation":
                result = tool.instance._run(image_path, ["Left Lung", "Right Lung"])
            elif tool_id == "medsam2":
                result = tool.instance._run(image_path, "auto", [])
            elif tool_id == "chexagent":
                result = tool.instance._run(image_path, "What abnormalities are visible?")
            elif tool_id == "medgemma":
                result = tool.instance._run(image_path, "What abnormalities are visible?")
            elif tool_id == "phrase_grounding":
                result = tool.instance._run(image_path, "opacity")
            else:
                results[tool_id] = {"error": f"Tool {tool_id} not configured for batch testing"}
                continue
            
            results[tool_id] = {
                "success": True,
                "result": convert_numpy_types(result[0]),
                "metadata": convert_numpy_types(result[1])
            }
        except Exception as e:
            logger.error(f"Batch test error for {tool_id}: {e}")
            results[tool_id] = {"error": str(e)}
    
    return results