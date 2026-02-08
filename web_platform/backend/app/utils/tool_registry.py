from __future__ import annotations

from typing import Any, Dict, List

TOOL_UI_REGISTRY: List[Dict[str, Any]] = [
    {
        "tool_name": "torchxrayvision_classifier",
        "display_name": "TorchXRayVision Classifier",
        "category": "classification",
        "panel": "TorchXRayVisionPanel",
        "description": "Classifies chest X-rays for 18 pathologies using TorchXRayVision DenseNet.",
        "input_schema": {
            "image_path": "string (required)",
        },
        "output_schema": {
            "Atelectasis": "float (0-1)",
            "Cardiomegaly": "float (0-1)",
            "...": "float (0-1)",
        },
        "notes": [
            "Output is a dict of pathology -> probability.",
            "Errors return {'error': '...'} in output.",
        ],
    },
    {
        "tool_name": "arcplus_classifier",
        "display_name": "ArcPlus Classifier",
        "category": "classification",
        "panel": "ArcPlusPanel",
        "description": "Multi-head chest X-ray classifier across multiple datasets.",
        "input_schema": {
            "image_path": "string (required)",
        },
        "output_schema": {
            "Atelectasis": "float (0-1)",
            "Cardiomegaly": "float (0-1)",
            "...": "float (0-1)",
        },
        "notes": [
            "Multiple dataset heads are concatenated; duplicate keys collapse to last value.",
            "Errors return {'error': '...'} in output.",
        ],
    },
    {
        "tool_name": "chest_xray_segmentation",
        "display_name": "Chest X-Ray Segmentation",
        "category": "segmentation",
        "panel": "ChestSegmentationPanel",
        "description": "Segments chest X-ray organs and returns metrics + overlay image.",
        "input_schema": {
            "image_path": "string (required)",
            "organs": "string[] (optional)",
        },
        "output_schema": {
            "segmentation_image_path": "string",
            "metrics": {"<organ>": "OrganMetrics"},
        },
        "notes": [
            "Organs are optional; if omitted, all supported organs are processed.",
            "Errors return {'error': '...'} in output.",
        ],
    },
    {
        "tool_name": "medsam2",
        "display_name": "MedSAM2 Segmentation",
        "category": "segmentation",
        "panel": "MedSAM2Panel",
        "description": "Prompted segmentation with box/point/auto prompts.",
        "input_schema": {
            "image_path": "string (required)",
            "prompt_type": "box | point | auto",
            "prompt_coords": "number[] (optional)",
            "slice_index": "number | null (optional)",
        },
        "output_schema": {
            "segmentation_image_path": "string",
            "confidence_scores": "number[]",
            "num_masks": "number",
            "best_mask_score": "number",
            "mask_summary": "object",
        },
        "notes": [
            "prompt_coords format depends on prompt_type.",
            "Errors return {'error': '...'} in output.",
        ],
    },
    {
        "tool_name": "chexagent_xray_vqa",
        "display_name": "CheXagent X-Ray VQA",
        "category": "vqa",
        "panel": "CheXagentVQAPanel",
        "description": "Visual Q&A for chest X-rays using CheXagent.",
        "input_schema": {
            "image_paths": "string[] (required)",
            "prompt": "string (required)",
            "max_new_tokens": "number (optional)",
        },
        "output_schema": {
            "response": "string",
        },
        "notes": [
            "Errors return {'error': '...'} in output.",
        ],
    },
    {
        "tool_name": "llava_med_qa",
        "display_name": "LLaVA-Med QA",
        "category": "vqa",
        "panel": "LlavaMedPanel",
        "description": "Medical QA with optional image input using LLaVA-Med.",
        "input_schema": {
            "question": "string (required)",
            "image_path": "string (optional)",
        },
        "output_schema": {
            "answer": "string",
        },
        "notes": [
            "Errors return {'error': '...'} in output.",
        ],
    },
    {
        "tool_name": "chest_xray_report_generator",
        "display_name": "Chest X-Ray Report Generator",
        "category": "generation",
        "panel": "ReportGeneratorPanel",
        "description": "Generates radiology findings + impression report.",
        "input_schema": {
            "image_path": "string (required)",
        },
        "output_schema": {
            "report": "string",
            "findings": "string",
            "impression": "string",
        },
        "notes": [
            "Errors return {'error': '...'} in output.",
        ],
    },
    {
        "tool_name": "xray_phrase_grounding",
        "display_name": "X-Ray Phrase Grounding",
        "category": "grounding",
        "panel": "PhraseGroundingPanel",
        "description": "Grounds a phrase to bounding boxes on the X-ray.",
        "input_schema": {
            "image_path": "string (required)",
            "phrase": "string (required)",
            "max_new_tokens": "number (optional)",
        },
        "output_schema": {
            "predictions": "object[]",
            "visualization_path": "string | null",
        },
        "notes": [
            "Errors return {'error': '...'} in output.",
        ],
    },
    {
        "tool_name": "chest_xray_generator",
        "display_name": "Chest X-Ray Generator",
        "category": "generation",
        "panel": "XRayGeneratorPanel",
        "description": "Generates synthetic chest X-rays from a text prompt.",
        "input_schema": {
            "prompt": "string (required)",
            "height": "number (optional)",
            "width": "number (optional)",
            "num_inference_steps": "number (optional)",
            "guidance_scale": "number (optional)",
        },
        "output_schema": {
            "image_path": "string",
        },
        "notes": [
            "Errors return {'error': '...'} in output.",
        ],
    },
    {
        "tool_name": "dicom_processor",
        "display_name": "DICOM Processor",
        "category": "processing",
        "panel": "DicomProcessorPanel",
        "description": "Converts DICOM to PNG and extracts metadata.",
        "input_schema": {
            "dicom_path": "string (required)",
            "window_center": "number (optional)",
            "window_width": "number (optional)",
        },
        "output_schema": {
            "image_path": "string",
        },
        "notes": [
            "Metadata includes DICOM fields like PatientID, StudyDate, Modality.",
            "Errors return {'error': '...'} in output.",
        ],
    },
    {
        "tool_name": "image_visualizer",
        "display_name": "Image Visualizer",
        "category": "utility",
        "panel": "ImageVisualizerPanel",
        "description": "Renders an image with optional title/description.",
        "input_schema": {
            "image_path": "string (required)",
            "title": "string | null (optional)",
            "description": "string | null (optional)",
            "width": "number (optional)",
            "height": "number (optional)",
            "cmap": "string (optional)",
        },
        "output_schema": {
            "image_path": "string",
        },
        "notes": [
            "Errors return {'error': '...'} in output.",
        ],
    },
    {
        "tool_name": "medical_knowledge_rag",
        "display_name": "Medical Knowledge RAG",
        "category": "retrieval",
        "panel": "RAGPanel",
        "description": "Answers medical questions using RAG with a knowledge base.",
        "input_schema": {
            "query": "string (required)",
        },
        "output_schema": {
            "answer": "string",
            "source_documents": "object[]",
        },
        "notes": [
            "Errors return {'error': '...'} in output.",
        ],
    },
    {
        "tool_name": "duckduckgo_search",
        "display_name": "DuckDuckGo Search",
        "category": "retrieval",
        "panel": "SearchPanel",
        "description": "Searches the web for medical information.",
        "input_schema": {
            "query": "string (required)",
            "max_results": "number (optional)",
            "region": "string (optional)",
        },
        "output_schema": {
            "results": "object[]",
            "summary": "string",
        },
        "notes": [
            "Errors return {'error': '...'} in output.",
        ],
    },
    {
        "tool_name": "web_browser",
        "display_name": "Web Browser",
        "category": "retrieval",
        "panel": "WebBrowserPanel",
        "description": "Fetches a URL or performs a web search.",
        "input_schema": {
            "query": "string (optional)",
            "url": "string (optional)",
            "max_content_length": "number (optional)",
            "max_links": "number (optional)",
        },
        "output_schema": {
            "title": "string",
            "content": "string",
            "url": "string",
            "links": "object[]",
        },
        "notes": [
            "Provide query OR url; url takes precedence if both provided.",
            "Errors return {'error': '...'} in output.",
        ],
    },
]

PREVIEW_FIELDS: Dict[str, Dict[str, Any]] = {
    "torchxrayvision_classifier": {
        "input_preview": {"image_paths": ["image_path"]},
        "output_preview": {"image_paths": []},
    },
    "arcplus_classifier": {
        "input_preview": {"image_paths": ["image_path"]},
        "output_preview": {"image_paths": []},
    },
    "chest_xray_segmentation": {
        "input_preview": {"image_paths": ["image_path"]},
        "output_preview": {"image_paths": ["segmentation_image_path"]},
    },
    "medsam2": {
        "input_preview": {"image_paths": ["image_path"]},
        "output_preview": {"image_paths": ["segmentation_image_path"]},
    },
    "chexagent_xray_vqa": {
        "input_preview": {"image_paths": ["image_paths"]},
        "output_preview": {"image_paths": []},
    },
    "llava_med_qa": {
        "input_preview": {"image_paths": ["image_path"]},
        "output_preview": {"image_paths": []},
    },
    "chest_xray_report_generator": {
        "input_preview": {"image_paths": ["image_path"]},
        "output_preview": {"image_paths": []},
    },
    "xray_phrase_grounding": {
        "input_preview": {"image_paths": ["image_path"]},
        "output_preview": {"image_paths": ["visualization_path"]},
    },
    "chest_xray_generator": {
        "input_preview": {"image_paths": []},
        "output_preview": {"image_paths": ["image_path"]},
    },
    "dicom_processor": {
        "input_preview": {"image_paths": ["dicom_path"]},
        "output_preview": {"image_paths": ["image_path"]},
    },
    "image_visualizer": {
        "input_preview": {"image_paths": ["image_path"]},
        "output_preview": {"image_paths": ["image_path"]},
    },
    "medical_knowledge_rag": {
        "input_preview": {"image_paths": []},
        "output_preview": {"image_paths": []},
    },
    "duckduckgo_search": {
        "input_preview": {"image_paths": []},
        "output_preview": {"image_paths": []},
    },
    "web_browser": {
        "input_preview": {"image_paths": []},
        "output_preview": {"image_paths": []},
    },
}


def get_tool_registry() -> List[Dict[str, Any]]:
    tools: List[Dict[str, Any]] = []
    for tool in TOOL_UI_REGISTRY:
        preview = PREVIEW_FIELDS.get(
            tool["tool_name"],
            {"input_preview": {"image_paths": []}, "output_preview": {"image_paths": []}},
        )
        tools.append({**tool, **preview})
    return tools


def get_tool_registry_map() -> Dict[str, Dict[str, Any]]:
    return {tool["tool_name"]: tool for tool in get_tool_registry()}
