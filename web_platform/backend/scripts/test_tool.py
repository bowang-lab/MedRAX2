#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import numpy as np
import os
import sys
from dotenv import load_dotenv
from pathlib import Path
from typing import Any, Dict, Tuple

load_dotenv()  # Load environment variables from .env file if present

# Ensure repo root is on sys.path for medrax imports
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def resolve_path(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    # Try relative to repo root
    candidate = REPO_ROOT / path
    if candidate.exists():
        return str(candidate)
    return str(p)


def run_tool(tool_id: str, args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if tool_id == "torchxrayvision_classifier":
        from medrax.tools.classification.torchxrayvision import TorchXRayVisionClassifierTool
        return TorchXRayVisionClassifierTool()._run(resolve_path(args.image_path))
    if tool_id == "arcplus_classifier":
        from medrax.tools.classification.arcplus import ArcPlusClassifierTool
        return ArcPlusClassifierTool(cache_dir=os.getenv("MODELWEIGHTS"))._run(resolve_path(args.image_path))
    if tool_id == "chest_xray_segmentation":
        from medrax.tools.segmentation.segmentation import ChestXRaySegmentationTool
        organs = args.organs or None
        return ChestXRaySegmentationTool()._run(resolve_path(args.image_path), organs)
    if tool_id == "medsam2":
        from medrax.tools.segmentation.medsam2 import MedSAM2Tool
        prompt_coords = None
        if args.prompt_coords:
            prompt_coords = [int(v) for v in args.prompt_coords.split(",")]
        return MedSAM2Tool(cache_dir=os.getenv("MODEL_CACHE_DIR"))._run(
            resolve_path(args.image_path),
            args.prompt_type,
            prompt_coords,
            args.slice_index,
        )
    if tool_id == "chexagent_xray_vqa":
        from medrax.tools.vqa.xray_vqa import CheXagentXRayVQATool
        return CheXagentXRayVQATool(cache_dir=os.getenv("MODEL_CACHE_DIR"))._run(
            [resolve_path(args.image_path)], args.prompt
        )
    if tool_id == "llava_med_qa":
        from medrax.tools.vqa.llava_med import LlavaMedTool
        image_path = resolve_path(args.image_path) if args.image_path else None
        return LlavaMedTool(cache_dir=os.getenv("MODEL_CACHE_DIR"))._run(args.question, image_path)
    if tool_id == "chest_xray_report_generator":
        from medrax.tools.report_generation import ChestXRayReportGeneratorTool
        return ChestXRayReportGeneratorTool(cache_dir=os.getenv("MODEL_CACHE_DIR"))._run(resolve_path(args.image_path))
    if tool_id == "xray_phrase_grounding":
        from medrax.tools.grounding import XRayPhraseGroundingTool
        return XRayPhraseGroundingTool(cache_dir=os.getenv("MODEL_CACHE_DIR"), load_in_8bit=True)._run(
            resolve_path(args.image_path), args.phrase
        )
    if tool_id == "dicom_processor":
        from medrax.tools.dicom import DicomProcessorTool
        return DicomProcessorTool()._run(resolve_path(args.dicom_path))
    if tool_id == "image_visualizer":
        from medrax.tools.utils import ImageVisualizerTool
        return ImageVisualizerTool()._run(resolve_path(args.image_path))
    if tool_id == "chest_xray_generator":
        from medrax.tools.xray_generation import ChestXRayGeneratorTool
        return ChestXRayGeneratorTool(cache_dir=os.getenv("MODEL_CACHE_DIR"))._run(args.prompt)
    if tool_id == "medical_knowledge_rag":
        from medrax.tools.rag import RAGTool, RAGConfig
        config = RAGConfig()
        return RAGTool(config=config)._run(args.query)
    if tool_id == "duckduckgo_search":
        from medrax.tools.browsing.duckduckgo import DuckDuckGoSearchTool
        return DuckDuckGoSearchTool()._run(args.query)
    if tool_id == "web_browser":
        from medrax.tools.browsing.web_browser import WebBrowserTool
        if args.url:
            return WebBrowserTool()._run(url=args.url)
        return WebBrowserTool()._run(query=args.query)
    raise ValueError(f"Unknown tool_id: {tool_id}")


def normalize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: normalize_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_json(v) for v in value]
    if isinstance(value, tuple):
        return [normalize_json(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    # Handle pydicom MultiValue and other non-serializable types
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
        try:
            return [normalize_json(v) for v in list(value)]
        except Exception:
            pass
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a single MedRAX tool with demo inputs.")
    parser.add_argument("--tool", required=True, help="Tool ID to run")

    # Common optional args
    parser.add_argument("--image-path", help="Path to image")
    parser.add_argument("--dicom-path", help="Path to DICOM file")
    parser.add_argument("--prompt", help="Prompt or question")
    parser.add_argument("--question", help="Question (LLaVA-Med)")
    parser.add_argument("--phrase", help="Phrase for grounding")
    parser.add_argument("--query", help="Query for RAG/search")
    parser.add_argument("--url", help="URL for web_browser")
    parser.add_argument("--organs", nargs="*", help="Organs list for segmentation")
    parser.add_argument("--prompt-type", default="auto", help="medsam2 prompt type: box|point|auto")
    parser.add_argument("--prompt-coords", help="medsam2 coords, comma-separated (e.g. 10,10,100,100)")
    parser.add_argument("--slice-index", type=int, default=None, help="medsam2 slice index")
    parser.add_argument("--json", action="store_true", help="Print JSON output only")

    args = parser.parse_args()
    tool_id = args.tool

    output, metadata = run_tool(tool_id, args)
    result = {"output": normalize_json(output), "metadata": normalize_json(metadata)}

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Tool: {tool_id}")
        print(json.dumps(result, indent=2))

    # Return non-zero if tool returned an error
    if isinstance(output, dict) and "error" in output:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
