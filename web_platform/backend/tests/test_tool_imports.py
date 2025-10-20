"""Test that all MedRAX tools can be imported without errors"""
import sys
import importlib
from pathlib import Path
import pytest

# Add medrax to path
medrax_path = Path(__file__).parent.parent.parent.parent / "medrax"
sys.path.insert(0, str(medrax_path.parent))


TOOLS_TO_TEST = [
    ("medrax.tools.classification.torchxrayvision", "TorchXRayVisionClassifierTool"),
    ("medrax.tools.classification.arcplus", "ArcPlusClassifierTool"),
    ("medrax.tools.vqa.xray_vqa", "CheXagentXRayVQATool"),
    ("medrax.tools.vqa.llava_med", "LlavaMedTool"),
    ("medrax.tools.vqa.medgemma.medgemma_client", "MedGemmaAPIClientTool"),
    ("medrax.tools.segmentation.medsam2", "MedSAM2Tool"),
    ("medrax.tools.segmentation.segmentation", "ChestXRaySegmentationTool"),
    ("medrax.tools.report_generation", "ChestXRayReportGeneratorTool"),
    ("medrax.tools.xray_generation", "ChestXRayGeneratorTool"),
    ("medrax.tools.grounding", "XRayPhraseGroundingTool"),
    ("medrax.tools.dicom", "DicomProcessorTool"),
    ("medrax.tools.rag", "RAGTool"),
    ("medrax.tools.browsing.duckduckgo", "DuckDuckGoSearchTool"),
    ("medrax.tools.browsing.web_browser", "WebBrowserTool"),
    ("medrax.tools.python_tool", "create_python_sandbox"),
]


@pytest.mark.parametrize("module_path,class_name", TOOLS_TO_TEST)
def test_tool_import(module_path, class_name):
    """Test that each tool can be imported without errors"""
    module = importlib.import_module(module_path)
    tool_class = getattr(module, class_name)
    assert tool_class is not None, f"{class_name} not found in {module_path}"


def test_all_tools_import():
    """Test that all 15 tools can be imported successfully"""
    passed = []
    failed = []
    
    for module_path, class_name in TOOLS_TO_TEST:
        try:
            module = importlib.import_module(module_path)
            tool_class = getattr(module, class_name)
            passed.append(f"✅ {class_name}")
        except Exception as e:
            failed.append(f"❌ {class_name}: {str(e)[:100]}")
    
    assert len(failed) == 0, f"Failed to import {len(failed)}/15 tools: {failed}"
    assert len(passed) == 15, f"Expected 15 tools, got {len(passed)}"
