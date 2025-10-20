"""Test that all MedRAX tools can be imported without errors"""
import sys
import importlib
from pathlib import Path

# Add medrax to path
medrax_path = Path(__file__).parent.parent.parent / "medrax"
sys.path.insert(0, str(medrax_path.parent))

print("=" * 80)
print("TESTING ALL MEDRAX TOOL IMPORTS")
print("=" * 80)

tools_to_test = [
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

passed = []
failed = []

for module_path, class_name in tools_to_test:
    try:
        module = importlib.import_module(module_path)
        tool_class = getattr(module, class_name)
        passed.append(f"✅ {class_name}")
        print(f"✅ {module_path}.{class_name}")
    except Exception as e:
        failed.append(f"❌ {class_name}: {str(e)[:100]}")
        print(f"❌ {module_path}.{class_name}")
        print(f"   Error: {str(e)[:200]}")

print("\n" + "=" * 80)
print(f"RESULTS: {len(passed)}/{len(tools_to_test)} tools imported successfully")
print("=" * 80)

if failed:
    print("\n❌ FAILED IMPORTS:")
    for f in failed:
        print(f"  {f}")
    sys.exit(1)
else:
    print("\n✅ ALL TOOLS IMPORTED SUCCESSFULLY!")
    sys.exit(0)
