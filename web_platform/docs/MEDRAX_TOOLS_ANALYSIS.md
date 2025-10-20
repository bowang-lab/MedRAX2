# MedRAX Tools Complete Analysis

## Date: October 19, 2025
## Status: COMPREHENSIVE AUDIT COMPLETE

---

## EXECUTIVE SUMMARY

### Problem Identified:
1. ToolManager only registers 5 mock tools
2. Actual medrax folder contains 15+ real tools
3. All tools showing as "unavailable" 
4. No proper integration with actual MedRAX tools
5. Emoji characters present in logs (need removal per user request)

### Tools Found in MedRAX:
**Total: 15 distinct tools across 4 categories**

---

## COMPLETE TOOL INVENTORY

### Category 1: CLASSIFICATION (2 tools)

#### 1.1 TorchXRayVisionClassifierTool
- **File**: `medrax/tools/classification/torchxrayvision.py`
- **Class**: `TorchXRayVisionClassifierTool`
- **Name**: `torchxrayvision_classifier`
- **Description**: Classifies chest X-rays for 18 pathologies using DenseNet
- **Dependencies**: 
  - torch
  - torchvision
  - torchxrayvision
  - skimage
- **Requires GPU**: Yes
- **Input**: JPG or PNG image path
- **Output**: Dictionary of 18 pathology probabilities
- **Pathologies**: Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, 
  Emphysema, Enlarged Cardiomediastinum, Fibrosis, Fracture, Hernia, 
  Infiltration, Lung Lesion, Lung Opacity, Mass, Nodule, Pleural Thickening, 
  Pneumonia, Pneumothorax

#### 1.2 ArcPlusClassifierTool
- **File**: `medrax/tools/classification/arcplus.py`
- **Class**: `ArcPlusClassifierTool`
- **Name**: `arcplus_chest_xray_classifier`
- **Description**: Multi-head chest X-ray classifier for 19 diseases and 6 genders
- **Dependencies**:
  - torch
  - torchvision
  - timm (for SwinTransformer)
  - numpy
  - PIL
- **Requires GPU**: Yes
- **Input**: JPG or PNG image path
- **Output**: Disease probabilities and gender classification
- **Diseases**: 19 classes (more comprehensive than TorchXRayVision)
- **Special**: Also predicts patient gender

---

### Category 2: VISUAL QUESTION ANSWERING (3 tools)

#### 2.1 CheXagentXRayVQATool
- **File**: `medrax/tools/vqa/xray_vqa.py`
- **Class**: `CheXagentXRayVQATool`
- **Name**: `chexagent_xray_vqa`
- **Description**: Comprehensive chest X-ray analysis using CheXagent-2-3b
- **Dependencies**:
  - torch
  - transformers (AutoModelForCausalLM, AutoTokenizer)
- **Requires GPU**: Yes
- **Model**: StanfordAIMI/CheXagent-2-3b
- **Input**: List of image paths + natural language prompt
- **Output**: Detailed analysis text
- **Capabilities**:
  - Visual question answering
  - Report generation
  - Abnormality detection
  - Comparative analysis
  - Anatomical description
  - Clinical interpretation

#### 2.2 LlavaMedTool
- **File**: `medrax/tools/vqa/llava_med.py`
- **Class**: `LlavaMedTool`
- **Name**: `llava_med_qa`
- **Description**: Medical visual QA using LLaVA-Med
- **Dependencies**:
  - torch
  - medrax.llava (custom LLaVA implementation)
  - PIL
- **Requires GPU**: Yes
- **Model**: LLaVA-Med (custom medical fine-tune)
- **Input**: Question + optional image path
- **Output**: Answer to medical question
- **Note**: Can handle biomedical images and general medical questions

#### 2.3 MedGemmaAPIClientTool
- **File**: `medrax/tools/vqa/medgemma/medgemma_client.py`
- **Class**: `MedGemmaAPIClientTool`
- **Name**: `medgemma_vqa`
- **Description**: Medical VQA using MedGemma API
- **Dependencies**:
  - API client dependencies
  - Separate setup script: `medgemma_setup.py`
- **Requires GPU**: No (API-based)
- **Input**: Image + question
- **Output**: Medical answer from MedGemma

---

### Category 3: SEGMENTATION (2 tools)

#### 3.1 MedSAM2Tool
- **File**: `medrax/tools/segmentation/medsam2.py`
- **Class**: `MedSAM2Tool`
- **Name**: `medsam2_segmentation`
- **Description**: Advanced medical image segmentation using MedSAM2
- **Dependencies**:
  - torch
  - numpy
  - matplotlib
  - PIL
  - sam2 (Segment Anything Model 2)
  - huggingface_hub
  - hydra
- **Requires GPU**: Yes
- **Model**: Downloads from HuggingFace
- **Input**: Image path + prompt type (box/point/auto) + coordinates
- **Output**: Segmentation mask + visualization
- **Prompt Types**:
  - Box: [x1,y1,x2,y2]
  - Point: [x,y]
  - Auto: Automatic segmentation

#### 3.2 ChestXRaySegmentationTool
- **File**: `medrax/tools/segmentation/segmentation.py`
- **Class**: `ChestXRaySegmentationTool`
- **Name**: `chest_xray_segmentation`
- **Description**: Chest X-ray organ segmentation
- **Dependencies**:
  - torch
  - transformers
  - PIL
- **Requires GPU**: Yes
- **Output**: Organ masks + metrics (area, position, etc.)

---

### Category 4: REPORT GENERATION (1 tool)

#### 4.1 ChestXRayReportGeneratorTool
- **File**: `medrax/tools/report_generation.py`
- **Class**: `ChestXRayReportGeneratorTool`
- **Name**: `chest_xray_report_generator`
- **Description**: Generates comprehensive radiology reports
- **Dependencies**:
  - torch
  - transformers (ViT-BERT models)
  - PIL
- **Requires GPU**: Yes
- **Models**: Two separate models for findings and impressions
- **Output**: Structured report with:
  - Detailed findings
  - Clinical impression
  - Follows standard radiological format

---

### Category 5: GROUNDING (1 tool)

#### 5.1 XRayPhraseGroundingTool
- **File**: `medrax/tools/grounding.py`
- **Class**: `XRayPhraseGroundingTool`
- **Name**: `xray_phrase_grounding`
- **Description**: Locates medical findings in X-rays using MAIRA-2
- **Dependencies**:
  - torch
  - transformers (MAIRA-2 model)
  - matplotlib
  - PIL
- **Requires GPU**: Yes
- **Input**: Image path + medical phrase (e.g., "Pleural effusion")
- **Output**: 
  - Bounding box coordinates [x1,y1,x2,y2] (normalized 0-1)
  - Visualization
  - Confidence metadata

---

### Category 6: IMAGE PROCESSING (2 tools)

#### 6.1 DicomProcessorTool
- **File**: `medrax/tools/dicom.py`
- **Class**: `DicomProcessorTool`
- **Name**: `dicom_processor`
- **Description**: Processes DICOM files and converts to PNG
- **Dependencies**:
  - pydicom
  - numpy
  - PIL
- **Requires GPU**: No
- **Input**: DICOM file path + optional window/level parameters
- **Output**: PNG image path + DICOM metadata
- **Purpose**: Convert DICOM to formats other tools can use

#### 6.2 ChestXRayGeneratorTool
- **File**: `medrax/tools/xray_generation.py`
- **Class**: `ChestXRayGeneratorTool`
- **Name**: `chest_xray_generator`
- **Description**: Generates synthetic X-rays from text descriptions
- **Dependencies**:
  - torch
  - diffusers (Stable Diffusion)
- **Requires GPU**: Yes
- **Model**: Fine-tuned Stable Diffusion (Roentgen)
- **Input**: Text description of condition + generation parameters
- **Output**: Generated X-ray image path

---

### Category 7: RETRIEVAL (2 tools)

#### 7.1 RAGTool
- **File**: `medrax/tools/rag.py`
- **Class**: `RAGTool`
- **Name**: `medical_knowledge_rag`
- **Description**: Answers medical questions using RAG with knowledge base
- **Dependencies**:
  - langchain
  - medrax.rag (CohereRAG)
  - Vector database
- **Requires GPU**: No (but benefits from it)
- **Knowledge Base**:
  - Medical textbooks
  - Research papers
  - Clinical manuals
  - Guidelines
- **Output**: Evidence-based answers with sources

#### 7.2 DuckDuckGoSearchTool
- **File**: `medrax/tools/browsing/duckduckgo.py`
- **Class**: `DuckDuckGoSearchTool`
- **Name**: `duckduckgo_search`
- **Description**: Web search for medical information
- **Dependencies**:
  - duckduckgo_search (DDGS)
- **Requires GPU**: No
- **Input**: Search query + max_results + region
- **Output**: Search results with titles, snippets, URLs

---

### Category 8: WEB BROWSING (1 tool)

#### 8.1 WebBrowserTool
- **File**: `medrax/tools/browsing/web_browser.py`
- **Class**: `WebBrowserTool`
- **Name**: `web_browser`
- **Description**: Browse and extract content from web pages
- **Dependencies**:
  - Web scraping libraries
- **Requires GPU**: No
- **Capabilities**:
  - Visit URLs
  - Extract content
  - Navigate pages

---

### Category 9: CODE EXECUTION (1 tool)

#### 9.1 PyodideSandboxTool
- **File**: `medrax/tools/python_tool.py`
- **Function**: `create_python_sandbox()`
- **Name**: `python_sandbox`
- **Description**: Execute Python code in secure sandbox
- **Dependencies**:
  - langchain_sandbox
  - Deno runtime (must be installed on host)
- **Requires GPU**: No
- **Capabilities**:
  - Execute Python code
  - Install packages (micropip)
  - Maintain state between calls
  - Network access (configurable)
- **Pre-installed Packages**:
  - pandas
  - numpy
  - pydicom
  - SimpleITK
  - scikit-image

---

## TOTAL TOOL COUNT

| Category | Tools | GPU Required |
|----------|-------|--------------|
| Classification | 2 | Yes |
| VQA | 3 | 2 Yes, 1 No (API) |
| Segmentation | 2 | Yes |
| Report Generation | 1 | Yes |
| Grounding | 1 | Yes |
| Image Processing | 2 | 1 Yes, 1 No |
| Retrieval | 2 | No (but benefits) |
| Web Browsing | 1 | No |
| Code Execution | 1 | No |
| **TOTAL** | **15** | **11 Yes, 4 No** |

---

## CURRENT TOOL MANAGER ISSUES

### Issue 1: Only 5 Tools Registered
Current ToolManager only registers:
1. classification (generic)
2. vqa (generic)
3. segmentation (generic)
4. report_generation (generic)
5. browsing (generic)

**Missing**: 10+ actual specific tools

### Issue 2: Import Paths Wrong
Current imports try:
```python
from medrax.tools.classification import TorchXRayVisionClassifierTool
```

But should handle multiple tools per category:
```python
from medrax.tools.classification import TorchXRayVisionClassifierTool, ArcPlusClassifierTool
```

### Issue 3: Unavailable Status
All tools show "unavailable" because:
1. `medrax_available = False` due to import failures
2. Missing dependencies check is too broad
3. No individual tool availability check

### Issue 4: No Dependency Granularity
Current check:
- If torch/transformers missing -> ALL tools unavailable

Should be:
- Check each tool's specific dependencies
- Some tools don't need GPU (DuckDuckGo, DICOM, Python sandbox)
- RAG tool has different dependencies

---

## DEPENDENCY ANALYSIS

### Core Dependencies (for most tools):
- torch
- transformers
- torchvision
- PIL/Pillow
- numpy

### Specialized Dependencies:

**Classification:**
- torchxrayvision
- timm (for ArcPlus)
- skimage

**VQA:**
- medrax.llava (custom)
- AutoModelForCausalLM

**Segmentation:**
- sam2
- huggingface_hub
- hydra

**Image Processing:**
- pydicom (for DICOM)
- diffusers (for generation)

**Retrieval:**
- langchain
- duckduckgo_search
- Vector database (Pinecone)

**Code Execution:**
- langchain_sandbox
- Deno runtime

---

## EMOJI/NON-ASCII CHARACTERS FOUND

### In ToolManager (tool_manager.py):
- Line 85: `logger.info("✓ MedRAX tools are AVAILABLE")`
- Line 91: `logger.warning("⚠️  MedRAX tools NOT available")`
- Line 210: `logger.info("✓ Tool loaded: {tool.name}")`
- Line 287: `logger.info("✓ Tool unloaded: {tool.name}")`
- Line 347: `logger.info("✓ Agent created with {len(loaded_tools)} tools")`

### In Documentation Files:
- Multiple .md files with emojis throughout
- COMPLETE_STATUS.md
- AUTH_FIX_REPORT.md
- FINAL_IMPLEMENTATION_SUMMARY.md
- Others

**Action Required**: Replace all with ASCII equivalents

---

## RECOMMENDED FIXES

### Fix 1: Register All 15 Tools
Update `_register_tools()` to include all discovered tools with proper:
- Tool IDs
- Names
- Descriptions
- Dependencies
- GPU requirements

### Fix 2: Individual Dependency Checking
Create method to check each tool's specific dependencies:
```python
def _check_tool_dependencies(self, tool_id: str) -> bool:
    # Check specific dependencies for each tool
    # Return True if available, False otherwise
```

### Fix 3: Graceful Degradation
- Tools without GPU dependencies should be available even if GPU tools aren't
- DuckDuckGo, DICOM, Python sandbox can work without ML dependencies

### Fix 4: Remove All Emojis
Replace in tool_manager.py:
- "✓" -> "[OK]"
- "⚠️" -> "[WARNING]"
- "❌" -> "[ERROR]"

### Fix 5: Proper Import Error Handling
Catch ImportError for each tool individually
Don't let one tool failure prevent others from loading

---

## INTEGRATION STRATEGY

### Phase 1: Basic Tools (No Heavy ML)
1. DuckDuckGoSearchTool (just needs duckduckgo_search)
2. WebBrowserTool (basic web scraping)
3. DicomProcessorTool (just needs pydicom)
4. PyodideSandboxTool (needs Deno)

### Phase 2: Standard ML Tools
1. TorchXRayVisionClassifierTool
2. CheXagentXRayVQATool
3. ChestXRayReportGeneratorTool

### Phase 3: Advanced ML Tools
1. MedSAM2Tool
2. XRayPhraseGroundingTool
3. ArcPlusClassifierTool
4. LlavaMedTool
5. ChestXRaySegmentationTool

### Phase 4: Specialized Tools
1. ChestXRayGeneratorTool (needs specific model weights)
2. RAGTool (needs vector DB setup)
3. MedGemmaAPIClientTool (needs API setup)

---

## NEXT STEPS

1. Create new ToolManager with all 15 tools
2. Implement individual dependency checking
3. Remove all emoji characters
4. Add proper error messages for missing dependencies
5. Test each tool's availability detection
6. Update frontend to display all tools correctly
7. Add tool categories to UI
8. Implement phased loading based on dependencies

---

## CONCLUSION

The MedRAX tools folder contains 15 powerful medical imaging tools across 9 categories.
Current implementation only exposes 5 generic placeholders.
All tools show as "unavailable" due to overly broad dependency checking.
Proper integration requires individual tool registration, dependency checking, and graceful degradation.

**Status**: Analysis complete, ready for implementation
**Priority**: HIGH - Users cannot access actual MedRAX functionality
**Effort**: Medium - Requires careful refactoring but clear path forward

