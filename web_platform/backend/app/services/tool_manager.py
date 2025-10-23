"""
Tool Manager Service

Handles optional loading/unloading of MedRAX tools.
Provides graceful degradation when tools are not available.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..utils.logging_config import logger

# Set PyTorch environment for better compatibility
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # Enable MPS fallback to CPU if needed
os.environ['TORCH_HOME'] = os.path.expanduser('~/.cache/torch')  # Set cache location


class ToolStatus:
    """Tool status constants."""
    AVAILABLE = "available"  # Tool can be loaded
    LOADED = "loaded"        # Tool is currently loaded
    LOADING = "loading"      # Tool is currently loading (async)
    UNLOADED = "unloaded"    # Tool is unloaded
    UNAVAILABLE = "unavailable"  # Tool dependencies not installed
    ERROR = "error"          # Tool had loading error


class ToolInfo:
    """Information about a tool."""
    
    def __init__(
        self,
        id: str,
        name: str,
        description: str,
        category: str,
        tool_class: str,
        module_path: str,
        dependencies: List[str] = None,
        requires_gpu: bool = False,
    ):
        self.id = id
        self.name = name
        self.description = description
        self.category = category
        self.tool_class = tool_class
        self.module_path = module_path
        self.dependencies = dependencies or []
        self.requires_gpu = requires_gpu
        self.status = ToolStatus.UNAVAILABLE
        self.instance = None
        self.error_message: Optional[str] = None
        self.loaded_at: Optional[datetime] = None


class ToolManager:
    """
    Manages optional MedRAX tools.
    
    Handles:
    - Tool discovery
    - On-demand loading/unloading
    - Dependency checking
    - Graceful degradation
    """
    
    def __init__(self):
        self.tools: Dict[str, ToolInfo] = {}
        self.medrax_path = None
        
        # Try to add MedRAX to path
        self._setup_medrax_path()
        
        # Register all available tools
        self._register_all_tools()
        
        # Check availability for each tool
        self._check_tool_availability()
        
    def _setup_medrax_path(self):
        """Setup MedRAX path for imports."""
        try:
            # Add MedRAX to path
            medrax_path = Path(__file__).parent.parent.parent.parent.parent / "medrax"
            if medrax_path.exists():
                sys.path.insert(0, str(medrax_path.parent))
                self.medrax_path = medrax_path
                logger.info(f"[OK] MedRAX path added: {medrax_path}")
            else:
                logger.warning(f"[WARNING] MedRAX path not found: {medrax_path}")
        except Exception as e:
            logger.error(f"[ERROR] Failed to setup MedRAX path: {e}")
    
    def _register_all_tools(self):
        """Register all available tools from MedRAX."""
        
        tool_definitions = [
            # CLASSIFICATION TOOLS
        ToolInfo(
            id="torchxrayvision",
            name="TorchXRayVision Classifier",
            description="Classifies chest X-rays for 18 pathologies using DenseNet model",
            category="classification",
            tool_class="TorchXRayVisionClassifierTool",
            module_path="medrax.tools.classification.torchxrayvision",
            dependencies=["torch", "torchvision", "torchxrayvision", "skimage"],
            requires_gpu=False  # Works on CPU, GPU optional for speed
        ),
            ToolInfo(
                id="arcplus",
                name="ArcPlus Classifier",
                description="Multi-head classifier for 19 diseases and 6 genders using Swin Transformer",
                category="classification",
                tool_class="ArcPlusClassifierTool",
                module_path="medrax.tools.classification.arcplus",
                dependencies=["torch", "torchvision", "timm", "numpy", "PIL"],
                requires_gpu=True
            ),
            
            # VQA TOOLS
            ToolInfo(
                id="chexagent",
                name="CheXagent VQA",
                description="Comprehensive chest X-ray analysis using CheXagent-2-3b model",
                category="vqa",
                tool_class="CheXagentXRayVQATool",
                module_path="medrax.tools.vqa.xray_vqa",
                dependencies=["torch", "transformers"],
                requires_gpu=True
            ),
            ToolInfo(
                id="llava_med",
                name="LLaVA-Med",
                description="Medical visual question answering using LLaVA-Med model",
                category="vqa",
                tool_class="LlavaMedTool",
                module_path="medrax.tools.vqa.llava_med",
                dependencies=["torch", "PIL"],
                requires_gpu=True
            ),
            ToolInfo(
                id="medgemma",
                name="MedGemma VQA",
                description="Medical VQA using MedGemma API",
                category="vqa",
                tool_class="MedGemmaAPIClientTool",
                module_path="medrax.tools.vqa.medgemma.medgemma_client",
                dependencies=[],
                requires_gpu=False
            ),
            
            # SEGMENTATION TOOLS
            ToolInfo(
                id="medsam2",
                name="MedSAM2",
                description="Advanced medical image segmentation using MedSAM2",
                category="segmentation",
                tool_class="MedSAM2Tool",
                module_path="medrax.tools.segmentation.medsam2",
                dependencies=["torch", "numpy", "matplotlib", "PIL", "sam2", "huggingface_hub", "hydra"],
                requires_gpu=True
            ),
            ToolInfo(
                id="chest_segmentation",
                name="Chest X-Ray Segmentation",
                description="Chest X-ray organ segmentation with metrics",
                category="segmentation",
                tool_class="ChestXRaySegmentationTool",
                module_path="medrax.tools.segmentation.segmentation",
                dependencies=["torch", "transformers", "PIL"],
                requires_gpu=True
            ),
            
            # REPORT GENERATION
            ToolInfo(
                id="report_generator",
                name="Radiology Report Generator",
                description="Generates comprehensive radiology reports with findings and impressions",
                category="generation",
                tool_class="ChestXRayReportGeneratorTool",
                module_path="medrax.tools.report_generation",
                dependencies=["torch", "transformers", "PIL"],
                requires_gpu=True
            ),
            
            # GROUNDING
            ToolInfo(
                id="phrase_grounding",
                name="X-Ray Phrase Grounding",
                description="Locates medical findings in X-rays using MAIRA-2",
                category="grounding",
                tool_class="XRayPhraseGroundingTool",
                module_path="medrax.tools.grounding",
                dependencies=["torch", "transformers", "matplotlib", "PIL"],
                requires_gpu=True
            ),
            
            # IMAGE PROCESSING
            ToolInfo(
                id="dicom_processor",
                name="DICOM Processor",
                description="Processes DICOM files and converts to PNG",
                category="processing",
                tool_class="DicomProcessorTool",
                module_path="medrax.tools.dicom",
                dependencies=["pydicom", "numpy", "PIL"],
                requires_gpu=False
            ),
            ToolInfo(
                id="xray_generator",
                name="X-Ray Generator",
                description="Generates synthetic chest X-rays from text descriptions",
                category="generation",
                tool_class="ChestXRayGeneratorTool",
                module_path="medrax.tools.xray_generation",
                dependencies=["torch", "diffusers"],
                requires_gpu=True
            ),
            
            # RETRIEVAL
            ToolInfo(
                id="rag",
                name="Medical Knowledge RAG",
                description="Answers medical questions using RAG with knowledge base",
                category="retrieval",
                tool_class="RAGTool",
                module_path="medrax.tools.rag",
                dependencies=["langchain"],
                requires_gpu=False
            ),
            ToolInfo(
                id="web_search",
                name="DuckDuckGo Search",
                description="Web search for medical information",
                category="retrieval",
                tool_class="DuckDuckGoSearchTool",
                module_path="medrax.tools.browsing.duckduckgo",
                dependencies=["duckduckgo_search"],
                requires_gpu=False
            ),
            ToolInfo(
                id="web_browser",
                name="Web Browser",
                description="Browse and extract content from web pages",
                category="retrieval",
                tool_class="WebBrowserTool",
                module_path="medrax.tools.browsing.web_browser",
                dependencies=[],
                requires_gpu=False
            ),
            
            # CODE EXECUTION
            ToolInfo(
                id="python_sandbox",
                name="Python Sandbox",
                description="Execute Python code in secure sandbox (requires Deno)",
                category="execution",
                tool_class="PyodideSandboxTool",
                module_path="medrax.tools.python_tool",
                dependencies=["langchain_sandbox"],
                requires_gpu=False
            ),
        ]
        
        for tool_def in tool_definitions:
            self.tools[tool_def.id] = tool_def
            logger.debug(f"Registered tool: {tool_def.name}")
            
        logger.info(f"[OK] Registered {len(tool_definitions)} tools")
    
    def _check_dependency(self, dep_name: str) -> bool:
        """Check if a single dependency is available."""
        try:
            __import__(dep_name)
            return True
        except ImportError:
            return False
    
    def _check_tool_availability(self):
        """Check availability for each tool individually."""
        for tool_id, tool in self.tools.items():
            if not tool.dependencies:
                # No dependencies, mark as available
                tool.status = ToolStatus.AVAILABLE
                continue
            
            # Check each dependency
            missing_deps = []
            for dep in tool.dependencies:
                if not self._check_dependency(dep):
                    missing_deps.append(dep)
            
            if missing_deps:
                tool.status = ToolStatus.UNAVAILABLE
                tool.error_message = f"Missing dependencies: {', '.join(missing_deps)}"
                logger.debug(f"Tool '{tool.name}' unavailable: {tool.error_message}")
            else:
                tool.status = ToolStatus.AVAILABLE
                logger.debug(f"Tool '{tool.name}' available")
        
        available_count = sum(1 for t in self.tools.values() if t.status == ToolStatus.AVAILABLE)
        unavailable_count = len(self.tools) - available_count
        logger.info(f"[OK] Tool availability: {available_count} available, {unavailable_count} unavailable")
    
    def get_all_tools(self) -> List[Dict[str, Any]]:
        """Get list of all tools with their status."""
        return [
            {
                "id": tool.id,
                "name": tool.name,
                "description": tool.description,
                "category": tool.category,
                "status": tool.status,
                "dependencies": tool.dependencies,
                "requires_gpu": tool.requires_gpu,
                "error_message": tool.error_message,
                "loaded_at": tool.loaded_at.isoformat() if tool.loaded_at else None,
            }
            for tool in self.tools.values()
        ]
    
    def get_tool(self, tool_id: str) -> Optional[ToolInfo]:
        """Get a specific tool."""
        return self.tools.get(tool_id)
    
    def load_tool(self, tool_id: str) -> Dict[str, Any]:
        """
        Initiate loading of a tool (returns immediately for async loading).
        
        Returns:
            Status information about the tool
        """
        tool = self.tools.get(tool_id)
        if not tool:
            return {"success": False, "error": f"Tool '{tool_id}' not found"}
        
        if tool.status == ToolStatus.UNAVAILABLE:
            return {
                "success": False,
                "error": f"Tool unavailable: {tool.error_message}"
            }
        
        if tool.status == ToolStatus.LOADED:
            return {
                "success": True,
                "message": f"Tool '{tool.name}' is already loaded",
                "tool": self._tool_to_dict(tool)
            }
        
        if tool.status == ToolStatus.LOADING:
            return {
                "success": True,
                "message": f"Tool '{tool.name}' is already loading",
                "tool": self._tool_to_dict(tool)
            }
        
        # Mark as loading and return immediately
        tool.status = ToolStatus.LOADING
        tool.error_message = None
        
        logger.info(f"Tool '{tool.name}' marked as loading (will load in background)")
        
        return {
            "success": True,
            "message": f"Tool '{tool.name}' is loading (may take several minutes for first-time model download)",
            "tool": self._tool_to_dict(tool)
        }
    
    def load_tool_in_background(self, tool_id: str):
        """
        Actually load the tool in background (can take a long time for large models).
        This is called as a background task after load_tool() returns.
        """
        tool = self.tools.get(tool_id)
        if not tool or tool.status != ToolStatus.LOADING:
            return
        
        try:
            logger.info(f"Background loading tool: {tool.name}")
            
            # Import and instantiate the tool (this may take 10-30 minutes for large models)
            tool_instance = self._load_tool_instance(tool)
            
            if tool_instance:
                tool.instance = tool_instance
                tool.status = ToolStatus.LOADED
                tool.loaded_at = datetime.utcnow()
                tool.error_message = None
                
                logger.info(f"[OK] Tool loaded in background: {tool.name}")
            else:
                tool.status = ToolStatus.ERROR
                tool.error_message = "Failed to instantiate tool"
                logger.error(f"Failed to load tool {tool.name}: Failed to instantiate")
                
        except Exception as e:
            logger.error(f"Failed to load tool {tool.name} in background: {e}")
            tool.status = ToolStatus.ERROR
            tool.error_message = str(e)
    
    def _load_tool_instance(self, tool: ToolInfo):
        """Load the actual tool instance with model caching."""
        try:
            # Set up model caching environment variables
            import os
            from ..config import settings
            
            # Ensure cache directories exist
            cache_dir = os.path.expanduser(settings.MODEL_CACHE_DIR)
            os.makedirs(cache_dir, exist_ok=True)
            
            # Set Hugging Face cache
            hf_cache = os.path.expanduser(settings.HUGGINGFACE_CACHE_DIR)
            os.makedirs(hf_cache, exist_ok=True)
            os.environ['HF_HOME'] = hf_cache
            os.environ['TRANSFORMERS_CACHE'] = hf_cache
            
            # Set Torch cache
            torch_cache = os.path.expanduser(settings.TORCH_CACHE_DIR)
            os.makedirs(torch_cache, exist_ok=True)
            os.environ['TORCH_HOME'] = torch_cache
            
            logger.info(f"Model caching configured for {tool.name}")
            logger.debug(f"  HF Cache: {hf_cache}")
            logger.debug(f"  Torch Cache: {torch_cache}")
            
            # Dynamic import
            module = __import__(tool.module_path, fromlist=[tool.tool_class])
            tool_class = getattr(module, tool.tool_class)
            
            # Instantiate (models will be downloaded to cache on first use)
            logger.info(f"Instantiating {tool.tool_class}...")
            return tool_class()
                
        except ImportError as e:
            logger.error(f"Import error for tool {tool.name}: {e}")
            raise Exception(f"Missing dependencies: {e}")
        except Exception as e:
            logger.error(f"Error loading tool {tool.name}: {e}")
            raise
    
    def unload_tool(self, tool_id: str) -> Dict[str, Any]:
        """
        Unload a specific tool.
        
        Returns:
            Status information about the tool
        """
        tool = self.tools.get(tool_id)
        if not tool:
            return {"success": False, "error": f"Tool '{tool_id}' not found"}
        
        if tool.status != ToolStatus.LOADED:
            return {
                "success": True,
                "message": f"Tool '{tool.name}' is not loaded",
                "tool": self._tool_to_dict(tool)
            }
        
        try:
            logger.info(f"Unloading tool: {tool.name}")
            
            # Clear the instance
            tool.instance = None
            tool.status = ToolStatus.AVAILABLE if tool.error_message is None else ToolStatus.UNAVAILABLE
            tool.loaded_at = None
            
            logger.info(f"[OK] Tool unloaded: {tool.name}")
            return {
                "success": True,
                "message": f"Tool '{tool.name}' unloaded successfully",
                "tool": self._tool_to_dict(tool)
            }
            
        except Exception as e:
            logger.error(f"Failed to unload tool {tool.name}: {e}")
            return {"success": False, "error": str(e)}
    
    def get_loaded_tools(self) -> List[Any]:
        """Get all currently loaded tool instances."""
        return [
            tool.instance
            for tool in self.tools.values()
            if tool.status == ToolStatus.LOADED and tool.instance is not None
        ]
    
    def is_agent_ready(self) -> bool:
        """Check if agent can be created with loaded tools."""
        loaded_tools = self.get_loaded_tools()
        return len(loaded_tools) > 0
    
    def create_agent(self, model=None, system_prompt: str = ""):
        """
        Create MedRAX agent with loaded tools and memory persistence.
        
        Args:
            model: Language model to use (if None, will use default)
            system_prompt: System prompt for the agent
            
        Returns:
            Agent instance or None if not available
        """
        if not self.is_agent_ready():
            logger.warning("Cannot create agent: no tools loaded")
            return None
        
        try:
            from medrax.agent import Agent
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langgraph.checkpoint.memory import MemorySaver
            from ..config import settings
            
            # Use provided model or create default (Gemini 2.5 Pro)
            if model is None:
                model = ChatGoogleGenerativeAI(
                    model="gemini-2.5-pro",
                    api_key=settings.GOOGLE_API_KEY,
                    temperature=0
                )
            
            # Get loaded tool instances directly
            tool_instances = self.get_loaded_tools()
            
            # Create in-memory checkpointer for conversation persistence
            checkpointer = MemorySaver()
            
            # Create agent with memory
            self.agent_instance = Agent(
                model=model,
                tools=tool_instances,
                checkpointer=checkpointer,
                system_prompt=system_prompt or self._get_default_system_prompt()
            )
            
            logger.info(f"[OK] Agent created with {len(tool_instances)} tools and memory")
            return self.agent_instance
            
        except Exception as e:
            logger.error(f"Failed to create agent: {e}")
            return None
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for medical agent."""
        loaded_tools = self.get_loaded_tools()
        tool_descriptions = []
        
        for tool in loaded_tools:
            if hasattr(tool, 'name'):
                tool_name = tool.name
            elif hasattr(tool, '__class__'):
                tool_name = tool.__class__.__name__
            else:
                tool_name = str(tool)
            
            if hasattr(tool, 'description'):
                tool_desc = tool.description
            else:
                tool_desc = "Available tool"
            
            tool_descriptions.append(f"- {tool_name}: {tool_desc}")
        
        tools_list = "\n".join(tool_descriptions) if tool_descriptions else "- Various medical imaging and analysis tools"
        
        return f"""You are MedRAX, an advanced AI assistant specialized in medical imaging analysis and clinical support.

You have access to the following tools:
{tools_list}

IMPORTANT: Use the available tools proactively whenever they can help answer the user's questions or requests. This includes:
- Using web search tools when asked to look up information, research topics, or find current data
- Using medical imaging tools for analyzing scans and images
- Using classification tools to identify pathologies
- Using question answering tools for medical queries
- Using any other available tools that are relevant to the request

Do not refuse to use tools based on assumptions about their purpose. If a tool is loaded and can help with the user's request, use it.

When you receive search results from tools:
- The results contain a "results" array with items having "title", "url", and "snippet" fields
- Present the information clearly to the user, citing sources when appropriate
- If search returns an error, inform the user about the specific issue

Always be thorough, accurate, and helpful in your responses."""
    
    def _tool_to_dict(self, tool: ToolInfo) -> Dict[str, Any]:
        """Convert tool to dictionary."""
        return {
            "id": tool.id,
            "name": tool.name,
            "description": tool.description,
            "category": tool.category,
            "status": tool.status,
            "loaded_at": tool.loaded_at.isoformat() if tool.loaded_at else None,
        }


# Global tool manager instance
tool_manager = ToolManager()
