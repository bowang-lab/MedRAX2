"""
Tool Manager Service

Handles optional loading/unloading of MedRAX tools.
Provides graceful degradation when tools are not available.
"""

import sys
import importlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..utils.logging_config import logger
from ..config import settings
# from .image_registry import image_registry  # TODO: Re-enable when wrapper is fixed
# from .tool_wrapper import wrap_tool_for_production  # TODO: Re-enable when wrapper is fixed
import threading

# Global lock for thread-safe imports to prevent Python import deadlocks
_import_lock = threading.Lock()

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
        self.cancel_event: Optional[threading.Event] = None  # For cancelling individual tool loads


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
        self._threads: Dict[str, threading.Thread] = {}
        self._shutdown_event = threading.Event()  # For graceful shutdown signaling
        self._threads_lock = threading.Lock()  # Thread-safe access to _threads dict
        self._tool_status_lock = threading.Lock()  # Prevent race conditions on tool status changes
        self._agent_lock = threading.Lock()  # Prevent race conditions on agent creation
        self._checkpointer_lock = threading.Lock()  # Lock for checkpointer memory access
        
        # Limit concurrent background loads to reduce resource contention
        # Use BoundedSemaphore for better error detection
        # NOTE: MUST be 1 to avoid Python import deadlocks when loading tools concurrently
        try:
            max_conc = getattr(settings, 'MAX_CONCURRENT_TOOLS', 1) or 1
        except Exception:
            max_conc = 1
        self._load_semaphore = threading.BoundedSemaphore(max_conc)
        self._semaphore_max = max_conc  # Track max value for cleanup
        
        # Agent and memory persistence
        self.agent_instance = None
        self.checkpointer = None
        self.chat_checkpointers = {}  # Per-chat checkpointers for isolation
        
        # Try to add MedRAX to path
        self._setup_medrax_path()
        
        # Register all available tools
        self._register_all_tools()

        # Backwards-compatible tool ID aliases (old -> canonical)
        self.tool_aliases = {
            "torchxrayvision": "torchxrayvision_classifier",
            "arcplus": "arcplus_classifier",
            "chexagent": "chexagent_xray_vqa",
            "llava_med": "llava_med_qa",
            "chest_segmentation": "chest_xray_segmentation",
            "report_generator": "chest_xray_report_generator",
            "phrase_grounding": "xray_phrase_grounding",
            "xray_generator": "chest_xray_generator",
            "rag": "medical_knowledge_rag",
            "web_search": "duckduckgo_search",
        }
        
        # Check availability for each tool
        self._check_tool_availability()
    
    def __del__(self):
        """Cleanup resources on deletion."""
        try:
            if hasattr(self, '_shutdown_event') and not self._shutdown_event.is_set():
                self.shutdown()
        except Exception:
            pass  # Ignore errors in destructor
        
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

            # Also add local MedSAM2 repo path so 'sam2' can be imported as a dependency
            medsam2_path = Path(__file__).parent.parent.parent.parent.parent / "MedSAM2"
            if medsam2_path.exists():
                # Insert at front to ensure resolution before site-packages fallbacks
                sys.path.insert(0, str(medsam2_path))
                logger.info(f"[OK] MedSAM2 path added: {medsam2_path}")
            else:
                logger.info(f"[INFO] MedSAM2 path not found (optional): {medsam2_path}")
        except Exception as e:
            logger.error(f"[ERROR] Failed to setup MedRAX path: {e}")
    
    def _register_all_tools(self):
        """Register all available tools from MedRAX."""
        
        tool_definitions = [
            # CLASSIFICATION TOOLS
        ToolInfo(
            id="torchxrayvision_classifier",
            name="TorchXRayVision Classifier",
            description="Classifies chest X-rays for 18 pathologies using DenseNet model",
            category="classification",
            tool_class="TorchXRayVisionClassifierTool",
            module_path="medrax.tools.classification.torchxrayvision",
            dependencies=["torch", "torchvision", "torchxrayvision", "skimage"],
            requires_gpu=False  # Works on CPU, GPU optional for speed
        ),
            ToolInfo(
                id="arcplus_classifier",
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
                id="chexagent_xray_vqa",
                name="CheXagent X-Ray VQA",
                description="Comprehensive chest X-ray analysis using CheXagent-2-3b model",
                category="vqa",
                tool_class="CheXagentXRayVQATool",
                module_path="medrax.tools.vqa.xray_vqa",
                dependencies=["torch", "transformers"],
                requires_gpu=True
            ),
            ToolInfo(
                id="llava_med_qa",
                name="LLaVA-Med QA",
                description="Medical visual question answering using LLaVA-Med model",
                category="vqa",
                tool_class="LlavaMedTool",
                module_path="medrax.tools.vqa.llava_med",
                dependencies=["torch", "PIL"],
                requires_gpu=True
            ),
            # SEGMENTATION TOOLS
            ToolInfo(
                id="medsam2",
                name="MedSAM2",
                description="Advanced medical image segmentation using MedSAM2",
                category="segmentation",
                tool_class="MedSAM2Tool",
                module_path="medrax.tools.segmentation.medsam2",
                dependencies=["torch", "numpy", "matplotlib", "PIL", "sam2", "huggingface_hub", "hydra", "iopath"],
                requires_gpu=True
            ),
            ToolInfo(
                id="chest_xray_segmentation",
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
                id="chest_xray_report_generator",
                name="Chest X-Ray Report Generator",
                description="Generates comprehensive radiology reports with findings and impressions",
                category="generation",
                tool_class="ChestXRayReportGeneratorTool",
                module_path="medrax.tools.report_generation",
                dependencies=["torch", "transformers", "PIL"],
                requires_gpu=True
            ),
            
            # GROUNDING
            ToolInfo(
                id="xray_phrase_grounding",
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
                id="image_visualizer",
                name="Image Visualizer",
                description="Displays images with optional annotations",
                category="utility",
                tool_class="ImageVisualizerTool",
                module_path="medrax.tools.utils",
                dependencies=["matplotlib", "skimage"],
                requires_gpu=False
            ),
            ToolInfo(
                id="chest_xray_generator",
                name="Chest X-Ray Generator",
                description="Generates synthetic chest X-rays from text descriptions",
                category="generation",
                tool_class="ChestXRayGeneratorTool",
                module_path="medrax.tools.xray_generation",
                dependencies=["torch", "diffusers"],
                requires_gpu=True
            ),
            
            # RETRIEVAL
            ToolInfo(
                id="medical_knowledge_rag",
                name="Medical Knowledge RAG",
                description="Answers medical questions using RAG with knowledge base",
                category="retrieval",
                tool_class="RAGTool",
                module_path="medrax.tools.rag",
                dependencies=["langchain"],
                requires_gpu=False
            ),
            ToolInfo(
                id="duckduckgo_search",
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
    
    def resolve_tool_id(self, tool_id: str) -> str:
        """Resolve legacy tool IDs to canonical IDs."""
        return self.tool_aliases.get(tool_id, tool_id)

    def get_tool(self, tool_id: str) -> Optional[ToolInfo]:
        """Get a specific tool."""
        return self.tools.get(self.resolve_tool_id(tool_id))
    
    def load_tool(self, tool_id: str) -> Dict[str, Any]:
        """
        Initiate loading of a tool (returns immediately for async loading).
        Thread-safe with status locking to prevent race conditions.
        
        Returns:
            Status information about the tool
        """
        tool = self.tools.get(self.resolve_tool_id(tool_id))
        if not tool:
            return {"success": False, "error": f"Tool '{tool_id}' not found"}
        
        # Use lock to prevent race conditions on status checks/updates
        with self._tool_status_lock:
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
        Checks shutdown event and per-tool cancel event to allow graceful cancellation.
        """
        tool = self.tools.get(self.resolve_tool_id(tool_id))
        if not tool or tool.status != ToolStatus.LOADING:
            if tool and tool.status == ToolStatus.LOADED:
                logger.debug(f"Tool {tool.name} already loaded, skipping")
            return
        
        # Check if shutdown was requested before even starting
        if self._shutdown_event.is_set():
            logger.info(f"Shutdown requested, aborting load of {tool.name}")
            tool.status = ToolStatus.AVAILABLE
            tool.cancel_event = None
            return
        
        # Check if this tool was cancelled
        if tool.cancel_event and tool.cancel_event.is_set():
            logger.info(f"Load cancelled for {tool.name}")
            tool.status = ToolStatus.AVAILABLE
            tool.cancel_event = None
            return
        
        acquired = False
        try:
            # Concurrency cap with timeout to allow shutdown/cancellation checks
            acquired = self._load_semaphore.acquire(timeout=1.0)
            if not acquired:
                # Couldn't acquire semaphore, probably at max concurrency
                # Check shutdown/cancellation and retry or abort
                if self._shutdown_event.is_set():
                    logger.info(f"Shutdown during semaphore wait for {tool.name}")
                    tool.status = ToolStatus.AVAILABLE
                    tool.cancel_event = None
                    return
                if tool.cancel_event and tool.cancel_event.is_set():
                    logger.info(f"Cancelled during semaphore wait for {tool.name}")
                    tool.status = ToolStatus.AVAILABLE
                    tool.cancel_event = None
                    return
                # Try again without timeout (blocking)
                self._load_semaphore.acquire()
                acquired = True
            
            # Check shutdown/cancellation again after acquiring semaphore
            if self._shutdown_event.is_set():
                logger.info(f"Shutdown after acquiring semaphore for {tool.name}")
                tool.status = ToolStatus.AVAILABLE
                tool.cancel_event = None
                return
            if tool.cancel_event and tool.cancel_event.is_set():
                logger.info(f"Cancelled after acquiring semaphore for {tool.name}")
                tool.status = ToolStatus.AVAILABLE
                tool.cancel_event = None
                return
            
            logger.info(f"Background loading tool: {tool.name}")
            
            # Import and instantiate the tool (this may take 10-30 minutes for large models)
            tool_instance = self._load_tool_instance(tool)
            
            # Final shutdown/cancellation check before marking as loaded
            if self._shutdown_event.is_set():
                logger.info(f"Shutdown during load of {tool.name}, discarding instance")
                tool.status = ToolStatus.AVAILABLE
                tool.cancel_event = None
                # Try to cleanup the instance if it has cleanup methods
                if tool_instance and hasattr(tool_instance, 'cleanup'):
                    try:
                        tool_instance.cleanup()
                    except Exception:
                        pass
                return
            if tool.cancel_event and tool.cancel_event.is_set():
                logger.info(f"Cancelled during load of {tool.name}, discarding instance")
                tool.status = ToolStatus.AVAILABLE
                tool.cancel_event = None
                # Try to cleanup the instance if it has cleanup methods
                if tool_instance and hasattr(tool_instance, 'cleanup'):
                    try:
                        tool_instance.cleanup()
                    except Exception:
                        pass
                return
            
            # Use lock when updating tool status
            with self._tool_status_lock:
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
            
            # Reset agent outside lock (to avoid holding lock during agent cleanup)
            if tool_instance:
                with self._agent_lock:
                    self.agent_instance = None
                
        except Exception as e:
            # Use lock when updating status on error
            with self._tool_status_lock:
                if not self._shutdown_event.is_set():
                    logger.error(f"Failed to load tool {tool.name} in background: {e}")
                    tool.status = ToolStatus.ERROR
                    tool.error_message = str(e)
                else:
                    logger.info(f"Exception during shutdown for {tool.name}: {e}")
                    tool.status = ToolStatus.AVAILABLE
        finally:
            # Always release semaphore if we acquired it
            if acquired:
                try:
                    self._load_semaphore.release()
                except ValueError:
                    # Semaphore already released or corrupted, ignore
                    pass

    def start_background_load(self, tool_id: str) -> bool:
        """Start background loading in a managed thread (tracked for clean shutdown)."""
        resolved_id = self.resolve_tool_id(tool_id)
        tool = self.tools.get(resolved_id)
        if not tool:
            return False
        if tool.status in (ToolStatus.LOADING, ToolStatus.LOADED):
            return True
        
        # Don't start new loads if shutting down
        if self._shutdown_event.is_set():
            logger.warning(f"Cannot start load during shutdown: {tool_id}")
            return False
        
        # Ensure marked loading
        load_result = self.load_tool(tool_id)
        if not load_result.get("success"):
            return False
        
        # Create cancellation event for this tool
        tool.cancel_event = threading.Event()
        
        # Create and start daemon thread
        def _runner():
            try:
                self.load_tool_in_background(resolved_id)
            finally:
                # Remove from tracking when done (thread-safe)
                with self._threads_lock:
                    self._threads.pop(resolved_id, None)
                # Clear cancel event
                if tool.cancel_event:
                    tool.cancel_event = None
        
        t = threading.Thread(target=_runner, name=f"tool-loader-{resolved_id}", daemon=True)
        
        # Add to tracking (thread-safe)
        with self._threads_lock:
            self._threads[resolved_id] = t
        
        t.start()
        return True

    def shutdown(self):
        """Attempt to cleanly stop background threads at app shutdown."""
        logger.info("Shutting down tool manager...")
        
        # Signal shutdown to all background threads
        self._shutdown_event.set()
        
        # Cancel all tool loads that might be in progress
        for tool in self.tools.values():
            if tool.cancel_event and tool.status == ToolStatus.LOADING:
                try:
                    tool.cancel_event.set()
                    logger.debug(f"Cancelled load for {tool.name}")
                except Exception as e:
                    logger.debug(f"Error cancelling {tool.name}: {e}")
        
        # Get snapshot of threads (thread-safe)
        with self._threads_lock:
            threads_snapshot = list(self._threads.items())
        
        # Join all threads with timeout, trying multiple times
        active_threads = []
        for tool_id, t in threads_snapshot:
            try:
                logger.debug(f"Waiting for thread {tool_id} to complete...")
                # First attempt with short timeout
                t.join(timeout=2.0)
                
                # If still alive, try one more time with longer timeout
                if t.is_alive():
                    logger.debug(f"Thread {tool_id} still running, waiting longer...")
                    t.join(timeout=5.0)
                
                if t.is_alive():
                    active_threads.append(tool_id)
                    logger.warning(f"Thread {tool_id} still active after shutdown timeout")
            except Exception as e:
                logger.debug(f"Error joining thread {tool_id}: {e}")
        
        # Clear thread tracking (thread-safe)
        with self._threads_lock:
            self._threads.clear()
        
        # Unload all loaded tools to free resources
        loaded_tools = [t.id for t in self.tools.values() if t.status == ToolStatus.LOADED]
        for tool_id in loaded_tools:
            try:
                self.unload_tool(tool_id)
            except Exception as e:
                logger.debug(f"Error unloading tool {tool_id}: {e}")
        
        # Force-release semaphore to prevent leaks
        # Try to release as many times as max value to drain it
        logger.debug("Force-releasing semaphore to prevent leaks...")
        released_count = 0
        for _ in range(self._semaphore_max):
            try:
                self._load_semaphore.release()
                released_count += 1
            except ValueError:
                # Can't release more, semaphore is at max
                break
        
        if released_count > 0:
            logger.debug(f"Force-released semaphore {released_count} times")
        
        # Explicitly delete threading primitives to clean up OS-level semaphores
        # This is crucial for preventing semaphore leaks
        primitives_to_cleanup = [
            ('_load_semaphore', self._load_semaphore),
            ('_shutdown_event', self._shutdown_event),
            ('_threads_lock', self._threads_lock),
            ('_tool_status_lock', self._tool_status_lock),
            ('_agent_lock', self._agent_lock),
            ('_checkpointer_lock', self._checkpointer_lock),
        ]
        
        for name, primitive in primitives_to_cleanup:
            try:
                # For Events, explicitly clear them before deletion
                if isinstance(primitive, threading.Event):
                    primitive.clear()
                delattr(self, name)
                logger.debug(f"Cleaned up {name}")
            except Exception as e:
                logger.debug(f"Error cleaning up {name}: {e}")
        
        # Clean up tool cancel events
        for tool in self.tools.values():
            if tool.cancel_event:
                try:
                    tool.cancel_event.clear()
                    tool.cancel_event = None
                except Exception as e:
                    logger.debug(f"Error cleaning up cancel event for {tool.name}: {e}")
        
        # Clean up global import lock (module-level)
        # Safe to do now since all threads have been joined
        global _import_lock
        try:
            del _import_lock
            logger.debug("Cleaned up global _import_lock")
        except Exception as e:
            logger.debug(f"Error cleaning up _import_lock: {e}")
        
        if active_threads:
            logger.info(f"Shutdown complete ({len(active_threads)} background threads will terminate with process)")
        else:
            logger.info("Shutdown complete (all threads joined cleanly)")
    
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
            
            # Thread-safe dynamic import (prevents Python import deadlocks)
            with _import_lock:
                logger.debug(f"Importing {tool.module_path}.{tool.tool_class}")
                module = importlib.import_module(tool.module_path)
                tool_class = getattr(module, tool.tool_class)
            
            # Instantiate (models will be downloaded to cache on first use)
            logger.info(f"Instantiating {tool.tool_class}...")
            
            # Special handling for tools that require configuration
            if tool.tool_class == "RAGTool":
                # RAGTool requires a RAGConfig parameter
                # Also need to set Cohere and Pinecone API keys as environment variables
                import os
                
                # Cohere SDK looks for CO_API_KEY env var
                if settings.COHERE_API_KEY:
                    os.environ['CO_API_KEY'] = settings.COHERE_API_KEY
                    os.environ['COHERE_API_KEY'] = settings.COHERE_API_KEY
                    logger.info(f"Set Cohere API key from settings")
                
                # Pinecone API key
                if settings.PINECONE_API_KEY:
                    os.environ['PINECONE_API_KEY'] = settings.PINECONE_API_KEY
                    logger.info(f"Set Pinecone API key from settings")
                
                from medrax.rag.rag import RAGConfig
                config = RAGConfig()  # Use default configuration
                logger.info(f"Creating RAGTool with default RAGConfig")
                return tool_class(config)
            elif tool.tool_class == "MedGemmaTool":
                # MedGemmaTool - direct integration with optional configuration
                medgemma_kwargs = {}
                
                # Optional: Use 4-bit quantization (saves VRAM)
                use_4bit = getattr(settings, 'MEDGEMMA_USE_4BIT', False)
                if use_4bit:
                    medgemma_kwargs['use_4bit'] = True
                    logger.info("MedGemma will use 4-bit quantization")
                
                # Optional: Custom cache directory
                cache_dir = getattr(settings, 'MODEL_CACHE_DIR', None)
                if cache_dir:
                    medgemma_kwargs['cache_dir'] = cache_dir
                
                logger.info(f"Creating MedGemmaTool (model loads on first use)")
                return tool_class(**medgemma_kwargs)
            elif tool.tool_class == "ArcPlusClassifierTool":
                # ArcPlusClassifierTool - needs cache_dir for model weights
                arcplus_kwargs = {}
                
                # Get model weights directory from settings
                modelweights_dir = getattr(settings, 'MODELWEIGHTS', None)
                if modelweights_dir:
                    arcplus_kwargs['cache_dir'] = modelweights_dir
                    logger.info(f"ArcPlus will load weights from: {modelweights_dir}")
                else:
                    logger.warning("MODELWEIGHTS not set - ArcPlus will not have pretrained weights")
                
                # Device configuration
                arcplus_kwargs['device'] = None  # Auto-detect
                
                logger.info(f"Creating ArcPlusClassifierTool")
                return tool_class(**arcplus_kwargs)
            else:
                # Most tools can be instantiated without parameters
                return tool_class()
                
        except ImportError as e:
            logger.error(f"Import error for tool {tool.name}: {e}")
            raise Exception(f"Missing dependencies: {e}")
        except Exception as e:
            logger.error(f"Error loading tool {tool.name}: {e}")
            raise
    
    def unload_tool(self, tool_id: str) -> Dict[str, Any]:
        """
        Unload a specific tool. If tool is currently loading, cancels the load.
        Thread-safe with status locking.
        
        Returns:
            Status information about the tool
        """
        tool = self.tools.get(self.resolve_tool_id(tool_id))
        if not tool:
            return {"success": False, "error": f"Tool '{tool_id}' not found"}
        
        # Use lock to prevent race conditions during status checks
        with self._tool_status_lock:
            # Handle LOADING status - cancel the load
            if tool.status == ToolStatus.LOADING:
                logger.info(f"Cancelling load of {tool.name}")
                # Signal cancellation
                if tool.cancel_event:
                    tool.cancel_event.set()
                # Note: Thread will clean up and set status to AVAILABLE
                return {
                    "success": True,
                    "message": f"Tool '{tool.name}' load cancelled",
                    "tool": self._tool_to_dict(tool)
                }
            
            if tool.status != ToolStatus.LOADED:
                return {
                    "success": True,
                    "message": f"Tool '{tool.name}' is not loaded",
                    "tool": self._tool_to_dict(tool)
                }
        
        try:
            logger.info(f"Unloading tool: {tool.name}")
            
            # Try to cleanup the instance if it has cleanup methods
            if tool.instance and hasattr(tool.instance, 'cleanup'):
                try:
                    logger.debug(f"Calling cleanup() on {tool.name}")
                    tool.instance.cleanup()
                except Exception as e:
                    logger.warning(f"Error during cleanup of {tool.name}: {e}")
            
            # Use lock when updating status
            with self._tool_status_lock:
                # Clear the instance
                tool.instance = None
                tool.status = ToolStatus.AVAILABLE if tool.error_message is None else ToolStatus.UNAVAILABLE
                tool.loaded_at = None
            
            # Reset agent outside lock
            with self._agent_lock:
                self.agent_instance = None
            
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
    
    def get_wrapped_tools_for_request(self, request_id: str) -> List[Any]:
        """
        Get loaded tools wrapped for production use with a specific request.
        
        This wraps tools that use image paths to automatically resolve
        simple references (image_1, image_2) to actual paths.
        
        Args:
            request_id: Request ID for image resolution
            
        Returns:
            List of wrapped tool instances
        """
        # For now, return unwrapped tools due to Pydantic compatibility issues
        # TODO: Fix wrapper to work with Pydantic-based BaseTool
        logger.warning(f"Tool wrapping temporarily disabled for request {request_id[:8]}")
        return self.get_loaded_tools()
    
    def is_agent_ready(self) -> bool:
        """Check if agent can be created with loaded tools."""
        # Check if any tools are loaded (not just the raw instances)
        for tool in self.tools.values():
            if tool.status == ToolStatus.LOADED and tool.instance:
                return True
        return False
    
    def create_agent(self, model=None, system_prompt: str = "", force_recreate: bool = False, request_id: str = None, chat_id: str = None):
        """
        Create MedRAX agent with loaded tools and memory persistence.
        Thread-safe with locking to prevent concurrent agent creation.
        
        Args:
            model: Language model to use (if None, will use default)
            system_prompt: System prompt for the agent
            force_recreate: If True, force recreation of agent even if one exists
            request_id: Optional request ID for production image path resolution
            chat_id: Optional chat ID for conversation memory isolation
            
        Returns:
            Agent instance or None if not available
        """
        if not self.is_agent_ready():
            logger.warning("Cannot create agent: no tools loaded")
            return None
        
        # Use lock to prevent concurrent agent creation
        with self._agent_lock:
            # For production mode with request_id, always create a new agent with wrapped tools
            if request_id:
                force_recreate = True
            
            # For production with request_id, NEVER reuse agents (isolation)
            # Only reuse for non-production mode (development/testing)
            if not request_id and self.agent_instance is not None and not force_recreate:
                return self.agent_instance
            
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
                
                # Get loaded tool instances - wrapped if request_id provided
                if request_id:
                    tool_instances = self.get_wrapped_tools_for_request(request_id)
                    logger.info(f"Using wrapped tools for request {request_id[:8]}: {len(tool_instances)} tools")
                    if not tool_instances:
                        logger.error(f"No wrapped tools available for request {request_id[:8]}")
                        return None
                else:
                    tool_instances = self.get_loaded_tools()
                    logger.info(f"Using raw tools: {len(tool_instances)} tools")
                
                # Get or create checkpointer for this specific chat (isolation)
                if request_id and chat_id:
                    # Production mode: use per-chat checkpointer for isolation
                    if chat_id not in self.chat_checkpointers:
                        self.chat_checkpointers[chat_id] = MemorySaver()
                        logger.info(f"[OK] Created new checkpointer for chat {chat_id[:8]}")
                    checkpointer = self.chat_checkpointers[chat_id]
                else:
                    # Development mode: use shared checkpointer
                    if self.checkpointer is None:
                        self.checkpointer = MemorySaver()
                        logger.info("[OK] Created shared checkpointer (dev mode)")
                    checkpointer = self.checkpointer
                
                # Create agent with memory
                agent = Agent(
                    model=model,
                    tools=tool_instances,
                    checkpointer=checkpointer,
                    system_prompt=system_prompt or self._get_default_system_prompt()
                )
                
                # Only store as instance if NOT using request_id (for dev/testing)
                if not request_id:
                    self.agent_instance = agent
                    logger.info(f"[OK] Agent created and stored with {len(tool_instances)} tools and memory")
                else:
                    logger.info(f"[OK] Agent created for request {request_id[:8]} with {len(tool_instances)} wrapped tools")
                
                return agent
                
            except Exception as e:
                logger.error(f"Failed to create agent: {e}")
                return None
    
    def cleanup_chat_resources(self, chat_id: str) -> bool:
        """
        Clean up resources for a specific chat.
        
        Args:
            chat_id: Chat ID to clean up
            
        Returns:
            True if cleanup successful
        """
        with self._agent_lock:
            # Clean up chat-specific checkpointer
            if chat_id in self.chat_checkpointers:
                del self.chat_checkpointers[chat_id]
                logger.info(f"Cleaned up checkpointer for chat {chat_id[:8]}")
                return True
        return False
    
    def cleanup_old_chats(self, max_age_hours: int = 24):
        """
        Clean up checkpointers for old chats to prevent memory leaks.
        
        Args:
            max_age_hours: Maximum age in hours before cleanup
        """
        # This would ideally track last access time
        # For now, we'll keep it simple
        with self._agent_lock:
            # Limit total number of chat checkpointers
            MAX_CHECKPOINTERS = 100
            if len(self.chat_checkpointers) > MAX_CHECKPOINTERS:
                # Remove oldest entries (simple FIFO for now)
                to_remove = len(self.chat_checkpointers) - MAX_CHECKPOINTERS
                for chat_id in list(self.chat_checkpointers.keys())[:to_remove]:
                    del self.chat_checkpointers[chat_id]
                logger.info(f"Cleaned up {to_remove} old chat checkpointers")
    
    def clear_chat_memory(self, thread_id: str) -> bool:
        """
        Clear memory for a specific chat thread.
        Thread-safe with locking to prevent concurrent checkpointer access issues.
        
        Args:
            thread_id: The chat ID / thread ID to clear memory for
            
        Returns:
            True if successful, False otherwise
        """
        if self.checkpointer is None:
            logger.warning("No checkpointer available to clear memory")
            return False
        
        try:
            # Use lock to prevent concurrent access to checkpointer storage
            with self._checkpointer_lock:
                # Clear the specific thread from checkpointer
                config = {"configurable": {"thread_id": thread_id}}
                # MemorySaver stores state in memory dict, clear it
                if hasattr(self.checkpointer, 'storage'):
                    # Remove the thread from storage
                    thread_key = (thread_id,)
                    if thread_key in self.checkpointer.storage:
                        del self.checkpointer.storage[thread_key]
                        logger.info(f"[OK] Cleared memory for thread: {thread_id[:8]}")
                        return True
                logger.warning(f"Could not clear memory for thread: {thread_id[:8]}")
                return False
        except Exception as e:
            logger.error(f"Failed to clear memory for thread {thread_id[:8]}: {e}")
            return False
    
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

IMPORTANT TOOL USAGE GUIDELINES:
1. Use tools by their exact names as listed above (e.g., 'torchxrayvision_classifier', 'arcplus_classifier', etc.)
2. NEVER call a tool named 'run' - this tool does not exist
3. When asked to "check all tools" or "use all tools", interpret this as using multiple relevant tools from the list above
4. Use the available tools proactively whenever they can help answer the user's questions or requests:
   - Medical imaging tools for analyzing scans and images
   - Classification tools to identify pathologies
   - Question answering tools for medical queries
   - Web search tools when asked to look up information
   - Any other relevant tools from the list

5. If a user asks you to analyze an image with "all tools", use the most relevant tools from your available list:
   - For chest X-rays: torchxrayvision_classifier, arcplus_classifier, chest_xray_report_generator, etc.
   - For general medical images: relevant VQA and classification tools

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
