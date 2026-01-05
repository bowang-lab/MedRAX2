"""
Tool Wrapper for Production Image Handling

Wraps tools to automatically resolve image references using the registry.
"""

from typing import Any, Dict, Optional, Tuple
from langchain.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
from pydantic import Field
import logging
import inspect
from .image_registry import image_registry

logger = logging.getLogger(__name__)


class ImageResolvingToolWrapper(BaseTool):
    """
    Wrapper that automatically resolves image references before passing to the actual tool.
    
    This allows the LLM to use simple references like "image_1" instead of full paths,
    preventing transcription errors.
    """
    
    # Declare fields for Pydantic
    wrapped_tool: BaseTool = Field(exclude=True)  # Exclude from serialization
    request_id: Optional[str] = Field(default=None, exclude=True)
    
    def __init__(self, tool: BaseTool, request_id: Optional[str] = None):
        """
        Initialize wrapper with the actual tool.
        
        Args:
            tool: The actual tool to wrap
            request_id: Current request ID for image resolution
        """
        # Store wrapped tool first
        self.wrapped_tool = tool
        self.request_id = request_id
        
        # Initialize parent with required fields
        super().__init__(
            name=tool.name,
            description=f"{tool.description} (Use 'image_1', 'image_2', etc. for image references)"
        )
        
        # Copy other tool attributes
        self.args_schema = tool.args_schema
        self.return_direct = tool.return_direct if hasattr(tool, 'return_direct') else False
        self.verbose = tool.verbose if hasattr(tool, 'verbose') else False
        self.callbacks = tool.callbacks if hasattr(tool, 'callbacks') else None
        self.tags = tool.tags if hasattr(tool, 'tags') else None
        self.metadata = tool.metadata if hasattr(tool, 'metadata') else None
        self.handle_tool_error = tool.handle_tool_error if hasattr(tool, 'handle_tool_error') else None
        self.handle_validation_error = tool.handle_validation_error if hasattr(tool, 'handle_validation_error') else None
    
    def set_request_id(self, request_id: str):
        """Set the current request ID for image resolution."""
        self.request_id = request_id
    
    def _resolve_image_args(self, *args, **kwargs) -> Tuple[tuple, dict]:
        """
        Resolve any image path arguments using the registry.
        
        Returns:
            Resolved args and kwargs
        """
        if not self.request_id:
            return args, kwargs
        
        # Process kwargs
        resolved_kwargs = {}
        for key, value in kwargs.items():
            # Check if this looks like an image path argument
            if key in ['image_path', 'image_paths', 'image', 'images', 'scan_path', 'scan_paths']:
                if isinstance(value, str):
                    # Single image path
                    resolved = image_registry.resolve_image(self.request_id, value)
                    if resolved:
                        logger.info(f"Resolved {key}: {value} -> {resolved}")
                        resolved_kwargs[key] = resolved
                    else:
                        logger.warning(f"Failed to resolve {key}: {value}")
                        resolved_kwargs[key] = value
                elif isinstance(value, list):
                    # List of image paths
                    resolved_list = []
                    for item in value:
                        if isinstance(item, str):
                            resolved = image_registry.resolve_image(self.request_id, item)
                            if resolved:
                                logger.info(f"Resolved {key}[]: {item} -> {resolved}")
                                resolved_list.append(resolved)
                            else:
                                logger.warning(f"Failed to resolve {key}[]: {item}")
                                resolved_list.append(item)
                        else:
                            resolved_list.append(item)
                    resolved_kwargs[key] = resolved_list
                else:
                    resolved_kwargs[key] = value
            else:
                resolved_kwargs[key] = value
        
        # Process positional args (less common but handle them)
        resolved_args = []
        for arg in args:
            if isinstance(arg, str) and arg.startswith('image_'):
                resolved = image_registry.resolve_image(self.request_id, arg)
                if resolved:
                    logger.info(f"Resolved arg: {arg} -> {resolved}")
                    resolved_args.append(resolved)
                else:
                    resolved_args.append(arg)
            else:
                resolved_args.append(arg)
        
        return tuple(resolved_args), resolved_kwargs
    
    def _run(
        self,
        *args,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs
    ) -> Any:
        """
        Run the wrapped tool with resolved image paths.
        """
        # Resolve image references
        resolved_args, resolved_kwargs = self._resolve_image_args(*args, **kwargs)
        
        # Call the wrapped tool's _run method
        if run_manager:
            resolved_kwargs['run_manager'] = run_manager
        
        return self.wrapped_tool._run(*resolved_args, **resolved_kwargs)
    
    async def _arun(
        self,
        *args,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs
    ) -> Any:
        """
        Async run the wrapped tool with resolved image paths.
        """
        # Resolve image references
        resolved_args, resolved_kwargs = self._resolve_image_args(*args, **kwargs)
        
        # Call the wrapped tool's _arun method
        if run_manager:
            resolved_kwargs['run_manager'] = run_manager
        
        return await self.wrapped_tool._arun(*resolved_args, **resolved_kwargs)


def wrap_tool_for_production(tool: BaseTool, request_id: Optional[str] = None) -> BaseTool:
    """
    Wrap a tool for production use with automatic image resolution.
    
    Args:
        tool: Tool to wrap
        request_id: Current request ID
        
    Returns:
        Wrapped tool or original if wrapping not needed
    """
    # Check if tool uses image paths
    if hasattr(tool, '_run'):
        sig = inspect.signature(tool._run)
        params = sig.parameters
        
        # Look for image-related parameters
        image_params = ['image_path', 'image_paths', 'image', 'images', 'scan_path', 'scan_paths']
        if any(param in params for param in image_params):
            logger.debug(f"Wrapping tool {tool.name} for image resolution")
            return ImageResolvingToolWrapper(tool, request_id)
    
    # Return original tool if no image parameters
    return tool
