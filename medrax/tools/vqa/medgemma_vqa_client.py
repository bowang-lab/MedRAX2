"""
medgemma_vqa_client.py

Client interface for performing medical VQA using the MedGemma model via a remote MCP server.

This module provides a LangChain-compatible tool that connects to a MedGemma MCP server, sending 
images and prompts for advanced medical VQA. It supports both synchronous and asynchronous 
operation, error handling, and metadata reporting.

Classes:
    - MedGemmaVQAInput: Pydantic schema for tool input arguments.
    - MedGemmaVQATool: LangChain tool for interacting with the MedGemma MCP server.
"""
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field
import base64
from langchain_core.tools import BaseTool
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

from fastmcp.client import Client  

class MedGemmaVQAInput(BaseModel):
    """
    Input schema for the MedGemma VQA tool.

    Attributes:
        image_paths (List[str]): List of paths to medical image files (JPG or PNG) to analyze.
        prompt (str): Question or instruction about the medical images.
        system_prompt (Optional[str]): System prompt to set the context for the model.
        max_new_tokens (int): Maximum number of tokens to generate in the response.
    """
    image_paths: List[str] = Field(
        ...,
        description="List of paths to medical image files to analyze, only supports JPG or PNG images",
    )
    prompt: str = Field(..., description="Question or instruction about the medical images")
    system_prompt: Optional[str] = Field(
        "You are an expert radiologist.",
        description="System prompt to set the context for the model",
    )
    max_new_tokens: int = Field(
        300, description="Maximum number of tokens to generate in the response"
    )

class MedGemmaVQATool(BaseTool):
    """
    Medical visual question answering tool that connects to a MedGemma MCP server.

    This LangChain-compatible tool sends medical images and prompts to a remote MedGemma MCP server for advanced VQA using Google's MedGemma 4B instruction-tuned model.

    Attributes:
        name (str): Name of the tool.
        description (str): Description of the tool's purpose.
        args_schema (type[BaseModel]): Pydantic schema for input arguments.
        return_direct (bool): Whether to return results directly.
        mcp_server_url (str): URL of the remote MCP server.
    """
    name: str = "medgemma_medical_vqa"
    description: str = (
        "Advanced medical visual question answering tool using Google's MedGemma 4B instruction-tuned model. "
        "Connects to a remote MCP server to perform analysis."
    )
    args_schema: type[BaseModel] = MedGemmaVQAInput
    return_direct: bool = False

    mcp_server_url: str

    def __init__(self, mcp_server_url: str = "http://localhost:8000", **kwargs: Any) -> None:
        """
        Initialize the MedGemmaVQATool.

        Args:
            mcp_server_url (str): URL of the MedGemma MCP server.
            **kwargs: Additional keyword arguments for BaseTool.
        """
        super().__init__(mcp_server_url=mcp_server_url, **kwargs)
    
    def _create_error_response(
        self,
        image_paths: List[str],
        prompt: str,
        error_message: str,
        error_type: str,
        error_details: str,
    ) -> Dict[str, Any]:
        """
        Helper method to create a standardized error response dictionary.

        Args:
            image_paths (List[str]): List of image file paths involved in the request.
            prompt (str): The user's prompt.
            error_message (str): Human-readable error message.
            error_type (str): Type/category of error.
            error_details (str): Detailed error information.

        Returns:
            Dict[str, Any]: A dictionary containing the error details.
        """
        return {
            "error": error_message,
            "image_paths": image_paths,
            "prompt": prompt,
            "analysis_status": "failed",
            "error_type": error_type,
            "error_details": error_details,
        }

    def _run(
        self,
        image_paths: List[str],
        prompt: str,
        system_prompt: str = "You are an expert radiologist.",
        max_new_tokens: int = 300,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for the asynchronous tool call.

        This allows the synchronous LangChain agent to use the async MCP client.

        Args:
            image_paths (List[str]): List of image file paths.
            prompt (str): User's question or instruction.
            system_prompt (str): System prompt for the model.
            max_new_tokens (int): Maximum number of tokens to generate.
            run_manager (Optional[CallbackManagerForToolRun]): Optional callback manager.

        Returns:
            Dict[str, Any]: The output dictionary from the tool.
        """
        try:
            # asyncio.run() starts an async event loop, runs our async code, and closes it.
            return asyncio.run(self._arun(
                image_paths=image_paths,
                prompt=prompt,
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens,
                run_manager=run_manager
            ))
        except Exception as e:
             return self._create_error_response(
                image_paths, prompt, f"Failed to run async VQA task: {str(e)}", "async_error", str(e)
            )

    async def _arun(
        self,
        image_paths: List[str],
        prompt: str,
        system_prompt: str = "You are an expert radiologist.",
        max_new_tokens: int = 300,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """
        Asynchronous method to perform VQA by calling the MedGemma MCP server.

        Encodes images, sends them with the prompt to the remote server, and returns the result.

        Args:
            image_paths (List[str]): List of image file paths.
            prompt (str): User's question or instruction.
            system_prompt (str): System prompt for the model.
            max_new_tokens (int): Maximum number of tokens to generate.
            run_manager (Optional[AsyncCallbackManagerForToolRun]): Optional async callback manager.

        Returns:
            Dict[str, Any]: The output dictionary from the tool.
        """
        try:
            async with Client(self.mcp_server_url) as client:
                image_strs = []
                for path in image_paths:
                    with open(path, "rb") as image_file:
                        image_strs.append(base64.b64encode(image_file.read()).decode('utf-8'))

                # Use client.call_tool to execute the remote operation
                result = await client.call_tool(
                    "medgemma_vqa",
                    {
                        "image_strs": image_strs,
                        "prompt": prompt,
                        "system_prompt": system_prompt,
                        "max_new_tokens": max_new_tokens,
                    }
                )

            if result.is_error:
                # If there's an error, the details are in structured_content
                error_details = result.structured_content or {"error": "Unknown server error"}
                return self._create_error_response(
                    image_paths,
                    prompt,
                    error_details.get("error", "Unknown server error"),
                    "server_error",
                    str(error_details),
                )

            # If successful, return the structured_content directly
            return result.structured_content

        except Exception as e:
            return self._create_error_response(
                image_paths, prompt, f"Failed to call MedGemma MCP server: {str(e)}", "client_error", str(e)
            )