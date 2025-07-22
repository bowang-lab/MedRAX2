"""
medgemma_mcp.py

This module provides an MCP server for VQA using the MedGemma model.
It initializes a pipeline compatible with Apple Silicon (MPS) or CPU,
exposes a VQA tool via FastMCP, and can be run as a standalone server.

Functions:
    - initialize_medgemma_pipeline: Initializes the MedGemma pipeline for inference.
    - medgemma_vqa: Performs VQA using the MedGemma model via MCP.
    - start_server: Initializes the pipeline and runs the MCP server.
"""
import torch
from PIL import Image
from transformers import pipeline, BitsAndBytesConfig
from fastmcp import FastMCP
from typing import List, Dict, Any
import base64
from io import BytesIO
import os

# Initialize the MCP server
mcp = FastMCP("MedGemmaVQAServer")

# Global variable to hold the pipeline
pipe = None

def initialize_medgemma_pipeline():
    """
    Initializes the MedGemma pipeline for Apple Silicon (MPS) or CPU.
    ... (rest of the function docstring) ...
    """
    global pipe
    if pipe is None:
        try:
            # Detect MPS for Apple Silicon
            if torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            print(f"Using device: {device}")

            dtype = torch.bfloat16

            model_kwargs = {
                "torch_dtype": dtype,
            }

            # Set device_map to the detected device
            model_kwargs["device_map"] = device

            # Get token for gated model access
            hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN", None)
            if hf_token is None:
                print("Warning: HUGGING_FACE_HUB_TOKEN not set. Trying anonymous access.")

            pipeline_kwargs = {
                "model": "google/medgemma-4b-it",
                "model_kwargs": model_kwargs,
                "trust_remote_code": True,
                "use_cache": True,
                "token": hf_token
            }
            pipe = pipeline("image-text-to-text", **pipeline_kwargs)
            print("MedGemma pipeline initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize MedGemma pipeline: {e}")
            raise

@mcp.tool
def medgemma_vqa(image_strs: List[str], prompt: str, system_prompt: str, max_new_tokens: int) -> Dict[str, Any]:
    """
    Performs Visual Question Answering using the MedGemma model.
    ... (rest of the function docstring) ...
    """
    global pipe
    if pipe is None:
        return {"error": "MedGemma pipeline not initialized."}

    try:
        images = []
        for img_str in image_strs:
            img_bytes = base64.b64decode(img_str)
            image = Image.open(BytesIO(img_bytes)).convert("RGB")
            images.append(image)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
                + [{"type": "image", "image": img} for img in images],
            },
        ]

        output = pipe(
            text=messages,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        if (
            isinstance(output, list)
            and output
            and isinstance(output[0].get("generated_text"), list)
        ):
            generated_text = output[0]["generated_text"]
            if generated_text:
                response = generated_text[-1].get("content", "").strip()
                return {"response": response}

        return {"response": "No response generated"}

    except Exception as e:
        return {"error": f"An error occurred during VQA: {str(e)}"}

# Encapsulated server startup logic into a function
def start_server(host="0.0.0.0", port=8000):
    """
    Initializes the MedGemma pipeline and starts the FastMCP server.
    """
    initialize_medgemma_pipeline()
    print(f"Starting MedGemma MCP server at http://{host}:{port}")
    mcp.run(transport="http", host=host, port=port)

if __name__ == "__main__":
    """
    Entry point for running the MedGemma MCP server directly.
    """
    # Call the new server-starting function
    start_server()