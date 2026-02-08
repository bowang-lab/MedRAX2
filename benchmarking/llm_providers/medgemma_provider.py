"""MedGemma LLM provider implementation using the MedGemma FastAPI service."""

import os
import time
import httpx
from tenacity import retry, wait_exponential, stop_after_attempt

from .base import LLMProvider, LLMRequest, LLMResponse


class MedGemmaProvider(LLMProvider):
    """MedGemma LLM provider that communicates with the MedGemma FastAPI service.
    
    This provider wraps Google's MedGemma 4B model as an LLMProvider for benchmarking.
    It communicates with a running MedGemma FastAPI service on localhost:8002.
    
    MedGemma is a specialized multimodal AI model trained on medical images and text.
    It provides expert-level analysis for chest X-rays, dermatology images,
    ophthalmology images, and histopathology slides.
    
    Requirements:
        - MedGemma FastAPI service must be running on the configured API URL
        - Default URL: http://localhost:8002
        - Can be overridden via MEDGEMMA_API_URL environment variable
    """

    def __init__(self, model_name: str, system_prompt: str, **kwargs):
        """Initialize MedGemma provider.
        
        Args:
            model_name (str): Model name (for consistency with other providers)
            system_prompt (str): System prompt identifier to load from file
            **kwargs: Additional configuration parameters
                - api_url: URL of the MedGemma FastAPI service
                - max_new_tokens: Maximum tokens to generate (default: 300)
        """
        self.provider_name = "medgemma"
        self.api_url = os.getenv("MEDGEMMA_API_URL", "http://0.0.0.0:8002")
        self.client = None
        
        # Call parent constructor
        super().__init__(model_name, system_prompt, **kwargs)

    def _setup(self) -> None:
        """Set up httpx client for communicating with MedGemma API."""
        # Create httpx client with reasonable timeouts
        timeout_config = httpx.Timeout(
            timeout=300.0,  # 5 minutes for inference
            connect=10.0    # 10 seconds to establish connection
        )
        self.client = httpx.Client(timeout=timeout_config)

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
    def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate response using MedGemma API.
        
        Args:
            request (LLMRequest): The request containing text, images, and parameters
            
        Returns:
            LLMResponse: The response from MedGemma
        """
        start_time = time.time()
        
        if self.client is None:
            return LLMResponse(
                content="Error: MedGemma client not initialized",
                duration=time.time() - start_time
            )
        
        try:
            # Validate and prepare images
            if not request.images:
                return LLMResponse(
                    content="Error: MedGemma requires at least one image",
                    duration=time.time() - start_time
                )
            
            valid_images = self._validate_image_paths(request.images)
            if not valid_images:
                return LLMResponse(
                    content="Error: No valid image paths provided",
                    duration=time.time() - start_time
                )
            
            # Prepare multipart form data
            files_to_send = []
            for image_path in valid_images:
                try:
                    # Read image file
                    with open(image_path, "rb") as f:
                        image_data = f.read()
                    
                    # Detect correct MIME type based on file extension
                    mime_type = self._get_image_mime_type(image_path)

                    # Add to files list
                    files_to_send.append(
                        ("images", (os.path.basename(image_path), image_data, mime_type))
                    )
                except Exception as e:
                    print(f"Error reading image {image_path}: {e}")
                    continue
            
            if not files_to_send:
                return LLMResponse(
                    content="Error: Failed to read any image files",
                    duration=time.time() - start_time
                )
            
            # Use system_prompt if provided, otherwise use default
            system_prompt_text = self.system_prompt if self.system_prompt else "You are an expert radiologist who is able to analyze radiological images at any resolution."
            
            # Prepare form data
            data = {
                "prompt": request.text,
                "system_prompt": system_prompt_text,
                "max_new_tokens": self.max_tokens,
            }
            
            # Make API request
            response = self.client.post(
                f"{self.api_url}/analyze-images/",
                data=data,
                files=files_to_send,
            )
            
            # Check for errors
            response.raise_for_status()
            
            # Parse response
            response_data = response.json()
            content = response_data.get("response", "")
            
            # record duration
            duration = time.time() - start_time

            # return response object
            return LLMResponse(
                content=content,
                usage=None,
                duration=duration
            )
            
        except httpx.TimeoutException as e:
            duration = time.time() - start_time
            error_msg = f"MedGemma API request timed out after {duration:.1f}s. The server might be overloaded or the model is taking too long to process."
            print(f"Error: {error_msg}")
            return LLMResponse(
                content=f"Error: {error_msg}",
                duration=duration
            )
            
        except httpx.ConnectError as e:
            duration = time.time() - start_time
            error_msg = f"Could not connect to MedGemma API at {self.api_url}. Please ensure the service is running."
            print(f"Error: {error_msg}")
            return LLMResponse(
                content=f"Error: {error_msg}",
                duration=duration
            )
            
        except httpx.HTTPStatusError as e:
            duration = time.time() - start_time
            error_msg = f"MedGemma API returned error {e.response.status_code}: {e.response.text}"
            print(f"Error: {error_msg}")
            return LLMResponse(
                content=f"Error: {error_msg}",
                duration=duration
            )
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Unexpected error calling MedGemma API: {str(e)}"
            print(f"Error: {error_msg}")
            return LLMResponse(
                content=f"Error: {error_msg}",
                duration=duration
            )

    def test_connection(self) -> bool:
        """Test the connection to the MedGemma API service.
        
        Returns:
            bool: True if connection is successful and service is responding
        """
        try:
            # Try to access the API health endpoint
            response = self.client.get(f"{self.api_url}/health")
            return response.json().get("status") == "ok"
        except Exception as e:
            print(f"MedGemma connection test failed: {e}")
            return False
    
    def __del__(self):
        """Clean up httpx client on deletion."""
        if self.client is not None:
            self.client.close()


