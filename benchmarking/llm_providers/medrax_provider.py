"""MedRAX LLM provider implementation."""

import os
import time
import re
import json
import uuid
from collections import deque

from .base import LLMProvider, LLMRequest, LLMResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from medrax.rag.rag import RAGConfig
from main import initialize_agent


class MedRAXProvider(LLMProvider):
    """MedRAX LLM provider that uses the full MedRAX agent system."""

    def __init__(self, model_name: str, system_prompt: str, **kwargs):
        """Initialize MedRAX provider.

        Args:
            model_name (str): Base LLM model name (e.g., "gpt-4.1-2025-04-14")
            system_prompt (str): System prompt to use
            **kwargs: Additional configuration parameters
        """
        # Set provider name
        self.provider_name = "medrax"

        self.agent = None
        self.tools_dict = None
        # Capture configurable parameters (fallback to previous defaults)
        self.selected_tools = kwargs.get("tools")
        self.rag_config_data = kwargs.get("rag")
        self.model_dir = kwargs.get(
            "model_dir", "/hpf/projects/mkoziarski/alian/igem/MedRAX2/model-weights"
        )
        self.temp_dir = kwargs.get("temp_dir", "temp")
        self.load_in_8bit = kwargs.get("load_in_8bit", True)
        # self.prompt_file = kwargs.get("prompt_file", "benchmarking/system_prompts.txt")

        super().__init__(model_name, system_prompt, **kwargs)

    def _setup(self) -> None:
        """Set up MedRAX agent system."""
        try:
            print("Starting server...")

            # Use configured tools or fall back to previous defaults
            selected_tools = self.selected_tools or [
                "TorchXRayVisionClassifierTool",  # For classifying chest X-ray images using TorchXRayVision
                "ArcPlusClassifierTool",  # For advanced chest X-ray classification using ArcPlus
                "ChestXRayReportGeneratorTool",  # For generating medical reports from X-rays
                "XRayPhraseGroundingTool",  # For locating described features in X-rays
                "MedGemmaVQATool",  # Google MedGemma VQA tool
                # "MedicalRAGTool",  # For retrieval-augmented generation with medical knowledge
                # "WebBrowserTool",  # For web browsing and search capabilities
                # "DuckDuckGoSearchTool",  # For privacy-focused web search using DuckDuckGo
            ]

            # Use configured RAG settings or fall back to previous defaults
            rag_config_data = self.rag_config_data or {
                "model": "command-a-03-2025",
                "embedding_model": "embed-v4.0",
                "rerank_model": "rerank-v3.5",
                "temperature": 0.3,
                "pinecone_index_name": "medrax2",
                "chunk_size": 1500,
                "chunk_overlap": 300,
                "retriever_k": 3,
                "local_docs_dir": "rag_docs",
                "huggingface_datasets": ["VictorLJZ/medrax2"],
                "dataset_split": "train",
            }
            rag_config = (
                rag_config_data
                if isinstance(rag_config_data, RAGConfig)
                else RAGConfig(**rag_config_data)
            )

            # Prepare any additional model-specific kwargs
            model_kwargs = {}

            agent, tools_dict = initialize_agent(
                prompt_file=self.prompt_file,
                tools_to_use=selected_tools,
                model_dir=self.model_dir,
                temp_dir=self.temp_dir,
                device=os.getenv("MEDRAX_DEVICE", "cuda:0"),
                model=self.model_name,  # Change this to the model you want to use, e.g. gpt-4.1-2025-04-14, gemini-2.5-pro
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                model_kwargs=model_kwargs,
                rag_config=rag_config,
                system_prompt=self.prompt_name,
                load_in_8bit=self.load_in_8bit,
            )

            self.agent = agent
            self.tools_dict = tools_dict

            print(f"MedRAX agent initialized with tools: {list(self.tools_dict.keys())}")

        except Exception as e:
            print(f"Error initializing MedRAX agent: {e}")
            raise

    def _to_raw_string(self, value):
        """Best-effort conversion to a raw string representation."""
        if value is None:
            return None
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, default=str)
        except Exception:
            return str(value)

    def _safe_json_parse(self, value):
        """Best-effort JSON parsing for strings while preserving dict/list objects."""
        if value is None:
            return None
        if isinstance(value, (dict, list, int, float, bool)):
            return value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                return json.loads(text)
            except Exception:
                return None
        return None

    def _to_json_safe(self, value):
        """Convert values to JSON-serializable forms for result persistence."""
        try:
            json.dumps(value)
            return value
        except Exception:
            try:
                return json.loads(json.dumps(value, default=str))
            except Exception:
                return self._to_raw_string(value)

    def _extract_tool_requests(self, msg):
        """Extract tool calls from exactly one request source per AI message."""
        tool_calls_in_msg = getattr(msg, "tool_calls", None)
        if tool_calls_in_msg:
            return list(tool_calls_in_msg), "tool_calls"

        additional_kwargs = getattr(msg, "additional_kwargs", {}) or {}
        function_call = additional_kwargs.get("function_call")
        if function_call:
            return [function_call], "function_call"

        function_calls = additional_kwargs.get("function_calls")
        if function_calls:
            if isinstance(function_calls, list):
                return function_calls, "function_calls"
            return [function_calls], "function_calls"

        return [], None

    def _extract_tool_name_and_kwargs(self, call):
        """Normalize tool name and kwargs payload across call schemas."""
        if not isinstance(call, dict):
            return str(call), None, None

        tool_name = (
            call.get("name")
            or call.get("tool")
            or call.get("tool_name")
            or call.get("function")
            or ""
        )

        kwargs_payload = None
        if "args" in call:
            kwargs_payload = call.get("args")
        elif "arguments" in call:
            kwargs_payload = call.get("arguments")
        elif "kwargs" in call:
            kwargs_payload = call.get("kwargs")

        kwargs_raw = self._to_raw_string(kwargs_payload)
        kwargs_parsed = self._safe_json_parse(kwargs_payload)
        return tool_name, kwargs_raw, kwargs_parsed

    def _extract_error_from_payload(self, payload):
        """Recursively scan payload for error markers."""
        if isinstance(payload, dict):
            analysis_status = str(payload.get("analysis_status", "")).strip().lower()
            if analysis_status == "failed":
                return (
                    self._to_raw_string(payload.get("error"))
                    or self._to_raw_string(payload.get("error_details"))
                    or "analysis_status=failed"
                )

            for key in ("error", "error_details", "error_message", "exception", "traceback"):
                if payload.get(key):
                    return self._to_raw_string(payload.get(key))

            for value in payload.values():
                nested_error = self._extract_error_from_payload(value)
                if nested_error:
                    return nested_error

        if isinstance(payload, list):
            for item in payload:
                nested_error = self._extract_error_from_payload(item)
                if nested_error:
                    return nested_error

        return None

    def _extract_error_from_raw(self, output_raw):
        """Scan raw output text for obvious failure indicators."""
        if not output_raw:
            return None

        error_tokens = [
            "error",
            "exception",
            "traceback",
            "analysis_status\": \"failed",
            "analysis_status=failed",
            "no_valid_response_generated",
            "status_code",
        ]
        raw_text = str(output_raw)
        lowered = raw_text.lower()
        if any(token in lowered for token in error_tokens):
            for line in raw_text.splitlines():
                lowered_line = line.lower()
                if any(token in lowered_line for token in error_tokens):
                    return line.strip()[:2000]
            return raw_text[:2000]
        return None

    def _infer_tool_response_status(self, output_raw, output_parsed):
        """Infer tool execution status and extract error if present."""
        parsed_error = self._extract_error_from_payload(output_parsed)
        if parsed_error:
            return "failed", parsed_error

        raw_error = self._extract_error_from_raw(output_raw)
        if raw_error:
            return "failed", raw_error

        return "completed", None

    def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate response using MedRAX agent.

        Args:
            request (LLMRequest): The request containing text, images, and parameters

        Returns:
            LLMResponse: The response from MedRAX agent
        """
        start_time = time.time()

        if self.agent is None:
            return LLMResponse(
                content="Error: MedRAX agent not initialized", duration=time.time() - start_time
            )

        try:
            # Build messages for the agent
            messages = []
            thread_id = str(uuid.uuid4())  # Globally unique thread ID (thread-safe)

            if request.images:
                # Build multimodal content with text and images
                content = [{"type": "text", "text": request.text}]

                # Validate image paths
                valid_images = self._validate_image_paths(request.images)
                print(f"Processing {len(valid_images)} images")

                # Add image paths for tools
                for image_path in valid_images:
                    content.append({"type": "text", "text": f"image_path: {image_path}"})

                # Add image content for multimodal LLM
                for image_path in valid_images:
                    try:
                        img_base64 = self._encode_image(image_path)
                        mime_type = self._get_image_mime_type(image_path)

                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{img_base64}"},
                            }
                        )
                    except Exception as e:
                        print(f"ERROR: Image encoding failed for {image_path}: {e}")
                        raise

                # Create single multimodal message
                messages.append(HumanMessage(content=content))

            else:
                # If no images, add text as simple string
                messages.append(HumanMessage(content=request.text))

            # Run the agent with proper message type handling
            final_response = ""
            chunk_history = []
            llm_requests = 0
            token_usage = {"input": 0, "output": 0, "reasoning": 0}
            tool_calls = []
            tool_execution_trace = []
            pending_tool_request_indices = deque()
            trace_sequence_index = 0

            for chunk_index, chunk in enumerate(
                self.agent.workflow.stream(
                    {"messages": messages},
                    {"configurable": {"thread_id": thread_id}},
                    stream_mode="updates",
                )
            ):
                if not isinstance(chunk, dict):
                    continue

                for node_name, node_output in chunk.items():
                    # Log chunk and get serializable version
                    serializable_chunk = self._log_chunk(node_output, node_name)
                    chunk_history.append(serializable_chunk)

                    if "messages" not in node_output:
                        continue

                    for message_index, msg in enumerate(node_output["messages"]):
                        if isinstance(msg, AIMessage):
                            llm_requests += 1

                            usage_meta = getattr(msg, "usage_metadata", {}) or {}
                            input_tokens = usage_meta.get("input_tokens")
                            if input_tokens is None:
                                input_tokens = usage_meta.get("prompt_tokens", 0)
                            output_tokens = usage_meta.get("output_tokens")
                            if output_tokens is None:
                                output_tokens = usage_meta.get("completion_tokens", 0)
                            token_usage["input"] += input_tokens or 0
                            token_usage["output"] += output_tokens or 0
                            token_usage["reasoning"] += usage_meta.get("reasoning_tokens", 0) or 0

                            # Use exactly one source per AI message for tool-call extraction.
                            collected_calls, request_source = self._extract_tool_requests(msg)
                            additional_kwargs = getattr(msg, "additional_kwargs", {}) or {}

                            if collected_calls:
                                for call in collected_calls:
                                    tool_name, kwargs_raw, kwargs_parsed = (
                                        self._extract_tool_name_and_kwargs(call)
                                    )
                                    tool_calls.append({"node": node_name, "tool": tool_name})

                                    trace_entry = {
                                        "sequence_index": trace_sequence_index,
                                        "node": node_name,
                                        "tool": tool_name,
                                        "request_source": request_source,
                                        "request": {
                                            "kwargs_raw": kwargs_raw,
                                            "kwargs_parsed": kwargs_parsed,
                                            "additional_kwargs": self._to_json_safe(
                                                additional_kwargs
                                            ),
                                        },
                                        "response": {
                                            "output_raw": None,
                                            "output_parsed": None,
                                            "status": None,
                                            "error": None,
                                        },
                                        "stream_indices": {
                                            "request_chunk_index": chunk_index,
                                            "request_message_index": message_index,
                                            "response_chunk_index": None,
                                            "response_message_index": None,
                                        },
                                    }
                                    tool_execution_trace.append(trace_entry)
                                    pending_tool_request_indices.append(len(tool_execution_trace) - 1)
                                    trace_sequence_index += 1

                            # Handle case where content is a list
                            content = msg.content or ""
                            if isinstance(content, list):
                                content = self._to_raw_string(content) or ""
                            elif not isinstance(content, str):
                                content = self._to_raw_string(content) or ""
                            # Clean up the content (remove temp paths, etc.)
                            if str(content).strip():
                                final_response = re.sub(r"temp/[^\s]*", "", content).strip()
                        elif isinstance(msg, ToolMessage) or type(msg).__name__ == "ToolMessage":
                            output_content = getattr(msg, "content", None)
                            output_raw = self._to_raw_string(output_content)
                            output_parsed = self._safe_json_parse(output_content)
                            status, error = self._infer_tool_response_status(output_raw, output_parsed)

                            if pending_tool_request_indices:
                                matched_trace_idx = pending_tool_request_indices.popleft()
                                matched_entry = tool_execution_trace[matched_trace_idx]
                                matched_entry["response"] = {
                                    "output_raw": output_raw,
                                    "output_parsed": self._to_json_safe(output_parsed),
                                    "status": status,
                                    "error": error,
                                }
                                matched_entry["stream_indices"]["response_chunk_index"] = chunk_index
                                matched_entry["stream_indices"]["response_message_index"] = (
                                    message_index
                                )
                            else:
                                orphan_tool_name = getattr(msg, "name", None) or "unknown"
                                tool_execution_trace.append(
                                    {
                                        "sequence_index": trace_sequence_index,
                                        "node": node_name,
                                        "tool": orphan_tool_name,
                                        "request_source": None,
                                        "request": {
                                            "kwargs_raw": None,
                                            "kwargs_parsed": None,
                                            "additional_kwargs": None,
                                        },
                                        "response": {
                                            "output_raw": output_raw,
                                            "output_parsed": self._to_json_safe(output_parsed),
                                            "status": "orphan_output",
                                            "error": error,
                                        },
                                        "stream_indices": {
                                            "request_chunk_index": None,
                                            "request_message_index": None,
                                            "response_chunk_index": chunk_index,
                                            "response_message_index": message_index,
                                        },
                                    }
                                )
                                trace_sequence_index += 1

            # Mark any pending calls that never produced a tool output.
            while pending_tool_request_indices:
                trace_idx = pending_tool_request_indices.popleft()
                pending_entry = tool_execution_trace[trace_idx]
                pending_entry["response"]["status"] = "unmatched_call"
                pending_entry["response"]["error"] = (
                    "No tool output was observed for this tool request."
                )

            # Determine the final response
            if final_response:
                response_content = final_response
            else:
                # Fallback if no LLM response was received
                response_content = "No response generated"

            duration = time.time() - start_time

            return LLMResponse(
                content=response_content,
                usage={
                    "agent_tools": list(self.tools_dict.keys()),
                    "llm_requests": llm_requests,
                    "tokens": token_usage,
                    "tool_calls": tool_calls,
                },
                duration=duration,
                chunk_history=chunk_history,
                tool_execution_trace=tool_execution_trace,
            )

        except Exception as e:
            print(f"ERROR: MedRAX agent failed: {e}")
            return LLMResponse(content=f"Error: {str(e)}", duration=time.time() - start_time)

    def _log_chunk(self, chunk: dict, node_name: str) -> dict:
        """Log and process a chunk from the agent workflow.

        Args:
            chunk (dict): The chunk data from the agent workflow
            node_name (str): Name of the node that produced the chunk

        Returns:
            dict: Serializable version of the chunk for debugging
        """
        # Log every chunk for debugging
        print(f"Chunk from node '{node_name}': {type(chunk)}")

        # Store serializable version of chunk for debugging
        serializable_chunk = {
            "node_name": node_name,
            "node_type": type(chunk).__name__,
        }

        # Log messages in this chunk
        if "messages" in chunk and isinstance(chunk, dict):
            chunk_messages = []
            for msg in chunk["messages"]:
                msg_info = {
                    "type": type(msg).__name__,
                    "content": str(msg.content) if hasattr(msg, "content") else str(msg),
                }
                
                # Extract response metadata (reasoning/thinking traces)
                if hasattr(msg, "response_metadata") and msg.response_metadata:
                    try:
                        msg_info["response_metadata"] = dict(msg.response_metadata)
                        
                        # Extract specific reasoning fields for easier access
                        # Gemini 2.0 Flash Thinking uses 'thoughts'
                        if "thoughts" in msg.response_metadata:
                            msg_info["thinking"] = msg.response_metadata["thoughts"]

                        # DeepSeek-R1 and similar models use 'reasoning_content'
                        if "reasoning_content" in msg.response_metadata:
                            msg_info["reasoning"] = msg.response_metadata["reasoning_content"]
                        
                        # Some models expose thinking in other fields
                        if "extended_thinking" in msg.response_metadata:
                            msg_info["extended_thinking"] = msg.response_metadata[
                                "extended_thinking"
                            ]
                    except Exception as e:
                        print(f"Warning: Could not serialize response_metadata: {e}")
                
                # Extract usage metadata (reasoning tokens for o1/o3 models)
                if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                    try:
                        msg_info["usage_metadata"] = dict(msg.usage_metadata)

                        # Highlight reasoning tokens if present
                        if (
                            isinstance(msg.usage_metadata, dict)
                            and "reasoning_tokens" in msg.usage_metadata
                        ):
                            msg_info["reasoning_tokens"] = msg.usage_metadata["reasoning_tokens"]
                    except Exception as e:
                        print(f"Warning: Could not serialize usage_metadata: {e}")
                
                # Extract additional kwargs (some models put reasoning here)
                if hasattr(msg, "additional_kwargs") and msg.additional_kwargs:
                    try:
                        # Filter for reasoning-related fields
                        reasoning_kwargs = {}
                        for key in ["thinking", "reasoning", "thoughts", "chain_of_thought"]:
                            if key in msg.additional_kwargs:
                                reasoning_kwargs[key] = msg.additional_kwargs[key]

                        if reasoning_kwargs:
                            msg_info["additional_reasoning"] = reasoning_kwargs

                        # Include full additional_kwargs for completeness (may contain other useful info)
                        msg_info["additional_kwargs"] = dict(msg.additional_kwargs)
                    except Exception as e:
                        print(f"Warning: Could not serialize additional_kwargs: {e}")

                chunk_messages.append(msg_info)

                # Enhanced logging for debugging
                log_msg = f"Message in chunk: type={msg_info['type']}"
                if "thinking" in msg_info:
                    log_msg += f", has_thinking=True (length={len(str(msg_info['thinking']))})"
                if "reasoning" in msg_info:
                    log_msg += f", has_reasoning=True (length={len(str(msg_info['reasoning']))})"
                if "reasoning_tokens" in msg_info:
                    log_msg += f", reasoning_tokens={msg_info['reasoning_tokens']}"
                print(log_msg)

                # Log the actual content
                content = msg_info.get("content", "")
                if content:
                    # Truncate very long content for readability (show first 500 chars)
                    content_str = str(content)
                    if len(content_str) > 500:
                        print(f"  Content (truncated): {content_str[:500]}...")
                    else:
                        print(f"  Content: {content_str}")

            serializable_chunk["messages"] = chunk_messages

        return serializable_chunk
