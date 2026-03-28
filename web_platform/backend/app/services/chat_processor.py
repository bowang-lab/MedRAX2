"""
Chat Processor Service

Handles chat message processing with tool execution tracking and memory persistence.
Inspired by the old ChatInterface but integrated with new architecture.
"""

import asyncio
import base64
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Optional, Dict, Any, List
from sqlalchemy.orm import Session

from ..models.message import Message
from ..models.scan import Scan
from ..models.tool_execution import ToolExecution, ToolExecutionLog, ToolExecutionResult
from ..utils.file_utils import filesystem_path_from_display_url
from ..utils.logging_config import logger
from ..config import resolve_upload_dir, resolve_generated_dir
# from .image_registry import image_registry  # TODO: Re-enable when wrapper is fixed


class ChatProcessor:
    """
    Processes chat messages with full tool execution tracking and memory persistence.
    
    Features:
    - Request ID tracking to group tool executions
    - Tool execution history with image path tracking
    - Memory persistence via LangGraph checkpointer
    - Real-time SSE event streaming
    """
    
    def __init__(self, agent, db: Session, chat_id: str, tool_target_message_id: str | None = None):
        """
        Initialize chat processor.
        
        Args:
            agent: MedRAX Agent instance with tools
            db: Database session
            chat_id: Chat ID for this conversation
        """
        self.agent = agent
        self.db = db
        self.chat_id = chat_id
        self.request_id = None  # Set when processing message
        # If provided, all tool executions will be attached to this message id instead of the triggering user message
        self.tool_target_message_id = tool_target_message_id

    def _resolve_image_path(self, scan: Scan) -> str:
        """
        Choose the best path for encoding an image to base64.
        Prefer a display-ready path for DICOMs if available.
        """
        if scan.file_type and scan.file_type.lower() in {"dcm", "dicom"} and scan.display_path:
            candidate = filesystem_path_from_display_url(scan.display_path)
            if candidate.exists():
                return str(candidate)
        return scan.file_path

    def _infer_mime_type(self, path: str) -> str:
        """Infer MIME type for data URI based on file extension."""
        ext = Path(path).suffix.lower()
        if ext in {".png"}:
            return "image/png"
        if ext in {".jpg", ".jpeg"}:
            return "image/jpeg"
        if ext in {".gif"}:
            return "image/gif"
        return "application/octet-stream"

    def _to_display_path(self, path: str) -> str:
        """Convert file paths to URLs under /medrax/uploads/ or /medrax/generated/."""
        if not path:
            return path
        p = path.strip()
        if p.startswith("/medrax/"):
            return p
        if p.startswith("/uploads/"):
            return f"/medrax/uploads/{p[len('/uploads/'):]}"
        if p.startswith("/temp/"):
            return f"/medrax/generated/{p[len('/temp/'):]}"
        if p.startswith("uploads/"):
            return f"/medrax/{p}"
        if p.startswith("temp/"):
            return f"/medrax/generated/{p[len('temp/'):]}"
        upload_root = resolve_upload_dir()
        gen_root = resolve_generated_dir()
        try:
            abs_p = Path(p).expanduser().resolve()
            if abs_p.is_relative_to(upload_root):
                rel = abs_p.relative_to(upload_root)
                return f"/medrax/uploads/{rel.as_posix()}"
            if abs_p.is_relative_to(gen_root):
                rel = abs_p.relative_to(gen_root)
                return f"/medrax/generated/{rel.as_posix()}"
        except (ValueError, OSError):
            pass
        return path
        
    async def process_message(
        self,
        message: Message,
        scan_ids: Optional[List[str]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process a message and yield SSE events.
        
        Args:
            message: User message to process
            scan_ids: Optional list of scan IDs attached to this message
            
        Yields:
            SSE events as dictionaries
        """
        # Generate unique request ID for this analysis
        self.request_id = str(uuid.uuid4())
        message.request_id = self.request_id
        self.db.flush()
        
        logger.info(f"processing_message message_id={message.id[:8]} request_id={self.request_id[:8]} chat_id={self.chat_id[:8]} scan_ids={scan_ids}")
        
        # Get attached scans
        scans = []
        if scan_ids is not None and len(scan_ids) > 0:
            # Get specific scans attached to this message (scoped to this chat to avoid mixing patients/chats)
            scans = self.db.query(Scan).filter(
                Scan.id.in_(scan_ids),
                Scan.chat_id == self.chat_id
            ).all()
            logger.info(f"scans_retrieved count={len(scans)} requested={len(scan_ids)}")
            for scan in scans:
                logger.info(f"scan_details scan_id={scan.id[:8]} path={scan.file_path} exists={Path(scan.file_path).exists()}")
        else:
            # No scan_ids provided (None) OR explicitly empty list: fall back to latest scans in this chat
            # This preserves image context across requests within the same chat without mixing chats/patients.
            all_chat_scans = self.db.query(Scan).filter(
                Scan.chat_id == self.chat_id
            ).order_by(Scan.uploaded_at.desc()).all()
            
            if all_chat_scans:
                scans = all_chat_scans
                logger.info(f"using_chat_scans count={len(scans)} from_chat_history")
                for scan in scans:
                    logger.info(f"chat_scan_details scan_id={scan.id[:8]} path={scan.file_path} exists={Path(scan.file_path).exists()}")
            else:
                logger.info("no_scan_ids_and_no_chat_scans")
        
        # Build messages for agent
        agent_messages = []
        
        # Add image paths and images if scans attached
        if scans:
            scan_paths = [scan.file_path for scan in scans]
            
            # Since wrapper is temporarily disabled, use actual paths
            # TODO: Re-enable this when wrapper is fixed
            # image_mapping = image_registry.register_images(self.request_id, scan_paths)
            
            # Create a clear message about available images with actual paths
            image_context = (
                f"[Image Context] The user has uploaded {len(scans)} medical image(s).\n"
                f"When calling tools that require image paths, use these exact paths:\n\n"
            )
            
            # Show the actual paths that the LLM should use
            for i, scan in enumerate(scans, 1):
                filename = Path(scan.file_path).name
                image_context += f"  • Image {i}: {scan.file_path}\n"
            
            # Add clear instructions to prevent path corruption
            image_context += (
                "\nIMPORTANT: Always use the EXACT file paths shown above when calling tools. "
                "Copy the entire path exactly as shown. Do NOT modify or abbreviate the paths."
            )
            
            logger.info(f"adding_image_context scans_count={len(scans)}")
            logger.debug(f"image_context_message: {image_context.strip()}")
            
            agent_messages.append({
                "role": "user",
                "content": image_context.strip()
            })
            
            # Also send the actual images for visual analysis
            images_encoded = 0
            for scan in scans:
                try:
                    image_path = self._resolve_image_path(scan)
                    with open(image_path, "rb") as f:
                        img_bytes = f.read()
                        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
                    mime_type = self._infer_mime_type(image_path)
                    agent_messages.append({
                        "role": "user",
                        "content": [{
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{img_base64}"}
                        }]
                    })
                    images_encoded += 1
                    logger.info(f"image_encoded scan_id={scan.id[:8]} path={image_path} size_bytes={len(img_bytes)}")
                except Exception as e:
                    logger.error(f"image_encoding_error scan_id={scan.id} path={scan.file_path} error={str(e)}")
            logger.info(f"images_encoded_total count={images_encoded}/{len(scans)}")
        else:
            logger.info("no_scans_to_attach_to_agent_messages")
        
        # Add user message
        agent_messages.append({
            "role": "user",
            "content": [{"type": "text", "text": message.content}]
        })
        
        yield {
            "type": "status",
            "message": "Processing message...",
            "request_id": self.request_id
        }
        
        try:
            config = {"configurable": {"thread_id": self.chat_id}}
            
            async for event in self.agent.workflow.astream(
                {"messages": agent_messages},
                config
            ):
                if isinstance(event, dict):
                    if "agent" in event:
                        messages = event["agent"]["messages"]
                        if messages and len(messages) > 0:
                            content = messages[-1].content
                            if content:
                                # Handle both string and list content
                                # LangChain messages can have content as string or list of content blocks
                                if isinstance(content, list):
                                    # Extract text from content blocks
                                    text_parts = []
                                    for block in content:
                                        if isinstance(block, dict) and block.get("type") == "text":
                                            text_parts.append(block.get("text", ""))
                                        elif isinstance(block, str):
                                            text_parts.append(block)
                                    content_str = "".join(text_parts)
                                else:
                                    content_str = str(content)
                                
                                if content_str:
                                    yield {
                                        "type": "content_chunk",
                                        "data": {"content": content_str}
                                    }
                    
                    elif "tools" in event:
                        for tool_message in event["tools"]["messages"]:
                            async for tool_event in self._process_tool_execution(
                                tool_message,
                                message,
                                [scan.display_path or scan.file_path for scan in scans]
                            ):
                                yield tool_event
            
            yield {
                "type": "complete",
                "message": "Message processed successfully"
            }
            
        except Exception as e:
            logger.error(f"message_processing_error message_id={message.id[:8]} error={str(e)}", exc_info=True)
            yield {
                "type": "error",
                "message": f"Error: {str(e)}"
            }
    
    async def _process_tool_execution(
        self,
        tool_message: Any,
        message: Message,
        image_paths: List[str]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process a tool execution and track it in database.
        
        Args:
            tool_message: Tool message from agent
            message: User message that triggered this
            image_paths: Image paths used in this execution (file paths)
            
        Yields:
            SSE events for tool execution
        """
        tool_name = tool_message.name
        
        # Convert file paths to display paths for frontend
        display_paths = [self._to_display_path(path) for path in image_paths]
        
        # Create tool execution record with display paths
        execution = ToolExecution(
            message_id=self.tool_target_message_id or message.id,
            request_id=self.request_id,
            tool_name=tool_name,
            status="running",
            image_paths=display_paths
        )
        self.db.add(execution)
        self.db.flush()
        # Commit early so other API requests (sidebar fetch) can see the running execution
        try:
            self.db.commit()
        except Exception:
            # In streaming contexts, commit may fail transiently; safe to proceed, later calls will attempt again
            self.db.rollback()
        
        # Yield tool start event
        yield {
            "type": "tool_start",
            "data": {
                "tool_name": tool_name,
                "execution_id": execution.id,
                "message_id": execution.message_id,
            }
        }
        
        try:
            result_data = None
            metadata = {}
            
            if tool_message.content:
                try:
                    import ast
                    parsed = ast.literal_eval(str(tool_message.content))
                    # Handle both tuple and list with 2+ elements (result, metadata)
                    if isinstance(parsed, (tuple, list)) and len(parsed) >= 2:
                        result_data, metadata = parsed[0], parsed[1]
                    elif isinstance(parsed, dict):
                        result_data = parsed
                    else:
                        result_data = {"raw": str(parsed)}
                except (ValueError, SyntaxError, TypeError):
                    result_data = {"raw": str(tool_message.content)}
            
            # Create result record
            if result_data is not None:
                exec_result = ToolExecutionResult(
                    execution_id=execution.id,
                    result_data=result_data if isinstance(result_data, dict) else {"raw": str(result_data)},
                    result_metadata=metadata if isinstance(metadata, dict) else {}
                )
                self.db.add(exec_result)
                
                # Extract generated image paths from tool result
                # Tools may return: image_path, segmentation_image_path, visualization_path, etc.
                generated_images = []
                for key, value in (result_data.items() if isinstance(result_data, dict) else []):
                    if 'image_path' in key.lower() or 'visualization' in key.lower():
                        if isinstance(value, str) and value:
                            generated_images.append(value)
                
                # Update execution with generated images (convert to display paths)
                if generated_images:
                    generated_display_paths = [self._to_display_path(path) for path in generated_images]
                    execution.image_paths = display_paths + generated_display_paths
            
            # Update execution status
            execution.status = "completed"
            execution.completed_at = datetime.utcnow()
            self.db.flush()
            try:
                self.db.commit()
            except Exception:
                self.db.rollback()
            
            # Yield tool output event (for UI panels)
            if result_data is not None:
                yield {
                    "type": "tool_output",
                    "data": {
                        "tool_name": tool_name,
                        "execution_id": execution.id,
                        "message_id": execution.message_id,
                        "result": result_data,
                        "metadata": metadata,
                        "image_paths": execution.image_paths or [],
                    },
                }

            # Yield tool completion
            yield {
                "type": "tool_done",
                "data": {
                    "tool_name": tool_name,
                    "execution_id": execution.id,
                    "message_id": execution.message_id,
                }
            }
            
            logger.info(f"tool_execution_tracked execution_id={execution.id[:8]} tool_name={tool_name} request_id={self.request_id[:8]}")
            
        except Exception as e:
            # Mark as failed
            execution.status = "failed"
            execution.completed_at = datetime.utcnow()
            
            # Log error
            log = ToolExecutionLog(
                execution_id=execution.id,
                log_level="error",
                message=str(e)
            )
            self.db.add(log)
            self.db.flush()
            try:
                self.db.commit()
            except Exception:
                self.db.rollback()
            
            # Yield error event
            yield {
                "type": "tool_error",
                "data": {
                    "tool_name": tool_name,
                    "execution_id": execution.id,
                    "message_id": execution.message_id,
                    "error": str(e)
                }
            }
            
            logger.error(f"tool_execution_error execution_id={execution.id[:8]} tool_name={tool_name} error={str(e)}")
    
    def get_tool_history(
        self,
        filter_by_request: Optional[str] = None,
        filter_by_image: Optional[str] = None,
        latest_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get tool execution history for this chat.
        
        Args:
            filter_by_request: Only return executions from this request
            filter_by_image: Only return executions that used this image
            latest_only: Only return latest execution per tool
            
        Returns:
            List of execution records
        """
        query = self.db.query(ToolExecution).join(Message).filter(
            Message.chat_id == self.chat_id
        )
        
        if filter_by_request:
            query = query.filter(ToolExecution.request_id == filter_by_request)
        
        if filter_by_image:
            # Filter by image path in JSON array
            query = query.filter(ToolExecution.image_paths.contains(filter_by_image))
        
        query = query.order_by(ToolExecution.started_at.desc())
        
        executions = query.all()
        
        # If latest_only, keep only most recent per tool
        if latest_only:
            seen_tools = set()
            filtered = []
            for execution in executions:
                if execution.tool_name not in seen_tools:
                    filtered.append(execution)
                    seen_tools.add(execution.tool_name)
            executions = filtered
        
        # Convert to dict format
        history = []
        for execution in executions:
            record = {
                "execution_id": execution.id,
                "request_id": execution.request_id,
                "tool_name": execution.tool_name,
                "status": execution.status,
                "started_at": execution.started_at.isoformat() if execution.started_at else None,
                "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                "image_paths": execution.image_paths or [],
                "result": None
            }
            
            # Add result if available
            if execution.result:
                record["result"] = execution.result.result_data
                record["metadata"] = execution.result.result_metadata
            
            history.append(record)
        
        return history
