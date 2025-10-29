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
from ..utils.logging_config import logger


class ChatProcessor:
    """
    Processes chat messages with full tool execution tracking and memory persistence.
    
    Features:
    - Request ID tracking to group tool executions
    - Tool execution history with image path tracking
    - Memory persistence via LangGraph checkpointer
    - Real-time SSE event streaming
    """
    
    def __init__(self, agent, db: Session, chat_id: str):
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
        
        logger.info(f"processing_message message_id={message.id[:8]} request_id={self.request_id[:8]} chat_id={self.chat_id[:8]}")
        
        # Get attached scans
        scans = []
        if scan_ids:
            scans = self.db.query(Scan).filter(
                Scan.id.in_(scan_ids),
                Scan.chat_id == self.chat_id
            ).all()
        
        # Build messages for agent
        agent_messages = []
        
        # Add image paths if scans attached
        if scans:
            scan_paths = [scan.file_path for scan in scans]
            agent_messages.append({
                "role": "user",
                "content": f"image_paths: {', '.join(scan_paths)}"
            })
            
            for scan in scans:
                try:
                    with open(scan.file_path, "rb") as f:
                        img_base64 = base64.b64encode(f.read()).decode("utf-8")
                    agent_messages.append({
                        "role": "user",
                        "content": [{
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                        }]
                    })
                except Exception as e:
                    logger.error(f"image_encoding_error scan_id={scan.id} error={str(e)}")
        
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
                        content = event["agent"]["messages"][-1].content
                        if content:
                            yield {
                                "type": "content_chunk",
                                "data": {"content": content}
                            }
                    
                    elif "tools" in event:
                        for tool_message in event["tools"]["messages"]:
                            async for tool_event in self._process_tool_execution(
                                tool_message,
                                message,
                                [scan.file_path for scan in scans]
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
            image_paths: Image paths used in this execution
            
        Yields:
            SSE events for tool execution
        """
        tool_name = tool_message.name
        
        # Create tool execution record
        execution = ToolExecution(
            message_id=message.id,
            request_id=self.request_id,
            tool_name=tool_name,
            status="running",
            image_paths=image_paths
        )
        self.db.add(execution)
        self.db.flush()
        
        # Yield tool start event
        yield {
            "type": "tool_start",
            "data": {
                "tool_name": tool_name,
                "execution_id": execution.id
            }
        }
        
        try:
            result_data = None
            metadata = {}
            
            if tool_message.content:
                try:
                    import ast
                    parsed = ast.literal_eval(str(tool_message.content))
                    if isinstance(parsed, tuple) and len(parsed) >= 2:
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
                
                # Update execution with generated images
                if generated_images:
                    execution.image_paths = image_paths + generated_images
            
            # Update execution status
            execution.status = "completed"
            execution.completed_at = datetime.utcnow()
            self.db.flush()
            
            # Yield tool completion
            yield {
                "type": "tool_done",
                "data": {
                    "tool_name": tool_name,
                    "execution_id": execution.id
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
            
            # Yield error event
            yield {
                "type": "tool_error",
                "data": {
                    "tool_name": tool_name,
                    "execution_id": execution.id,
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

