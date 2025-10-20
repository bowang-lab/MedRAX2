"""
Comprehensive Agent Integration Tests

These tests verify the ACTUAL agent functionality with real tools,
memory, and context. They should catch issues like:
- Missing dependencies (langgraph)
- Logger syntax errors
- Memory/context persistence
- Tool execution
- Response generation
"""

import pytest
import json
from fastapi.testclient import TestClient


class TestAgentMemoryAndContext:
    """Test that the agent remembers context across messages."""
    
    def test_remembers_name_in_conversation(self, client: TestClient, auth_headers, test_chat):
        """
        CRITICAL TEST: Agent should remember information from previous messages.
        This tests the ACTUAL memory/context feature.
        """
        # Load a tool first (required for agent)
        from app.services.tool_manager import tool_manager as tm
        tm.load_tool('dicom_processor')
        tm.load_tool_in_background('dicom_processor')
        
        # First message: Tell the agent your name
        response1 = client.post(
            f"/api/chats/{test_chat.id}/stream",
            headers=auth_headers,
            json={
                "content": "My name is Alankrit",
                "scan_ids": []
            }
        )
        
        assert response1.status_code == 200
        
        # Parse response
        events1 = []
        for line in response1.iter_lines():
            line_str = line.decode('utf-8') if isinstance(line, bytes) else line
            if line_str.startswith('data: '):
                try:
                    data_str = line_str[6:].strip()
                    if data_str:
                        event_data = json.loads(data_str)
                        if event_data:
                            events1.append(event_data)
                except json.JSONDecodeError:
                    pass
        
        # Should get some response
        assert len(events1) > 0, "Should have received events from first message"
        
        # Check no errors
        has_error = any(e.get('type') == 'error' for e in events1 if e)
        assert not has_error, "First message should not have errors"
        
        # Second message: Ask for the name
        response2 = client.post(
            f"/api/chats/{test_chat.id}/stream",
            headers=auth_headers,
            json={
                "content": "What is my name?",
                "scan_ids": []
            }
        )
        
        assert response2.status_code == 200
        
        # Parse response
        events2 = []
        content_chunks = []
        for line in response2.iter_lines():
            line_str = line.decode('utf-8') if isinstance(line, bytes) else line
            if line_str.startswith('data: '):
                try:
                    data_str = line_str[6:].strip()
                    if data_str:
                        event_data = json.loads(data_str)
                        if event_data:
                            events2.append(event_data)
                            if event_data.get('type') == 'content_chunk':
                                content_chunks.append(event_data.get('data', {}).get('content', ''))
                except json.JSONDecodeError:
                    pass
        
        # Should have received events
        assert len(events2) > 0, "Should have received events from second message"
        
        # The response should mention "Alankrit"
        full_content = ''.join(content_chunks)
        assert 'Alankrit' in full_content or 'alankrit' in full_content.lower(), \
            f"Agent should remember the name 'Alankrit'. Got: {full_content}"
    
    def test_remembers_multiple_facts(self, client: TestClient, auth_headers, test_chat):
        """Test that agent remembers multiple pieces of information."""
        # Load tool
        from app.services.tool_manager import tool_manager as tm
        tm.load_tool('dicom_processor')
        tm.load_tool_in_background('dicom_processor')
        
        # Share multiple facts
        facts = [
            "I am 25 years old",
            "I live in Mumbai",
            "I am a doctor"
        ]
        
        for fact in facts:
            response = client.post(
                f"/api/chats/{test_chat.id}/stream",
                headers=auth_headers,
                json={"content": fact, "scan_ids": []}
            )
            assert response.status_code == 200
            # Consume response
            for _ in response.iter_lines():
                pass
        
        # Ask about all facts
        response = client.post(
            f"/api/chats/{test_chat.id}/stream",
            headers=auth_headers,
            json={
                "content": "Tell me everything you know about me",
                "scan_ids": []
            }
        )
        
        assert response.status_code == 200
        
        # Parse response
        content_chunks = []
        for line in response.iter_lines():
            line_str = line.decode('utf-8') if isinstance(line, bytes) else line
            if line_str.startswith('data: '):
                try:
                    data_str = line_str[6:].strip()
                    if data_str:
                        event_data = json.loads(data_str)
                        if event_data and event_data.get('type') == 'content_chunk':
                            content_chunks.append(event_data.get('data', {}).get('content', ''))
                except json.JSONDecodeError:
                    pass
        
        full_content = ''.join(content_chunks).lower()
        
        # Should remember at least some facts
        remembered_facts = sum([
            '25' in full_content or 'twenty' in full_content,
            'mumbai' in full_content,
            'doctor' in full_content
        ])
        
        assert remembered_facts >= 2, \
            f"Agent should remember at least 2 out of 3 facts. Got: {full_content}"


class TestAgentDependencies:
    """Test that all required dependencies are present and work."""
    
    def test_langgraph_is_available(self):
        """
        CRITICAL: This would have caught the missing langgraph issue!
        """
        try:
            import langgraph
            import langgraph.checkpoint
            assert True
        except ImportError as e:
            pytest.fail(f"langgraph is not installed: {e}")
    
    def test_logger_import_in_messages(self):
        """
        CRITICAL: This would have caught the missing logger import!
        """
        try:
            from app.api.messages import logger
            assert logger is not None
        except (ImportError, AttributeError) as e:
            pytest.fail(f"logger is not imported in messages.py: {e}")
    
    def test_logger_import_in_chat_processor(self):
        """
        CRITICAL: This would have caught logger import issues!
        """
        try:
            from app.services.chat_processor import logger
            assert logger is not None
        except (ImportError, AttributeError) as e:
            pytest.fail(f"logger is not imported in chat_processor.py: {e}")
    
    def test_agent_can_be_created_with_tools(self):
        """
        CRITICAL: Test that agent creation actually works.
        """
        from app.services.tool_manager import tool_manager as tm
        
        # Load a tool
        result = tm.load_tool('dicom_processor')
        assert result['success'], f"Failed to load tool: {result.get('error')}"
        
        tm.load_tool_in_background('dicom_processor')
        
        # Try to create agent
        agent = tm.create_agent()
        assert agent is not None, "Agent should be created when tools are loaded"
        
        # Verify it has the necessary attributes
        assert hasattr(agent, 'workflow'), "Agent should have workflow"


class TestAgentToolExecution:
    """Test that tools are actually called and work."""
    
    def test_dicom_processor_tool_loads(self):
        """Test that DICOM processor tool can be loaded."""
        from app.services.tool_manager import tool_manager as tm
        
        result = tm.load_tool('dicom_processor')
        assert result['success']
        
        tm.load_tool_in_background('dicom_processor')
        
        # Check it's loaded
        loaded = tm.get_loaded_tools()
        assert len(loaded) > 0, "Should have at least one loaded tool"
    
    def test_agent_responds_to_simple_query(self, client: TestClient, auth_headers, test_chat):
        """Test that agent can respond to a simple query."""
        # Load tool
        from app.services.tool_manager import tool_manager as tm
        tm.load_tool('dicom_processor')
        tm.load_tool_in_background('dicom_processor')
        
        response = client.post(
            f"/api/chats/{test_chat.id}/stream",
            headers=auth_headers,
            json={
                "content": "Hello, can you help me with medical imaging?",
                "scan_ids": []
            }
        )
        
        assert response.status_code == 200
        
        # Parse response
        events = []
        has_content = False
        for line in response.iter_lines():
            line_str = line.decode('utf-8') if isinstance(line, bytes) else line
            if line_str.startswith('data: '):
                try:
                    data_str = line_str[6:].strip()
                    if data_str:
                        event_data = json.loads(data_str)
                        if event_data:
                            events.append(event_data)
                            if event_data.get('type') == 'content_chunk':
                                has_content = True
                except json.JSONDecodeError:
                    pass
        
        # Should have received events
        assert len(events) > 0, "Should have received events"
        
        # Should have content (AI response)
        assert has_content, "Should have received AI content response"
        
        # Should not have errors
        has_error = any(e.get('type') == 'error' for e in events if e)
        if has_error:
            error_msgs = [e.get('data', {}).get('error') for e in events if e and e.get('type') == 'error']
            pytest.fail(f"Should not have errors. Got: {error_msgs}")


class TestAgentErrorHandling:
    """Test that agent handles errors gracefully."""
    
    def test_agent_error_when_no_tools_loaded(self, client: TestClient, auth_headers, test_chat):
        """Test that agent gives proper error when no tools are loaded."""
        # Make sure no tools are loaded
        from app.services.tool_manager import tool_manager as tm
        
        # Unload all tools
        for tool in tm.get_all_tools():
            if tool.get('status') == 'loaded':
                tm.unload_tool(tool.get('id'))
        
        response = client.post(
            f"/api/chats/{test_chat.id}/stream",
            headers=auth_headers,
            json={
                "content": "Hello",
                "scan_ids": []
            }
        )
        
        assert response.status_code == 200
        
        # Parse response
        events = []
        for line in response.iter_lines():
            line_str = line.decode('utf-8') if isinstance(line, bytes) else line
            if line_str.startswith('data: '):
                try:
                    data_str = line_str[6:].strip()
                    if data_str:
                        event_data = json.loads(data_str)
                        if event_data:
                            events.append(event_data)
                except json.JSONDecodeError:
                    pass
        
        # Should have error about no tools
        has_no_tools_error = any(
            e.get('type') == 'error' and 
            'No MedRAX tools' in str(e.get('data', {}).get('error', ''))
            for e in events if e
        )
        
        assert has_no_tools_error, "Should have error about no tools loaded"
    
    def test_logger_doesnt_crash_on_valid_calls(self):
        """
        CRITICAL: Test that logger calls don't crash.
        This would have caught the keyword argument bug!
        """
        from app.utils.logging_config import logger
        
        # These should NOT crash
        try:
            # Correct f-string format
            logger.info(f"test message_id={'abc123'} chat_id={'xyz789'}")
            logger.error(f"test error message_id={'abc123'}")
            
            # These SHOULD crash (if they don't, the bug is present)
            try:
                # This is the WRONG format that was causing crashes
                logger.info("test", message_id="abc123", chat_id="xyz789")
                pytest.fail("Logger should NOT accept keyword arguments!")
            except TypeError:
                # Good! It correctly rejects keyword arguments
                pass
                
        except Exception as e:
            pytest.fail(f"Logger crashed with valid f-string format: {e}")


class TestCompleteChatFlow:
    """Test complete chat flows end-to-end."""
    
    def test_complete_conversation_flow(self, client: TestClient, auth_headers, test_patient, db_session):
        """
        Complete flow: Create chat → Load tool → Have conversation → Check memory
        """
        # 1. Create chat
        chat_response = client.post(
            f"/api/patients/{test_patient.id}/chats",
            headers=auth_headers,
            json={}
        )
        assert chat_response.status_code == 201
        chat = chat_response.json()
        chat_id = chat['id']
        
        # 2. Load tool
        from app.services.tool_manager import tool_manager as tm
        tm.load_tool('dicom_processor')
        tm.load_tool_in_background('dicom_processor')
        
        # 3. Have conversation
        messages = [
            "Hello, I'm Dr. Kumar",
            "I specialize in radiology",
            "What is my name and specialty?"
        ]
        
        for msg in messages:
            response = client.post(
                f"/api/chats/{chat_id}/stream",
                headers=auth_headers,
                json={"content": msg, "scan_ids": []}
            )
            assert response.status_code == 200
            
            # Consume response
            for _ in response.iter_lines():
                pass
        
        # 4. Check that messages were saved
        messages_response = client.get(
            f"/api/chats/{chat_id}/messages",
            headers=auth_headers
        )
        assert messages_response.status_code == 200
        saved_messages = messages_response.json()
        
        # Should have 3 user messages + 3 assistant messages
        assert len(saved_messages) >= 6, \
            f"Should have at least 6 messages (3 user + 3 assistant). Got: {len(saved_messages)}"
        
        # 5. Verify last response mentions "Kumar" and "radiology"
        last_assistant_message = None
        for msg in reversed(saved_messages):
            if msg['role'] == 'assistant' and msg['content']:
                last_assistant_message = msg
                break
        
        assert last_assistant_message is not None, "Should have assistant response"
        content = last_assistant_message['content'].lower()
        
        # Should remember at least one piece of information
        remembers_something = 'kumar' in content or 'radiology' in content
        assert remembers_something, \
            f"Agent should remember context. Got: {last_assistant_message['content']}"


@pytest.fixture(autouse=True)
def cleanup_tools():
    """Clean up loaded tools after each test."""
    yield
    # Unload all tools after test
    from app.services.tool_manager import tool_manager as tm
    for tool in tm.get_all_tools():
        if tool.get('status') == 'loaded':
            try:
                tm.unload_tool(tool.get('id'))
            except:
                pass

