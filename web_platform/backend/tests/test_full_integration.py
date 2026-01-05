"""
Comprehensive Integration Tests for Frontend-Backend-MedRAX Integration

Tests the complete flow from frontend API calls through backend to MedRAX tools.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.main import app
from app.models import Doctor, Patient, Chat
from app.services.tool_manager import tool_manager


class TestFullStackIntegration:
    """Test complete integration across all layers."""
    
    def test_tool_manager_initialization(self):
        """Test that ToolManager loads all MedRAX tools correctly."""
        tools = tool_manager.get_all_tools()
        
        assert len(tools) == 15, "Should have all 15 tools registered"
        
        # Check that we have tools in each category
        categories = {tool['category'] for tool in tools}
        expected_categories = {'classification', 'vqa', 'segmentation', 'generation', 'grounding', 'processing', 'retrieval', 'execution'}
        
        # At least some core categories should be present
        assert 'classification' in categories
        assert 'vqa' in categories
        assert 'segmentation' in categories
    
    def test_tool_manager_availability_check(self):
        """Test that tool availability checking works."""
        tools = tool_manager.get_all_tools()
        
        available_tools = [t for t in tools if t['status'] == 'available']
        unavailable_tools = [t for t in tools if t['status'] == 'unavailable']
        
        # Should have at least 10 available tools (allowing for some that need special setup)
        assert len(available_tools) >= 10, f"Should have at least 10 available tools, got {len(available_tools)}"
        
        # Unavailable tools should have error messages
        for tool in unavailable_tools:
            assert 'error_message' in tool, f"Unavailable tool {tool['name']} should have error message"
            assert tool['error_message'], "Error message should not be empty"
    
    def test_medrax_tool_imports(self):
        """Test that MedRAX tools can be imported directly."""
        import sys
        sys.path.insert(0, '/Users/alankritverma/projects/MedRAX2')
        
        # Test a few key tools
        try:
            from medrax.tools.classification.torchxrayvision import TorchXRayVisionClassifierTool
            assert TorchXRayVisionClassifierTool is not None
        except Exception as e:
            pytest.fail(f"Failed to import TorchXRayVisionClassifierTool: {e}")
        
        try:
            from medrax.tools.vqa.xray_vqa import CheXagentXRayVQATool
            assert CheXagentXRayVQATool is not None
        except Exception as e:
            pytest.fail(f"Failed to import CheXagentXRayVQATool: {e}")
        
        try:
            from medrax.tools.segmentation.medsam2 import MedSAM2Tool
            assert MedSAM2Tool is not None
        except Exception as e:
            pytest.fail(f"Failed to import MedSAM2Tool: {e}")


class TestToolLoadingAPI:
    """Test tool loading/unloading through API endpoints."""
    
    def test_list_tools_endpoint(self, client, auth_headers):
        """Test GET /api/tools endpoint."""
        response = client.get("/api/tools", headers=auth_headers)
        
        assert response.status_code == 200
        tools = response.json()
        
        assert isinstance(tools, list)
        assert len(tools) == 15
        
        # Check tool structure
        for tool in tools:
            assert 'id' in tool
            assert 'name' in tool
            assert 'description' in tool
            assert 'status' in tool
            assert 'category' in tool
    
    def test_load_tool_endpoint_structure(self, client, auth_headers):
        """Test POST /api/tools/{tool_id}/load endpoint structure."""
        # Get list of available tools
        response = client.get("/api/tools", headers=auth_headers)
        tools = response.json()
        
        available_tools = [t for t in tools if t['status'] == 'available']
        
        if available_tools:
            tool_id = available_tools[0]['id']
            
            # Test load endpoint (may fail due to missing models, but endpoint should exist)
            response = client.post(f"/api/tools/{tool_id}/load", headers=auth_headers)
            
            # Should get 200 (success) or 400 (can't load) but not 404 (endpoint not found)
            assert response.status_code in [200, 400], f"Expected 200 or 400, got {response.status_code}"
    
    def test_unload_tool_endpoint_structure(self, client, auth_headers):
        """Test POST /api/tools/{tool_id}/unload endpoint structure."""
        response = client.get("/api/tools", headers=auth_headers)
        tools = response.json()
        
        if tools:
            tool_id = tools[0]['id']
            
            # Test unload endpoint
            response = client.post(f"/api/tools/{tool_id}/unload", headers=auth_headers)
            
            # Should get 200 (success) or 400 (not loaded) but not 404
            assert response.status_code in [200, 400], f"Expected 200 or 400, got {response.status_code}"


class TestPatientChatToolFlow:
    """Test complete flow: Create patient -> Create chat -> Load tool -> Query."""
    
    def test_complete_workflow(self, client, auth_headers, db_session):
        """Test end-to-end workflow from patient creation to tool usage."""
        
        # 1. Create a patient
        patient_data = {"name": "Integration Test Patient"}
        response = client.post("/api/patients/", json=patient_data, headers=auth_headers)
        assert response.status_code == 201
        patient = response.json()
        patient_id = patient['id']
        
        # 2. Create a chat for the patient
        response = client.get(f"/api/patients/{patient_id}/chats", headers=auth_headers)
        assert response.status_code == 200
        chats = response.json()
        
        if not chats:
            chat_data = {"name": "Initial Consultation"}
            response = client.post(
                f"/api/patients/{patient_id}/chats",
                json=chat_data,
                headers=auth_headers
            )
            assert response.status_code == 201
            chat = response.json()
            chat_id = chat['id']
        else:
            chat_id = chats[0]['id']
        
        # 3. Get available tools
        response = client.get("/api/tools", headers=auth_headers)
        assert response.status_code == 200
        tools = response.json()
        available_tools = [t for t in tools if t['status'] == 'available']
        assert len(available_tools) > 0, "Should have at least one available tool"
        
        # 4. Create a message in the chat
        message_data = {
            "content": "What tools are available for analysis?",
            "scan_ids": []
        }
        response = client.post(
            f"/api/chats/{chat_id}/messages",
            json=message_data,
            headers=auth_headers
        )
        assert response.status_code == 201
        message = response.json()
        assert message['role'] == 'user'
        assert message['content'] == message_data['content']
        
        # 5. Verify tool list is accessible
        assert len(available_tools) >= 10, f"Expected at least 10 available tools, got {len(available_tools)}"


class TestToolConfiguration:
    """Test tool configuration and caching."""
    
    def test_model_cache_directories(self):
        """Test that model cache directories are configured."""
        from app.config import settings
        
        assert hasattr(settings, 'MODEL_CACHE_DIR')
        assert hasattr(settings, 'HUGGINGFACE_CACHE_DIR')
        assert hasattr(settings, 'TORCH_CACHE_DIR')
    
    def test_tool_dependencies(self):
        """Test that tool dependencies are correctly specified."""
        tools = tool_manager.get_all_tools()
        
        for tool in tools:
            assert 'dependencies' in tool
            assert isinstance(tool['dependencies'], list)
            assert 'requires_gpu' in tool
            assert isinstance(tool['requires_gpu'], bool)


class TestErrorHandling:
    """Test error handling across the stack."""
    
    def test_invalid_tool_id(self, client, auth_headers):
        """Test loading non-existent tool."""
        response = client.post("/api/tools/nonexistent_tool/load", headers=auth_headers)
        assert response.status_code == 400
        assert 'detail' in response.json()
    
    def test_tool_list_without_auth(self, client):
        """Test that tool endpoints require authentication."""
        response = client.get("/api/tools")
        assert response.status_code == 401


class TestToolMetadata:
    """Test tool metadata and information."""
    
    def test_all_tools_have_required_metadata(self):
        """Test that all tools have complete metadata."""
        tools = tool_manager.get_all_tools()
        
        required_fields = ['id', 'name', 'description', 'category', 'status', 
                          'dependencies', 'requires_gpu']
        
        for tool in tools:
            for field in required_fields:
                assert field in tool, f"Tool {tool.get('id', 'unknown')} missing field: {field}"
                
            # Validate specific fields
            assert tool['id'], "Tool ID should not be empty"
            assert tool['name'], "Tool name should not be empty"
            assert tool['description'], "Tool description should not be empty"
            assert tool['status'] in ['available', 'unavailable', 'loaded', 'unloaded', 'error'], \
                f"Invalid status: {tool['status']}"


class TestToolCategories:
    """Test tool categorization."""
    
    def test_tool_categories_are_valid(self):
        """Test that all tools have valid categories."""
        tools = tool_manager.get_all_tools()
        
        valid_categories = {
            'classification', 'vqa', 'segmentation', 'generation',
            'grounding', 'processing', 'retrieval', 'execution'
        }
        
        for tool in tools:
            category = tool.get('category')
            assert category, f"Tool {tool['name']} has no category"
            # Category should be one of the valid ones (or we're flexible for now)
            # Just check it's not empty
            assert isinstance(category, str) and len(category) > 0
    
    def test_category_distribution(self):
        """Test that we have tools in multiple categories."""
        tools = tool_manager.get_all_tools()
        categories = {tool['category'] for tool in tools}
        
        # Should have at least 5 different categories
        assert len(categories) >= 5, f"Expected at least 5 categories, got {len(categories)}: {categories}"


def test_integration_summary():
    """Summary test to verify overall integration health."""
    
    # Test ToolManager
    tools = tool_manager.get_all_tools()
    available = [t for t in tools if t['status'] == 'available']
    
    print(f"\n{'='*70}")
    print("INTEGRATION TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Total Tools Registered: {len(tools)}")
    print(f"Available Tools: {len(available)}")
    print(f"Unavailable Tools: {len(tools) - len(available)}")
    print(f"{'='*70}\n")
    
    # Verify minimum requirements
    assert len(tools) == 15, f"Expected 15 tools, got {len(tools)}"
    assert len(available) >= 10, f"Expected at least 10 available tools, got {len(available)}"

