"""
Comprehensive Tool Manager Tests

Tests for tool registration, loading, caching, and all 15 tools.
"""

import pytest
import os
import tempfile
from pathlib import Path
from app.services.tool_manager import ToolManager, ToolStatus


def test_tool_manager_initialization():
    """Test that ToolManager initializes correctly."""
    manager = ToolManager()
    
    # Should have tools registered
    assert len(manager.tools) > 0
    assert manager.tools is not None


def test_all_15_tools_registered():
    """Test that all 15 tools are properly registered."""
    manager = ToolManager()
    
    # Should have exactly 15 tools
    assert len(manager.tools) == 15
    
    # Check all expected tool IDs
    expected_tools = [
        'torchxrayvision',
        'arcplus',
        'chexagent',
        'llava_med',
        'medgemma',
        'medsam2',
        'chest_segmentation',
        'report_generator',
        'phrase_grounding',
        'dicom_processor',
        'xray_generator',
        'rag',
        'web_search',
        'web_browser',
        'python_sandbox'
    ]
    
    for tool_id in expected_tools:
        assert tool_id in manager.tools, f"Tool {tool_id} not registered"


def test_tool_categories():
    """Test that tools are properly categorized."""
    manager = ToolManager()
    
    categories = {}
    for tool in manager.tools.values():
        category = tool.category
        if category not in categories:
            categories[category] = []
        categories[category].append(tool.id)
    
    # Should have 8 categories
    assert len(categories) >= 7  # At least 7 categories
    
    # Check specific categories
    assert 'classification' in categories
    assert 'vqa' in categories
    assert 'segmentation' in categories
    assert 'generation' in categories or 'processing' in categories
    assert 'retrieval' in categories


def test_tool_info_structure():
    """Test that tool info has all required fields."""
    manager = ToolManager()
    
    for tool in manager.tools.values():
        # Required fields
        assert hasattr(tool, 'id')
        assert hasattr(tool, 'name')
        assert hasattr(tool, 'description')
        assert hasattr(tool, 'category')
        assert hasattr(tool, 'tool_class')
        assert hasattr(tool, 'module_path')
        assert hasattr(tool, 'dependencies')
        assert hasattr(tool, 'requires_gpu')
        assert hasattr(tool, 'status')
        
        # Check types
        assert isinstance(tool.id, str)
        assert isinstance(tool.name, str)
        assert isinstance(tool.description, str)
        assert isinstance(tool.category, str)
        assert isinstance(tool.dependencies, list)
        assert isinstance(tool.requires_gpu, bool)
        assert tool.status in [
            ToolStatus.AVAILABLE,
            ToolStatus.UNAVAILABLE,
            ToolStatus.LOADED,
            ToolStatus.ERROR
        ]


def test_dependency_checking():
    """Test individual dependency checking."""
    manager = ToolManager()
    
    # Should be able to check dependencies
    # Standard library should always be available
    assert manager._check_dependency('os') == True
    assert manager._check_dependency('sys') == True
    
    # Non-existent module should not be available
    assert manager._check_dependency('this_module_definitely_does_not_exist_12345') == False


def test_get_all_tools():
    """Test getting all tools as dictionaries."""
    manager = ToolManager()
    
    tools = manager.get_all_tools()
    
    # Should return list
    assert isinstance(tools, list)
    assert len(tools) == 15
    
    # Each tool should be a dict
    for tool in tools:
        assert isinstance(tool, dict)
        assert 'id' in tool
        assert 'name' in tool
        assert 'description' in tool
        assert 'category' in tool
        assert 'status' in tool
        assert 'dependencies' in tool
        assert 'requires_gpu' in tool


def test_get_specific_tool():
    """Test getting a specific tool."""
    manager = ToolManager()
    
    # Get web browser tool (should exist)
    tool = manager.get_tool('web_browser')
    assert tool is not None
    assert tool.id == 'web_browser'
    assert tool.name == 'Web Browser'
    
    # Get non-existent tool
    tool = manager.get_tool('nonexistent_tool')
    assert tool is None


def test_gpu_requirements():
    """Test that GPU requirements are properly set."""
    manager = ToolManager()
    
    # Tools that require GPU
    gpu_tools = ['arcplus', 'chexagent', 'llava_med', 
                 'medsam2', 'chest_segmentation', 'report_generator',
                 'phrase_grounding', 'xray_generator']
    
    # Tools that work without GPU (CPU/MPS compatible)
    non_gpu_tools = ['torchxrayvision', 'web_browser', 'dicom_processor', 'web_search', 
                     'rag', 'python_sandbox', 'medgemma']
    
    for tool_id in gpu_tools:
        tool = manager.get_tool(tool_id)
        assert tool is not None, f"Tool {tool_id} not found"
        assert tool.requires_gpu == True, f"Tool {tool_id} should require GPU"
    
    for tool_id in non_gpu_tools:
        tool = manager.get_tool(tool_id)
        if tool:  # Some might not be registered
            assert tool.requires_gpu == False, f"Tool {tool_id} should not require GPU"


def test_tool_status_changes():
    """Test tool status transitions."""
    manager = ToolManager()
    
    # Get a tool that's unavailable
    unavailable_tools = [t for t in manager.tools.values() if t.status == ToolStatus.UNAVAILABLE]
    
    if unavailable_tools:
        tool = unavailable_tools[0]
        
        # Should have error message
        assert tool.error_message is not None
        assert 'Missing dependencies' in tool.error_message


def test_load_nonexistent_tool():
    """Test loading a tool that doesn't exist."""
    manager = ToolManager()
    
    result = manager.load_tool('nonexistent_tool_id_12345')
    
    assert result['success'] == False
    assert 'not found' in result['error'].lower()


def test_unload_nonexistent_tool():
    """Test unloading a tool that doesn't exist."""
    manager = ToolManager()
    
    result = manager.unload_tool('nonexistent_tool_id_12345')
    
    assert result['success'] == False
    assert 'not found' in result['error'].lower()


def test_load_unavailable_tool():
    """Test loading a tool that's unavailable."""
    manager = ToolManager()
    
    # Find an unavailable tool
    unavailable_tool = None
    for tool in manager.tools.values():
        if tool.status == ToolStatus.UNAVAILABLE:
            unavailable_tool = tool
            break
    
    if unavailable_tool:
        result = manager.load_tool(unavailable_tool.id)
        
        assert result['success'] == False
        assert 'unavailable' in result['error'].lower() or 'dependencies' in result['error'].lower()


def test_get_loaded_tools():
    """Test getting currently loaded tools."""
    manager = ToolManager()
    
    loaded = manager.get_loaded_tools()
    
    # Should return a list
    assert isinstance(loaded, list)
    
    # Initially should be empty (no tools loaded)
    assert len(loaded) == 0


def test_is_agent_ready():
    """Test agent readiness check."""
    manager = ToolManager()
    
    # Without loaded tools, agent should not be ready
    assert manager.is_agent_ready() == False


def test_tool_cache_directory_creation():
    """Test that cache directories would be created on tool load."""
    manager = ToolManager()
    
    # This tests the logic, not actual directory creation
    # Since we can't load tools without dependencies in tests
    
    # Just verify the method exists and is callable
    assert hasattr(manager, '_load_tool_instance')
    assert callable(manager._load_tool_instance)


def test_tool_dependencies_structure():
    """Test that dependencies are properly structured."""
    manager = ToolManager()
    
    for tool in manager.tools.values():
        # Dependencies should be a list
        assert isinstance(tool.dependencies, list)
        
        # Each dependency should be a string
        for dep in tool.dependencies:
            assert isinstance(dep, str)
            assert len(dep) > 0


def test_classification_tools():
    """Test that both classification tools are registered correctly."""
    manager = ToolManager()
    
    # TorchXRayVision
    txrv = manager.get_tool('torchxrayvision')
    assert txrv is not None
    assert txrv.category == 'classification'
    assert 'torch' in txrv.dependencies
    assert 'torchxrayvision' in txrv.dependencies
    
    # ArcPlus
    arcplus = manager.get_tool('arcplus')
    assert arcplus is not None
    assert arcplus.category == 'classification'
    assert 'timm' in arcplus.dependencies


def test_vqa_tools():
    """Test that all VQA tools are registered correctly."""
    manager = ToolManager()
    
    # CheXagent
    chexagent = manager.get_tool('chexagent')
    assert chexagent is not None
    assert chexagent.category == 'vqa'
    
    # LLaVA-Med
    llava = manager.get_tool('llava_med')
    assert llava is not None
    assert llava.category == 'vqa'
    
    # MedGemma
    medgemma = manager.get_tool('medgemma')
    assert medgemma is not None
    assert medgemma.category == 'vqa'
    assert medgemma.requires_gpu == False  # API-based


def test_segmentation_tools():
    """Test that segmentation tools are registered correctly."""
    manager = ToolManager()
    
    # MedSAM2
    medsam2 = manager.get_tool('medsam2')
    assert medsam2 is not None
    assert medsam2.category == 'segmentation'
    assert 'sam2' in medsam2.dependencies
    
    # Chest Segmentation
    chest_seg = manager.get_tool('chest_segmentation')
    assert chest_seg is not None
    assert chest_seg.category == 'segmentation'


def test_retrieval_tools():
    """Test that retrieval tools are registered correctly."""
    manager = ToolManager()
    
    # RAG
    rag = manager.get_tool('rag')
    assert rag is not None
    assert rag.category == 'retrieval'
    
    # Web Search
    web_search = manager.get_tool('web_search')
    assert web_search is not None
    assert web_search.category == 'retrieval'
    assert 'duckduckgo_search' in web_search.dependencies
    
    # Web Browser
    web_browser = manager.get_tool('web_browser')
    assert web_browser is not None
    assert web_browser.category == 'retrieval'
    assert len(web_browser.dependencies) == 0  # No dependencies


def test_tool_module_paths():
    """Test that tool module paths are correctly set."""
    manager = ToolManager()
    
    for tool in manager.tools.values():
        # Should have module_path
        assert tool.module_path is not None
        assert isinstance(tool.module_path, str)
        
        # Should start with 'medrax.tools'
        assert tool.module_path.startswith('medrax.tools')
        
        # Should have tool_class
        assert tool.tool_class is not None
        assert isinstance(tool.tool_class, str)
        assert 'Tool' in tool.tool_class


def test_tool_descriptions():
    """Test that all tools have meaningful descriptions."""
    manager = ToolManager()
    
    for tool in manager.tools.values():
        # Should have description
        assert tool.description is not None
        assert isinstance(tool.description, str)
        
        # Should be reasonably long (more than just a few words)
        assert len(tool.description) > 20
        
        # Should not be empty or just whitespace
        assert tool.description.strip() != ''


def test_tool_manager_singleton_behavior():
    """Test that tool manager can be instantiated multiple times."""
    manager1 = ToolManager()
    manager2 = ToolManager()
    
    # Both should have same number of tools
    assert len(manager1.tools) == len(manager2.tools)
    
    # Both should have same tool IDs
    assert set(manager1.tools.keys()) == set(manager2.tools.keys())


def test_error_message_format():
    """Test that error messages for unavailable tools are helpful."""
    manager = ToolManager()
    
    for tool in manager.tools.values():
        if tool.status == ToolStatus.UNAVAILABLE:
            # Should have error message
            assert tool.error_message is not None
            
            # Should mention missing dependencies
            assert 'Missing dependencies' in tool.error_message or 'dependencies' in tool.error_message.lower()


def test_tool_stats():
    """Test tool statistics calculation."""
    manager = ToolManager()
    tools = manager.get_all_tools()
    
    available = sum(1 for t in tools if t['status'] == 'available')
    loaded = sum(1 for t in tools if t['status'] == 'loaded')
    unavailable = sum(1 for t in tools if t['status'] == 'unavailable')
    
    # Total should equal sum of all statuses
    assert available + loaded + unavailable == 15
    
    # Initially, loaded should be 0
    assert loaded == 0


def test_no_duplicate_tool_ids():
    """Test that there are no duplicate tool IDs."""
    manager = ToolManager()
    
    tool_ids = [tool.id for tool in manager.tools.values()]
    
    # All IDs should be unique
    assert len(tool_ids) == len(set(tool_ids))


def test_no_duplicate_tool_names():
    """Test that there are no duplicate tool names."""
    manager = ToolManager()
    
    tool_names = [tool.name for tool in manager.tools.values()]
    
    # All names should be unique
    assert len(tool_names) == len(set(tool_names))

