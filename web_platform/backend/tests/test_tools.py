"""
Tool Management API Tests
"""

import pytest


def test_list_tools(client, auth_headers):
    """Test listing all tools."""
    response = client.get("/api/tools", headers=auth_headers)
    assert response.status_code == 200
    
    tools = response.json()
    assert isinstance(tools, list)
    assert len(tools) > 0
    
    # Check tool structure
    tool = tools[0]
    assert "id" in tool
    assert "name" in tool
    assert "status" in tool
    assert "description" in tool
    assert "category" in tool


def test_list_tools_unauthorized(client):
    """Test listing tools without authentication."""
    response = client.get("/api/tools")
    assert response.status_code == 401


def test_load_tool_without_deps(client, auth_headers):
    """Test loading a tool when dependencies not installed."""
    response = client.post(
        "/api/tools/classification/load",
        json={},
        headers=auth_headers
    )
    # Should fail gracefully with 400 if deps not installed
    assert response.status_code in [200, 400]
    
    data = response.json()
    if response.status_code == 400:
        assert "error" in data or "detail" in data


def test_unload_tool(client, auth_headers):
    """Test unloading a tool."""
    response = client.post(
        "/api/tools/classification/unload",
        json={},
        headers=auth_headers
    )
    # Should succeed even if not loaded
    assert response.status_code in [200, 400]


def test_load_nonexistent_tool(client, auth_headers):
    """Test loading a tool that doesn't exist."""
    response = client.post(
        "/api/tools/nonexistent/load",
        json={},
        headers=auth_headers
    )
    assert response.status_code == 400
    assert "error" in response.json() or "detail" in response.json()

