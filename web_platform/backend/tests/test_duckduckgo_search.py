"""
Test DuckDuckGo Search Tool

Tests the web search tool's functionality and output format.
"""

import pytest
import json
from medrax.tools.browsing.duckduckgo import DuckDuckGoSearchTool


@pytest.fixture
def search_tool():
    """Create a DuckDuckGo search tool instance."""
    return DuckDuckGoSearchTool()


def test_search_tool_initialization(search_tool):
    """Test that the search tool initializes correctly."""
    assert search_tool.name == "duckduckgo_search"
    assert "search" in search_tool.description.lower()
    assert search_tool.return_direct == False


def test_search_tool_returns_json_string(search_tool):
    """Test that the search tool returns a JSON string."""
    # Use a common search term that should return results
    result = search_tool._run(query="Python programming", max_results=3)
    
    # Result should be a string
    assert isinstance(result, str)
    
    # Should be valid JSON
    parsed = json.loads(result)
    assert isinstance(parsed, dict)
    
    # Should have output and metadata keys
    assert "output" in parsed
    assert "metadata" in parsed


def test_search_tool_output_structure(search_tool):
    """Test that the search tool output has the correct structure."""
    result = search_tool._run(query="medical imaging", max_results=2)
    parsed = json.loads(result)
    
    output = parsed["output"]
    metadata = parsed["metadata"]
    
    # Output structure
    assert "query" in output
    assert "results_count" in output
    assert "results" in output
    assert "search_engine" in output
    assert "timestamp" in output
    
    # Metadata structure
    assert "query" in metadata
    assert "max_results" in metadata
    assert "region" in metadata
    assert "tool" in metadata
    assert "operation" in metadata
    assert metadata["tool"] == "duckduckgo_search"


def test_search_tool_results_format(search_tool):
    """Test that search results have the correct format."""
    result = search_tool._run(query="chest X-ray analysis", max_results=1)
    parsed = json.loads(result)
    
    output = parsed["output"]
    results = output.get("results", [])
    
    # If we got results, check their format
    if len(results) > 0:
        first_result = results[0]
        assert "rank" in first_result
        assert "title" in first_result
        assert "url" in first_result
        assert "snippet" in first_result
        assert "source" in first_result
        assert first_result["source"] == "DuckDuckGo"


@pytest.mark.asyncio
async def test_search_tool_async(search_tool):
    """Test async search functionality."""
    result = await search_tool._arun(query="pneumonia treatment", max_results=2)
    
    # Should return JSON string
    assert isinstance(result, str)
    
    # Should be valid JSON with correct structure
    parsed = json.loads(result)
    assert "output" in parsed
    assert "metadata" in parsed
    
    # Check output structure
    output = parsed["output"]
    assert "query" in output
    assert "results_count" in output


def test_search_tool_error_handling(search_tool):
    """Test that errors are handled gracefully."""
    # Empty query might cause an error, but should still return valid JSON
    result = search_tool._run(query="", max_results=1)
    parsed = json.loads(result)
    
    # Should have output and metadata even on error
    assert "output" in parsed
    assert "metadata" in parsed
    
    # Metadata should indicate failure if there was an error
    output = parsed["output"]
    if "error" in output:
        assert parsed["metadata"]["analysis_status"] == "failed"
        assert "error_details" in parsed["metadata"]


def test_search_tool_max_results_respected(search_tool):
    """Test that max_results parameter is respected."""
    result = search_tool._run(query="medical AI", max_results=3)
    parsed = json.loads(result)
    
    output = parsed["output"]
    results = output.get("results", [])
    
    # Should have at most 3 results (might be fewer if DuckDuckGo returns less)
    assert len(results) <= 3


def test_search_tool_region_parameter(search_tool):
    """Test that region parameter is accepted."""
    result = search_tool._run(query="healthcare", max_results=1, region="uk-en")
    parsed = json.loads(result)
    
    # Should parse successfully
    assert "output" in parsed
    assert "metadata" in parsed
    
    # Metadata should reflect the region
    metadata = parsed["metadata"]
    assert metadata["region"] == "uk-en"


def test_search_summary_method(search_tool):
    """Test the get_search_summary helper method."""
    summary = search_tool.get_search_summary(query="radiology", max_results=2)
    
    assert "query" in summary
    assert "status" in summary
    assert summary["query"] == "radiology"
    
    # If successful, should have results
    if summary["status"] == "success":
        assert "total_results" in summary
        assert "titles" in summary
        assert "urls" in summary
        assert "snippets" in summary






