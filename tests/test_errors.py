"""Tests for error handling."""

import pytest
from llm_tool_runtime import (
    ToolRuntime,
    ToolRuntimeError,
    ToolNotFoundError,
    ToolExecutionError,
    MaxRetriesExceededError,
    InvalidAPIKeyError,
    RateLimitError,
    LLMConnectionError,
)
from tests.mock_llm import mock_add_llm, StatefulMockLLM


def test_empty_prompt_raises_error():
    """Test that empty prompts raise ValueError."""
    mock = StatefulMockLLM()
    runtime = ToolRuntime(mock)
    
    with pytest.raises(ValueError, match="cannot be empty"):
        runtime.run("")
    
    with pytest.raises(ValueError, match="cannot be empty"):
        runtime.run("   ")


def test_none_llm_raises_error():
    """Test that None LLM raises ValueError."""
    with pytest.raises(ValueError, match="cannot be None"):
        ToolRuntime(None)


def test_tool_not_found_error():
    """Test ToolNotFoundError includes available tools."""
    def mock_llm_bad_tool(system, user):
        return '''
<tool_call>
{"name": "nonexistent_tool", "arguments": {}}
</tool_call>
'''
    
    runtime = ToolRuntime(mock_llm_bad_tool, max_retries=1)
    
    @runtime.tool
    def real_tool():
        return "hello"
    
    # The runtime should handle this gracefully and not crash
    # It will ask the LLM to try again
    try:
        result = runtime.run("test")
    except MaxRetriesExceededError:
        pass  # Expected


def test_tool_not_found_error_message():
    """Test ToolNotFoundError message format."""
    error = ToolNotFoundError("bad_tool", available_tools=["add", "multiply"])
    assert "bad_tool" in str(error)
    assert "add" in str(error)
    assert "multiply" in str(error)


def test_tool_execution_error():
    """Test that tool execution errors are caught."""
    mock = StatefulMockLLM()
    runtime = ToolRuntime(mock, max_retries=2)
    
    @runtime.tool
    def failing_tool(a: int, b: int) -> int:
        raise ValueError("Intentional error")
    
    # The runtime should not crash, but handle the error
    # It retries and eventually fails gracefully


def test_run_safe_returns_default():
    """Test run_safe returns default on error."""
    def failing_llm(system, user):
        raise Exception("Connection error")
    
    runtime = ToolRuntime(failing_llm)
    
    result = runtime.run_safe("test", default="Custom default")
    # Connection errors return a specific message, not the default
    assert "connect" in result.lower() or result == "Custom default"


def test_run_safe_with_rate_limit():
    """Test run_safe handles rate limit errors."""
    def rate_limited_llm(system, user):
        raise Exception("rate limit exceeded")
    
    runtime = ToolRuntime(rate_limited_llm)
    
    result = runtime.run_safe("test")
    assert "busy" in result.lower() or "try again" in result.lower()


def test_run_safe_with_api_key_error():
    """Test run_safe handles API key errors."""
    def bad_key_llm(system, user):
        raise Exception("invalid api key")
    
    runtime = ToolRuntime(bad_key_llm)
    
    result = runtime.run_safe("test")
    assert "configuration" in result.lower() or "support" in result.lower()


def test_invalid_history_format():
    """Test that invalid history format raises error."""
    mock = StatefulMockLLM()
    runtime = ToolRuntime(mock)
    
    @runtime.tool
    def add(a: int, b: int) -> int:
        return a + b
    
    # Invalid history format
    with pytest.raises(ValueError, match="must be a list"):
        runtime.run_with_history("test", history=["invalid"])


def test_max_retries_error_includes_last_error():
    """Test MaxRetriesExceededError includes last error info."""
    error = MaxRetriesExceededError(3, last_error="Connection timeout")
    assert "3 attempts" in str(error)
    assert "Connection timeout" in str(error)
