"""Tests for the tool runtime with add tool."""

import pytest
from llm_tool_runtime.runtime import ToolRuntime
from llm_tool_runtime.errors import ToolNotFoundError, MaxRetriesExceededError
from tests.mock_llm import (
    mock_add_llm,
    mock_final_answer_llm,
    mock_no_tool_llm,
    StatefulMockLLM
)


def test_add_tool_registration():
    """Test that tools can be registered."""
    rt = ToolRuntime(mock_add_llm)

    @rt.tool
    def add(a: int, b: int) -> int:
        return a + b

    assert "add" in rt.registry.list_tools()


def test_add_tool_call():
    """Test that the add tool is called correctly."""
    mock = StatefulMockLLM()
    rt = ToolRuntime(mock)

    @rt.tool
    def add(a: int, b: int) -> int:
        return a + b

    result = rt.run("Add 2 and 3")
    assert result is not None
    assert "5" in result


def test_no_tool_needed():
    """Test response when no tool call is made."""
    rt = ToolRuntime(mock_no_tool_llm)

    @rt.tool
    def add(a: int, b: int) -> int:
        return a + b

    result = rt.run("What is the meaning of life?")
    assert "42" in result


def test_tool_with_description():
    """Test registering a tool with custom description."""
    rt = ToolRuntime(mock_add_llm)

    @rt.tool(description="Adds two numbers together")
    def add(a: int, b: int) -> int:
        return a + b

    tool = rt.registry.get("add")
    assert tool.description == "Adds two numbers together"


def test_multiple_tools():
    """Test registering multiple tools."""
    rt = ToolRuntime(mock_add_llm)

    @rt.tool
    def add(a: int, b: int) -> int:
        return a + b

    @rt.tool
    def multiply(a: int, b: int) -> int:
        return a * b

    @rt.tool
    def greet(name: str) -> str:
        return f"Hello, {name}!"

    tools = rt.registry.list_tools()
    assert len(tools) == 3
    assert "add" in tools
    assert "multiply" in tools
    assert "greet" in tools
