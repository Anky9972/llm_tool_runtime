"""Tests for the tool registry."""

import pytest
from llm_tool_runtime.registry import Tool, ToolRegistry
from llm_tool_runtime.errors import ToolNotFoundError, ToolExecutionError


def test_tool_creation():
    """Test creating a Tool wrapper."""
    def sample_func(x: int, y: str) -> str:
        return f"{y}: {x}"
    
    tool = Tool(sample_func)
    assert tool.name == "sample_func"
    assert "x" in tool.signature.parameters
    assert "y" in tool.signature.parameters


def test_tool_call():
    """Test calling a tool."""
    def add(a: int, b: int) -> int:
        return a + b
    
    tool = Tool(add)
    result = tool.call({"a": 5, "b": 3})
    assert result == 8


def test_tool_type_conversion():
    """Test that string arguments are converted to proper types."""
    def add(a: int, b: int) -> int:
        return a + b
    
    tool = Tool(add)
    # Pass strings, should be converted to int
    result = tool.call({"a": "5", "b": "3"})
    assert result == 8


def test_tool_schema():
    """Test getting tool schema."""
    def greet(name: str, count: int) -> str:
        """Greet someone multiple times."""
        return f"Hello, {name}!" * count
    
    tool = Tool(greet)
    schema = tool.get_schema()
    
    assert schema["name"] == "greet"
    assert "Greet someone" in schema["description"]
    assert schema["parameters"]["name"] == "str"
    assert schema["parameters"]["count"] == "int"


def test_registry_register():
    """Test registering tools in registry."""
    registry = ToolRegistry()
    
    @registry.register
    def my_tool():
        return "hello"
    
    assert "my_tool" in registry.list_tools()


def test_registry_register_with_description():
    """Test registering with custom description."""
    registry = ToolRegistry()
    
    @registry.register(description="Custom description")
    def my_tool():
        return "hello"
    
    tool = registry.get("my_tool")
    assert tool.description == "Custom description"


def test_registry_get_not_found():
    """Test getting non-existent tool raises error."""
    registry = ToolRegistry()
    
    with pytest.raises(ToolNotFoundError) as exc_info:
        registry.get("nonexistent")
    
    assert "nonexistent" in str(exc_info.value)


def test_registry_get_all_schemas():
    """Test getting all tool schemas."""
    registry = ToolRegistry()
    
    @registry.register
    def tool1(x: int) -> int:
        return x
    
    @registry.register
    def tool2(y: str) -> str:
        return y
    
    schemas = registry.get_all_schemas()
    assert len(schemas) == 2
    names = [s["name"] for s in schemas]
    assert "tool1" in names
    assert "tool2" in names
