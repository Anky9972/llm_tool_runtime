"""Type definitions for the LLM Tool Runtime."""

from typing import TypedDict, Optional, Any, Dict


class ToolCall(TypedDict):
    """Represents a parsed tool call from LLM output."""
    name: str
    arguments: Dict[str, Any]


class ToolResult(TypedDict):
    """Represents the result of a tool execution."""
    success: bool
    result: Optional[Any]
    error: Optional[str]
