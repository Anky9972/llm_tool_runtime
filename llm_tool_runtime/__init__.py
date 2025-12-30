"""LLM Tool Runtime - Tool calling runtime for text-only LLMs."""

from .runtime import ToolRuntime
from .registry import Tool, ToolRegistry
from .parser import parse_tool_call
from .prompt import build_system_prompt
from .errors import (
    ToolRuntimeError,
    ToolNotFoundError,
    ToolExecutionError,
    ParseError,
    MaxRetriesExceededError,
    LLMConnectionError,
    InvalidAPIKeyError,
    RateLimitError,
    InvalidToolArgumentsError,
    ModelNotSupportedError,
)

__version__ = "0.1.0"
__all__ = [
    # Main classes
    "ToolRuntime",
    "Tool",
    "ToolRegistry",
    # Utility functions
    "parse_tool_call",
    "build_system_prompt",
    # Exceptions
    "ToolRuntimeError",
    "ToolNotFoundError",
    "ToolExecutionError",
    "ParseError",
    "MaxRetriesExceededError",
    "LLMConnectionError",
    "InvalidAPIKeyError",
    "RateLimitError",
    "InvalidToolArgumentsError",
    "ModelNotSupportedError",
]
