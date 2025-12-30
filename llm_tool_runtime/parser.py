"""Parser for extracting tool calls from LLM output."""

import json
import re
from typing import Optional
from .types import ToolCall

# Pattern to match tool call blocks
TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL
)


def parse_tool_call(text: str) -> Optional[ToolCall]:
    """
    Parse a tool call from LLM output text.
    
    Args:
        text: The raw LLM output text
        
    Returns:
        ToolCall dict with 'name' and 'arguments' if found, None otherwise
    """
    if not text:
        return None
        
    match = TOOL_CALL_PATTERN.search(text)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(1))
        
        # Validate structure
        if not isinstance(parsed, dict):
            return None
        if "name" not in parsed:
            return None
        if "arguments" not in parsed:
            parsed["arguments"] = {}
        if not isinstance(parsed["arguments"], dict):
            return None
            
        return ToolCall(
            name=parsed["name"],
            arguments=parsed["arguments"]
        )
    except json.JSONDecodeError:
        return None


def extract_all_tool_calls(text: str) -> list[ToolCall]:
    """
    Extract all tool calls from LLM output (for future multi-tool support).
    
    Args:
        text: The raw LLM output text
        
    Returns:
        List of ToolCall dicts found in the text
    """
    calls = []
    for match in TOOL_CALL_PATTERN.finditer(text):
        try:
            parsed = json.loads(match.group(1))
            if isinstance(parsed, dict) and "name" in parsed:
                calls.append(ToolCall(
                    name=parsed["name"],
                    arguments=parsed.get("arguments", {})
                ))
        except json.JSONDecodeError:
            continue
    return calls
