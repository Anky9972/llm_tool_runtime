"""Prompt builder for enforcing tool calling protocol."""

import json
from typing import Dict
from .registry import Tool


def build_system_prompt(tools: Dict[str, Tool]) -> str:
    """
    Build the system prompt that instructs the LLM on how to call tools.
    
    Args:
        tools: Dictionary of tool name to Tool objects
        
    Returns:
        System prompt string with tool definitions and calling format
    """
    tool_defs = []

    for tool in tools.values():
        schema = tool.get_schema()
        tool_defs.append(schema)

    tools_json = json.dumps(tool_defs, indent=2)

    return f"""You are a helpful assistant with access to tools. You can call tools by responding ONLY in this exact format:

<tool_call>
{{"name": "tool_name", "arguments": {{"arg1": "value1", "arg2": "value2"}}}}
</tool_call>

IMPORTANT RULES:
1. Use ONLY the exact tool names provided below
2. Provide ALL required arguments with correct types
3. Arguments must be valid JSON values
4. Only make ONE tool call at a time
5. If no tool is needed, respond normally without the <tool_call> tags

Available tools:
{tools_json}

When you receive a tool result, use it to formulate your final response to the user.""".strip()


def build_tool_result_prompt(tool_name: str, result: str) -> str:
    """
    Build the prompt for providing tool results back to the LLM.
    
    Args:
        tool_name: Name of the tool that was called
        result: String representation of the tool result
        
    Returns:
        Formatted prompt with tool result
    """
    return f"""Tool '{tool_name}' returned:
{result}

Now provide your final answer based on this result."""
