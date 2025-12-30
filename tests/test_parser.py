"""Tests for the tool call parser."""

import pytest
from llm_tool_runtime.parser import parse_tool_call, extract_all_tool_calls


def test_parse_valid_tool_call():
    """Test parsing a valid tool call."""
    text = """
I'll help you with that.

<tool_call>
{ "name": "add", "arguments": { "a": 5, "b": 3 } }
</tool_call>
"""
    result = parse_tool_call(text)
    assert result is not None
    assert result["name"] == "add"
    assert result["arguments"]["a"] == 5
    assert result["arguments"]["b"] == 3


def test_parse_no_tool_call():
    """Test parsing text without a tool call."""
    text = "Just a regular response without any tool calls."
    result = parse_tool_call(text)
    assert result is None


def test_parse_empty_arguments():
    """Test parsing a tool call with no arguments."""
    text = """
<tool_call>
{ "name": "get_time", "arguments": {} }
</tool_call>
"""
    result = parse_tool_call(text)
    assert result is not None
    assert result["name"] == "get_time"
    assert result["arguments"] == {}


def test_parse_missing_arguments_key():
    """Test parsing a tool call without arguments key."""
    text = """
<tool_call>
{ "name": "get_time" }
</tool_call>
"""
    result = parse_tool_call(text)
    assert result is not None
    assert result["name"] == "get_time"
    assert result["arguments"] == {}


def test_parse_malformed_json():
    """Test parsing malformed JSON returns None."""
    text = """
<tool_call>
{ "name": "add", "arguments": { broken } }
</tool_call>
"""
    result = parse_tool_call(text)
    assert result is None


def test_parse_empty_text():
    """Test parsing empty text."""
    result = parse_tool_call("")
    assert result is None


def test_parse_none_text():
    """Test parsing None returns None."""
    result = parse_tool_call(None)
    assert result is None


def test_parse_with_extra_whitespace():
    """Test parsing with extra whitespace."""
    text = """
<tool_call>

    { "name": "add", "arguments": { "x": 1 } }

</tool_call>
"""
    result = parse_tool_call(text)
    assert result is not None
    assert result["name"] == "add"


def test_extract_multiple_tool_calls():
    """Test extracting multiple tool calls."""
    text = """
<tool_call>
{ "name": "first", "arguments": { "a": 1 } }
</tool_call>

Some text in between.

<tool_call>
{ "name": "second", "arguments": { "b": 2 } }
</tool_call>
"""
    results = extract_all_tool_calls(text)
    assert len(results) == 2
    assert results[0]["name"] == "first"
    assert results[1]["name"] == "second"
