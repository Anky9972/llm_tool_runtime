"""Tests for multi-step tool chaining."""

import pytest
from llm_tool_runtime import ToolRuntime
from tests.mock_llm import StatefulMockLLM


class ChainMockLLM:
    """Mock LLM that simulates a multi-step conversation."""
    
    def __init__(self):
        self.steps = 0
        
    def __call__(self, system: str, user_conversation: str) -> str:
        self.steps += 1
        
        # Helper to check if a tool has been run
        has_weather_result = "Tool 'search_weather' result" in user_conversation
        
        # Step 1: Call search_weather if we haven't yet
        if not has_weather_result and "search_weather" not in user_conversation:
            return """
I'll check the weather.
<tool_call>
{ "name": "search_weather", "arguments": { "city": "Delhi" } }
</tool_call>
"""
        
        # Step 2: Call celsius_to_fahrenheit after getting weather result
        if has_weather_result and "celsius_to_fahrenheit" not in user_conversation:
            return """
The weather is 32°C. Let me convert that.
<tool_call>
{ "name": "celsius_to_fahrenheit", "arguments": { "celsius": 32 } }
</tool_call>
"""
        
        # Step 3: Final answer
        return "The temperature in Delhi is 89.6°F."


def test_weather_conversion_chain():
    """Test a 2-step chain: search_weather -> celsius_to_fahrenheit -> answer."""
    mock = ChainMockLLM()
    runtime = ToolRuntime(mock, max_steps=5)
    
    @runtime.tool
    def search_weather(city: str) -> int:
        return 32  # Returns 32 degrees Celsius
        
    @runtime.tool
    def celsius_to_fahrenheit(celsius: float) -> float:
        return (celsius * 9/5) + 32
        
    result = runtime.run("What is the weather in Delhi in Fahrenheit?")
    
    # Assert final result contains the answer
    assert "89.6°F" in result
    
    # Assert we took 3 steps (2 tools + 1 final)
    assert mock.steps >= 3


def test_max_steps_limit():
    """Test that execution stops after max_steps."""
    # An LLM that keeps calling tools forever
    def loop_llm(system, conversation):
        return """
<tool_call>
{ "name": "noop", "arguments": {} }
</tool_call>
"""
    
    runtime = ToolRuntime(loop_llm, max_steps=3)
    
    @runtime.tool
    def noop():
        return "ok"
        
    # Should raise error after 3 steps
    from llm_tool_runtime.errors import MaxRetriesExceededError
    with pytest.raises(MaxRetriesExceededError) as exc:
        runtime.run("Go")
        
    assert "3 steps/attempts" in str(exc.value)
