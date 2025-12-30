"""Mock LLM implementations for testing without real models."""


def mock_add_llm(system: str, user: str) -> str:
    """Mock LLM that always calls the add tool."""
    return """
I'll add those numbers for you.

<tool_call>
{ "name": "add", "arguments": { "a": 2, "b": 3 } }
</tool_call>
"""


def mock_final_answer_llm(system: str, user: str) -> str:
    """Mock LLM that returns a final answer after tool result."""
    if "Tool '" in user and "returned:" in user:
        return "The result is 5. I used the add tool to calculate 2 + 3 = 5."
    return """
<tool_call>
{ "name": "add", "arguments": { "a": 2, "b": 3 } }
</tool_call>
"""


def mock_no_tool_llm(system: str, user: str) -> str:
    """Mock LLM that never calls tools."""
    return "I don't need any tools to answer that. The answer is 42."


def mock_invalid_tool_llm(system: str, user: str) -> str:
    """Mock LLM that calls a non-existent tool."""
    return """
<tool_call>
{ "name": "nonexistent_tool", "arguments": {} }
</tool_call>
"""


def mock_malformed_json_llm(system: str, user: str) -> str:
    """Mock LLM that returns malformed JSON."""
    return """
<tool_call>
{ "name": "add", "arguments": { broken json here } }
</tool_call>
"""


class StatefulMockLLM:
    """Mock LLM that tracks call count and changes behavior."""
    
    def __init__(self):
        self.call_count = 0
        
    def __call__(self, system: str, user: str) -> str:
        self.call_count += 1
        
        if self.call_count == 1:
            return """
<tool_call>
{ "name": "add", "arguments": { "a": 2, "b": 3 } }
</tool_call>
"""
        else:
            return "The result is 5."
