"""Custom exceptions for the LLM Tool Runtime."""


class ToolRuntimeError(Exception):
    """Base exception for tool runtime errors."""
    pass


class ToolNotFoundError(ToolRuntimeError):
    """Raised when a requested tool is not found in the registry."""
    def __init__(self, tool_name: str, available_tools: list = None):
        self.tool_name = tool_name
        self.available_tools = available_tools or []
        msg = f"Tool '{tool_name}' not found in registry"
        if self.available_tools:
            msg += f". Available tools: {', '.join(self.available_tools)}"
        super().__init__(msg)


class ToolExecutionError(ToolRuntimeError):
    """Raised when a tool execution fails."""
    def __init__(self, tool_name: str, original_error: Exception):
        self.tool_name = tool_name
        self.original_error = original_error
        super().__init__(f"Tool '{tool_name}' execution failed: {original_error}")


class ParseError(ToolRuntimeError):
    """Raised when parsing tool call from LLM output fails."""
    def __init__(self, message: str = "Failed to parse tool call from LLM output", raw_output: str = None):
        self.raw_output = raw_output
        super().__init__(message)


class MaxRetriesExceededError(ToolRuntimeError):
    """Raised when maximum retry attempts are exceeded."""
    def __init__(self, max_retries: int, last_error: str = None):
        self.max_retries = max_retries
        self.last_error = last_error
        msg = f"Operation failed after {max_retries} attempts"
        if last_error:
            msg += f". Last error: {last_error}"
        super().__init__(msg)


class LLMConnectionError(ToolRuntimeError):
    """Raised when connection to the LLM fails."""
    def __init__(self, message: str, original_error: Exception = None):
        self.original_error = original_error
        super().__init__(message)


class InvalidAPIKeyError(ToolRuntimeError):
    """Raised when the API key is invalid or missing."""
    def __init__(self, provider: str = "LLM"):
        self.provider = provider
        super().__init__(f"Invalid or missing API key for {provider}. Please check your environment variables.")


class RateLimitError(ToolRuntimeError):
    """Raised when the API rate limit is exceeded."""
    def __init__(self, retry_after: int = None):
        self.retry_after = retry_after
        msg = "API rate limit exceeded"
        if retry_after:
            msg += f". Retry after {retry_after} seconds"
        super().__init__(msg)


class InvalidToolArgumentsError(ToolRuntimeError):
    """Raised when tool arguments are invalid or missing."""
    def __init__(self, tool_name: str, expected: list, received: dict):
        self.tool_name = tool_name
        self.expected = expected
        self.received = received
        super().__init__(
            f"Invalid arguments for tool '{tool_name}'. "
            f"Expected: {expected}, Received: {list(received.keys())}"
        )


class ModelNotSupportedError(ToolRuntimeError):
    """Raised when the model doesn't support required features."""
    def __init__(self, model_name: str, feature: str):
        self.model_name = model_name
        self.feature = feature
        super().__init__(f"Model '{model_name}' does not support {feature}")
