"""Core runtime engine for LLM tool calling."""

from typing import Callable, Optional, Union, Any
from .registry import ToolRegistry
from .prompt import build_system_prompt, build_tool_result_prompt
from .parser import parse_tool_call
from .errors import (
    MaxRetriesExceededError, 
    ToolRuntimeError,
    LLMConnectionError,
    InvalidAPIKeyError,
    RateLimitError,
    ToolNotFoundError,
)

# LangChain imports (optional)
try:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import SystemMessage, HumanMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseChatModel = None


class ToolRuntime:
    """
    Runtime engine for LLM tool calling.
    
    Supports both:
    - Custom callable: def my_llm(system_prompt: str, user_prompt: str) -> str
    - LangChain models: Any BaseChatModel instance
    
    Example:
        >>> from langchain_google_genai import ChatGoogleGenerativeAI
        >>> llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        >>> runtime = ToolRuntime(llm)
        >>> 
        >>> @runtime.tool
        >>> def add(a: int, b: int) -> int:
        ...     return a + b
        >>> 
        >>> runtime.run("What is 5 + 3?")
        'The result is 8.'
    """
    
    def __init__(
        self, 
        llm: Union[Callable[[str, str], str], Any],
        max_retries: int = 3,
        verbose: bool = False
    ):
        """
        Initialize the tool runtime.
        
        Args:
            llm: Either a callable (system, user) -> str, or a LangChain model
            max_retries: Maximum number of tool call attempts before failing
            verbose: If True, print debug information
            
        Raises:
            ValueError: If llm is None or invalid
        """
        if llm is None:
            raise ValueError("LLM cannot be None. Provide a LangChain model or callable.")
        
        if not callable(llm) and not self._check_langchain_model(llm):
            raise ValueError(
                "LLM must be either a callable(system, user) -> str or a LangChain BaseChatModel"
            )
        
        self.llm = llm
        self.registry = ToolRegistry()
        self.max_retries = max(1, max_retries)  # At least 1 attempt
        self.verbose = verbose
        self._is_langchain = self._check_langchain_model(llm)
        self._use_combined_prompt = False  # Track if we need to skip system messages

    def _check_langchain_model(self, llm: Any) -> bool:
        """Check if the provided LLM is a LangChain model."""
        if not LANGCHAIN_AVAILABLE:
            return False
        if BaseChatModel is None:
            return False
        return isinstance(llm, BaseChatModel)

    def _handle_api_error(self, error: Exception) -> None:
        """Convert common API errors to our custom exceptions."""
        error_str = str(error).lower()
        
        # Check for API key errors
        if any(phrase in error_str for phrase in [
            'api key', 'invalid key', 'unauthorized', 'authentication', 
            'api_key', 'invalid_api_key', '401', 'forbidden'
        ]):
            raise InvalidAPIKeyError() from error
        
        # Check for rate limit errors
        if any(phrase in error_str for phrase in [
            'rate limit', 'rate_limit', 'too many requests', '429', 
            'quota exceeded', 'quota_exceeded'
        ]):
            raise RateLimitError() from error
        
        # Check for connection errors
        if any(phrase in error_str for phrase in [
            'connection', 'timeout', 'network', 'unreachable', 
            'dns', 'ssl', 'certificate'
        ]):
            raise LLMConnectionError(f"Failed to connect to LLM: {error}", error) from error

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        Call the LLM with system and user prompts.
        
        Handles models that don't support system instructions by automatically
        falling back to combined prompts.
        """
        if self._is_langchain:
            try:
                # If we know this model doesn't support system messages, skip trying
                if self._use_combined_prompt:
                    combined_prompt = f"{system_prompt}\n\n---\n\nUser: {user_prompt}"
                    messages = [HumanMessage(content=combined_prompt)]
                    response = self.llm.invoke(messages)
                    return response.content
                
                # Try with system message first
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]
                response = self.llm.invoke(messages)
                return response.content
                
            except Exception as e:
                error_str = str(e)
                
                # If system message not supported, switch to combined prompt mode
                if "Developer instruction is not enabled" in error_str or \
                   "system" in error_str.lower() and "not supported" in error_str.lower():
                    if self.verbose:
                        print("System instructions not supported, using combined prompt...")
                    self._use_combined_prompt = True
                    combined_prompt = f"{system_prompt}\n\n---\n\nUser: {user_prompt}"
                    messages = [HumanMessage(content=combined_prompt)]
                    response = self.llm.invoke(messages)
                    return response.content
                
                # Handle other API errors
                self._handle_api_error(e)
                raise LLMConnectionError(f"LLM call failed: {e}", e) from e
        else:
            # Custom callable
            try:
                result = self.llm(system_prompt, user_prompt)
                if result is None:
                    raise ValueError("LLM callable returned None")
                return str(result)
            except Exception as e:
                self._handle_api_error(e)
                raise LLMConnectionError(f"Custom LLM call failed: {e}", e) from e

    def tool(self, fn: Optional[Callable] = None, *, description: Optional[str] = None):
        """
        Decorator to register a function as a tool.
        
        Usage:
            @runtime.tool
            def my_tool(arg1: str) -> str:
                return "result"
                
            @runtime.tool(description="Custom description")
            def another_tool(x: int) -> int:
                return x * 2
        """
        return self.registry.register(fn, description=description)

    def run(self, user_prompt: str) -> str:
        """
        Run the tool calling loop for a user prompt.
        
        Args:
            user_prompt: The user's input/question
            
        Returns:
            The final LLM response after any tool calls
            
        Raises:
            ValueError: If user_prompt is empty
            MaxRetriesExceededError: If tool calling fails after max retries
            InvalidAPIKeyError: If API key is invalid or missing
            RateLimitError: If API rate limit is exceeded
            LLMConnectionError: If connection to LLM fails
        """
        # Validate input
        if not user_prompt or not user_prompt.strip():
            raise ValueError("User prompt cannot be empty")
        
        # Check if any tools are registered
        if not self.registry.tools:
            if self.verbose:
                print("Warning: No tools registered. LLM will respond without tool access.")
        
        system_prompt = build_system_prompt(self.registry.tools)
        current_prompt = user_prompt.strip()
        last_error = None

        for attempt in range(self.max_retries):
            if self.verbose:
                print(f"\n[Attempt {attempt + 1}/{self.max_retries}]")
                print(f"User prompt: {current_prompt[:100]}...")

            try:
                output = self._call_llm(system_prompt, current_prompt)
            except (InvalidAPIKeyError, RateLimitError, LLMConnectionError):
                # Don't retry these errors - they need user intervention
                raise
            except Exception as e:
                last_error = str(e)
                if self.verbose:
                    print(f"LLM call error: {e}")
                if attempt == self.max_retries - 1:
                    raise LLMConnectionError(f"LLM call failed after {self.max_retries} attempts: {e}", e)
                continue
            
            if self.verbose:
                print(f"LLM output: {output[:200]}...")

            call = parse_tool_call(output)

            if not call:
                # No tool call, return the response
                if self.verbose:
                    print("No tool call detected, returning response")
                return output

            tool_name = call["name"]
            tool_args = call["arguments"]
            
            if self.verbose:
                print(f"Tool call: {tool_name}({tool_args})")

            try:
                tool = self.registry.get(tool_name)
                result = tool.call(tool_args)
                
                if self.verbose:
                    print(f"Tool result: {result}")
                
                # Build the next prompt with tool result
                current_prompt = build_tool_result_prompt(tool_name, str(result))
                
            except ToolNotFoundError as e:
                if self.verbose:
                    print(f"Tool not found: {e}")
                # Tell the LLM the tool doesn't exist
                available = self.registry.list_tools()
                current_prompt = (
                    f"Error: Tool '{tool_name}' does not exist. "
                    f"Available tools are: {', '.join(available) if available else 'none'}. "
                    f"Please try again with a valid tool or respond without using tools."
                )
                last_error = str(e)
                
            except ToolRuntimeError as e:
                if self.verbose:
                    print(f"Tool error: {e}")
                # Include error in next prompt for LLM to handle
                current_prompt = (
                    f"Error calling tool '{tool_name}': {e}\n\n"
                    f"Please try a different approach or respond without using tools."
                )
                last_error = str(e)
                
            except Exception as e:
                if self.verbose:
                    print(f"Unexpected tool error: {e}")
                current_prompt = (
                    f"Unexpected error with tool '{tool_name}': {e}\n\n"
                    f"Please respond without using this tool."
                )
                last_error = str(e)

        raise MaxRetriesExceededError(self.max_retries, last_error)

    def run_safe(self, user_prompt: str, default: str = "I encountered an error processing your request.") -> str:
        """
        Run the tool calling loop with automatic error handling.
        
        This method never raises exceptions - it returns a default message on error.
        Useful for production environments where you want graceful degradation.
        
        Args:
            user_prompt: The user's input/question
            default: Default message to return on error
            
        Returns:
            The LLM response or the default message on error
        """
        try:
            return self.run(user_prompt)
        except InvalidAPIKeyError:
            return "Service configuration error. Please contact support."
        except RateLimitError:
            return "Service is temporarily busy. Please try again in a moment."
        except LLMConnectionError:
            return "Unable to connect to the AI service. Please check your connection."
        except MaxRetriesExceededError:
            return "Unable to complete the request. Please try rephrasing your question."
        except Exception as e:
            if self.verbose:
                print(f"Unexpected error in run_safe: {e}")
            return default

    def run_with_history(self, user_prompt: str, history: list = None) -> tuple[str, list]:
        """
        Run with conversation history support.
        
        Args:
            user_prompt: The user's input
            history: List of previous (user, assistant) message tuples
            
        Returns:
            Tuple of (response, updated_history)
        """
        history = history or []
        
        # Validate history format
        for item in history:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("History must be a list of (user_message, assistant_message) tuples")
        
        # Build context from history
        context_parts = []
        for user_msg, assistant_msg in history[-5:]:  # Last 5 exchanges
            context_parts.append(f"User: {user_msg}")
            context_parts.append(f"Assistant: {assistant_msg}")
        
        if context_parts:
            full_prompt = "\n".join(context_parts) + f"\n\nUser: {user_prompt}"
        else:
            full_prompt = user_prompt
        
        response = self.run(full_prompt)
        history.append((user_prompt, response))
        
        return response, history
