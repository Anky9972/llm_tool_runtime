"""Tool registry for managing registered functions."""

from typing import Callable, Dict, Any, Optional
from inspect import signature, Signature
from .errors import ToolNotFoundError, ToolExecutionError


class Tool:
    """Wrapper for a callable function registered as a tool."""
    
    def __init__(self, fn: Callable, description: Optional[str] = None):
        self.fn = fn
        self.name = fn.__name__
        self.signature: Signature = signature(fn)
        self.description = description or fn.__doc__ or f"Tool: {self.name}"

    def call(self, args: Dict[str, Any]) -> Any:
        """Execute the tool with the given arguments."""
        # Convert argument types based on signature annotations
        converted_args = {}
        for param_name, param in self.signature.parameters.items():
            if param_name in args:
                value = args[param_name]
                # Try to convert to annotated type if available
                if param.annotation != param.empty:
                    try:
                        converted_args[param_name] = param.annotation(value)
                    except (ValueError, TypeError):
                        converted_args[param_name] = value
                else:
                    converted_args[param_name] = value
        
        try:
            return self.fn(**converted_args)
        except Exception as e:
            raise ToolExecutionError(self.name, e)

    def get_schema(self) -> Dict[str, Any]:
        """Get the tool schema for prompt building."""
        params = {}
        for name, param in self.signature.parameters.items():
            param_type = "any"
            if param.annotation != param.empty:
                param_type = getattr(param.annotation, "__name__", str(param.annotation))
            params[name] = param_type
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": params
        }


class ToolRegistry:
    """Registry for managing tool functions."""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, fn: Optional[Callable] = None, *, description: Optional[str] = None):
        """
        Register a function as a tool.
        
        Can be used as a decorator with or without arguments:
            @registry.register
            def my_tool(): ...
            
            @registry.register(description="My tool description")
            def my_tool(): ...
        """
        def decorator(func: Callable) -> Callable:
            tool = Tool(func, description=description)
            self.tools[tool.name] = tool
            return func
        
        if fn is not None:
            return decorator(fn)
        return decorator

    def get(self, name: str) -> Tool:
        """Get a tool by name, raises ToolNotFoundError if not found."""
        if name not in self.tools:
            raise ToolNotFoundError(name, available_tools=self.list_tools())
        return self.tools[name]

    def list_tools(self) -> list:
        """List all registered tool names."""
        return list(self.tools.keys())

    def get_all_schemas(self) -> list:
        """Get schemas for all registered tools."""
        return [tool.get_schema() for tool in self.tools.values()]
