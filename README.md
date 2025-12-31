# 🛠️ LLM Tool Runtime

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-22%20passed-brightgreen.svg)]()

A lightweight, **model-agnostic** tool calling runtime for text-only LLMs. Works with any language model through LangChain or custom callables.

## ✨ Features

- 🔧 **Simple Tool Registration** - Use `@runtime.tool` decorator to register any Python function
- 🔄 **Automatic Retry Loop** - Handles tool call failures gracefully with configurable retries
- 🔌 **Model Agnostic** - Works with OpenAI, Anthropic, Google, Ollama, and any LLM
- 🛡️ **Safe Parsing** - Robust JSON extraction from LLM outputs
- 📝 **Type Conversion** - Automatic argument type conversion based on function signatures
- 🧪 **Fully Testable** - Mock LLMs included for testing without API calls
- 📦 **Zero Dependencies** - Core package has no required dependencies

---

## 📦 Installation

### From Source (Development)

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-tool-runtime.git
cd llm-tool-runtime

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install in development mode with your preferred provider
pip install -e ".[dev]"              # Just dev tools (pytest)
pip install -e ".[google]"           # Google Gemini/Gemma
pip install -e ".[openai]"           # OpenAI GPT models
pip install -e ".[ollama]"           # Ollama (local models)
pip install -e ".[all]"              # All providers
```

### From PyPI (Coming Soon)

```bash
pip install llm-tool-runtime
pip install llm-tool-runtime[google]  # With Google support
```

---

## 🚀 Quick Start

### 1. Basic Usage with Google Gemini

```python
import os
from llm_tool_runtime import ToolRuntime
from langchain_google_genai import ChatGoogleGenerativeAI

# Set your API key
os.environ["GOOGLE_API_KEY"] = "your-api-key"

# Initialize runtime with any LangChain model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
runtime = ToolRuntime(llm)

# Register tools using the decorator
@runtime.tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

@runtime.tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    # Your weather API logic here
    return f"Weather in {city}: Sunny, 25°C"

# Run with natural language
result = runtime.run("What is 15 + 27?")
print(result)  # "The result of 15 + 27 is 42."

result = runtime.run("What's the weather in Tokyo?")
print(result)  # "The weather in Tokyo is Sunny, 25°C."
```

### 2. With OpenAI

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")
runtime = ToolRuntime(llm, verbose=True)

@runtime.tool
def search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

runtime.run("Search for Python tutorials")
```

### 3. With Ollama (Local, Free!)

```python
from langchain_ollama import ChatOllama

# No API key needed - runs locally
llm = ChatOllama(model="llama3.2")
runtime = ToolRuntime(llm)

@runtime.tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

runtime.run("Calculate 2 ** 10")
```

### 4. With Any Custom LLM

```python
import requests
from llm_tool_runtime import ToolRuntime

def my_llm(system_prompt: str, user_prompt: str) -> str:
    """Custom LLM that calls any API."""
    response = requests.post("https://your-api.com/chat", json={
        "system": system_prompt,
        "user": user_prompt
    })
    return response.json()["text"]

runtime = ToolRuntime(my_llm)

@runtime.tool
def greet(name: str) -> str:
    return f"Hello, {name}!"

runtime.run("Say hello to Alice")
```

---

## 📖 API Reference

### `ToolRuntime`

The main class for managing tools and executing LLM interactions.

```python
ToolRuntime(
    llm,                    # LangChain model or callable(system, user) -> str
    max_retries: int = 3,   # Max tool call retry attempts
    verbose: bool = False   # Print debug information
)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `tool(fn)` | Decorator to register a function as a tool |
| `run(prompt)` | Execute the tool calling loop |
| `run_with_history(prompt, history)` | Run with conversation context |

### `@runtime.tool` Decorator

```python
# Simple registration
@runtime.tool
def my_tool(arg: str) -> str:
    """Tool description (used in prompt)."""
    return "result"

# With custom description
@runtime.tool(description="Custom description for the LLM")
def another_tool(x: int, y: int) -> int:
    return x + y
```

### Conversation History

```python
history = []

response, history = runtime.run_with_history("What's 5 + 3?", history)
# history = [("What's 5 + 3?", "The result is 8.")]

response, history = runtime.run_with_history("Multiply that by 2", history)
# Uses context from previous exchange
```

---

## 🔧 How It Works

```
User: "What's 15 + 27?"
           ↓
┌──────────────────────────────────────────────┐
│ 1. Build system prompt with tool definitions │
│ 2. Send to LLM                               │
│ 3. LLM responds:                             │
│    <tool_call>                               │
│    {"name": "add", "arguments": {"a": 15...}}│
│    </tool_call>                              │
│ 4. Parse tool call from response             │
│ 5. Execute: add(15, 27) → 42                 │
│ 6. Send result back to LLM                   │
│ 7. LLM provides final answer                 │
└──────────────────────────────────────────────┘
           ↓
Response: "The sum of 15 and 27 is 42."
```

### Tool Call Format

The runtime instructs LLMs to respond with:

```xml
<tool_call>
{"name": "function_name", "arguments": {"arg1": "value1"}}
</tool_call>
```

---

## 🛡️ Error Handling

The runtime includes a robust error handling system to ensure your application stays stable.

### Safe Execution (Recommended for Production)

Use `run_safe()` to handle errors gracefully without crashing your app. It catches API connection issues, rate limits, and authentication errors automatically.

```python
# Returns a friendly string instead of raising an exception
response = runtime.run_safe("What is 25 + 17?")

# You can customize the default error message
response = runtime.run_safe(
    "Complexity query...", 
    default="I apologize, but I'm having trouble connecting right now."
)
```

### Catching Specific Errors

For more control, you can catch specific exceptions:

```python
from llm_tool_runtime import (
    ToolRuntime, 
    InvalidAPIKeyError, 
    RateLimitError, 
    LLMConnectionError
)

try:
    result = runtime.run("My prompt")
except InvalidAPIKeyError:
    print("Please check your API key")
except RateLimitError:
    print("System is busy, please try again later")
except LLMConnectionError as e:
    print(f"Connection failed: {e}")
except MaxRetriesExceededError:
    print("Failed to get a valid response after multiple attempts")
```

---

## 🌍 Supported Models

Works with **any LLM** that can follow instructions. Tested with:

| Provider | Models | Package |
|----------|--------|---------|
| Google | Gemini 1.5/2.0, Gemma 3 | `langchain-google-genai` |
| OpenAI | GPT-4o, GPT-4, o1 | `langchain-openai` |
| Anthropic | Claude 3.5 Sonnet/Opus | `langchain-anthropic` |
| Ollama | Llama 3, Mistral, Qwen | `langchain-ollama` |
| Groq | Llama, Mixtral (fast!) | `langchain-groq` |
| DeepSeek | DeepSeek Chat/Coder | `langchain-openai` (custom base_url) |
| Together AI | Open source models | `langchain-together` |
| AWS Bedrock | Claude, Titan, Llama | `langchain-aws` |

---

## 🧪 Testing

### Run All Tests

```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Run tests
pytest -v

# With coverage
pytest --cov=llm_tool_runtime --cov-report=html
```

### Testing Without Real LLM

The package includes mock LLMs for testing:

```python
from llm_tool_runtime import ToolRuntime
from tests.mock_llm import StatefulMockLLM

def test_my_tool():
    mock = StatefulMockLLM()
    runtime = ToolRuntime(mock)
    
    @runtime.tool
    def add(a: int, b: int) -> int:
        return a + b
    
    result = runtime.run("Add 2 and 3")
    assert "5" in result
```

---

## 📁 Project Structure

```
llm_tool_runtime/
├── llm_tool_runtime/          # Main package
│   ├── __init__.py            # Package exports
│   ├── runtime.py             # Core ToolRuntime class
│   ├── registry.py            # Tool registration and management
│   ├── prompt.py              # System prompt builder
│   ├── parser.py              # Tool call JSON parser
│   ├── errors.py              # Custom exceptions
│   └── types.py               # Type definitions
│
├── tests/                     # Test suite
│   ├── __init__.py
│   ├── mock_llm.py            # Mock LLMs for testing
│   ├── test_add_tool.py       # Runtime tests
│   ├── test_parser.py         # Parser tests
│   └── test_registry.py       # Registry tests
│
├── .env.example               # Environment template
├── .gitignore                 # Git ignore rules
├── example.py                 # Working example script
├── pyproject.toml             # Package configuration
├── README.md                  # This file
└── LICENSE                    # MIT License
```

---

## 🔒 Environment Variables

Create a `.env` file (never commit this!):

```bash
# Google Gemini
GOOGLE_API_KEY=your-google-api-key

# OpenAI
OPENAI_API_KEY=your-openai-api-key

# Anthropic
ANTHROPIC_API_KEY=your-anthropic-api-key
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Run tests: `pytest -v`
5. Commit: `git commit -m 'Add amazing feature'`
6. Push: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/llm-tool-runtime.git
cd llm-tool-runtime

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dev dependencies
pip install -e ".[dev,all]"

# Run tests
pytest -v
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgments

- Built with [LangChain](https://github.com/langchain-ai/langchain) for model integrations
- Inspired by OpenAI's function calling and Anthropic's tool use

---

## 📬 Support

- 🐛 **Bug Reports**: [Open an issue](https://github.com/yourusername/llm-tool-runtime/issues)
- 💡 **Feature Requests**: [Open an issue](https://github.com/yourusername/llm-tool-runtime/issues)
- 📧 **Contact**: ankygaur9972@gmail.com

---

Made with ❤️ for the AI community
