# llm-tool-runtime



**llm-tool-runtime** is a lightweight, extensible execution layer for Large Language Models. It allows you to register Python functions as "tools" and gives your LLM the ability to invoke them seamlessly while maintaining type safety and handling complex chaining.

---

## 🚀 Quick Start

### Installation

```bash
pip install llm-tool-runtime
```

Or with `uv`:

```bash
uv add llm-tool-runtime
```

### Optional Model Support
Depending on your provider, you may need additional dependencies. Install them using **extras**:

| Provider | Install Command |
| :--- | :--- |
| **Google** | ``` pip install "llm-tool-runtime[google]" ``` |
| **OpenAI** | ``` pip install "llm-tool-runtime[openai]" ``` |
| **Anthropic** | ``` pip install "llm-tool-runtime[anthropic]" ``` |
| **Ollama** | ``` pip install "llm-tool-runtime[ollama]" ``` |
| **All** | ``` pip install "llm-tool-runtime[all]" ``` |

---

## 💡 Basic Usage

Registering a tool is as simple as adding a decorator. The runtime uses your Python **type hints** and **docstrings** to explain the tool to the LLM.

```python
from llm_tool_runtime import ToolRuntime
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Initialize your LLM and the Runtime
llm = ChatGoogleGenerativeAI(model="gemma-3-27b-it")
runtime = ToolRuntime(llm, verbose=True)

# 2. Register a function as a tool
@runtime.tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

# 3. Run a natural language query
result = runtime.run("What is 15 + 27?")
print(result)  # "The result of 15 + 27 is 42."
```
## 🛠 Features

* **Type Safety**: Automatic validation of tool arguments using Python type hints.
* **Easy Registration**: Decorator-based tool definition (`@runtime.tool`).
* **Provider Agnostic**: Works with OpenAI, Anthropic, Google, and local models via Ollama.
* **Complex Chaining**: Handles multi-step logic where the output of one tool is needed for the next.
* **Custom Registry**: Manage and version your tools with the built-in `Registry` module.

---

## 📂 Project Architecture

The library is designed to be modular and easy to extend:

* **`ToolRuntime`**: The main entry point for orchestration.
* **`Registry`**: Handles the storage and lookup of available tools.
* **`Parser`**: Translates LLM responses into executable tool calls.
* **`Prompt`**: Manages the system instructions sent to the model.

---

## 📚 Resources

* **[Core Functions](https://Anky9972.github.io/llm_tool_runtime/reference/)**: Detailed documentation of all classes and methods.
* **[PyPI](https://pypi.org/project/llm-tool-runtime/)**: Package page for installation and version info.
* **[Example Scripts](https://github.com/Anky9972/llm_tool_runtime/blob/main/example.py)**: Explore `example.py` and `example_chain.py` for advanced patterns.
* **[GitHub Repository](https://github.com/Anky9972/llm_tool_runtime)**: Report issues or contribute to the project.
