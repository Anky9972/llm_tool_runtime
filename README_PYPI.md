# llm-tool-runtime

A lightweight, extensible runtime for calling LLM tools and chaining them together. Installable from PyPI for easy integration into your Python projects.

## Installation

Install the core package:

```bash
pip install llm-tool-runtime
```

### Optional Model Support

Some LLM providers require extra dependencies. You can install them with extras:

- **Google Gemma**:
  ```bash
  pip install "llm-tool-runtime[google]"
  ```
- **OpenAI**:
  ```bash
  pip install "llm-tool-runtime[openai]"
  ```
- **Anthropic**:
  ```bash
  pip install "llm-tool-runtime[anthropic]"
  ```
- **Ollama**:
  ```bash
  pip install "llm-tool-runtime[ollama]"
  ```
- **Together**:
  ```bash
  pip install "llm-tool-runtime[together]"
  ```
- **All supported providers**:
  ```bash
  pip install "llm-tool-runtime[all]"
  ```

## Usage

Import and use the runtime in your Python code:

```python
from llm_tool_runtime import ToolRuntime
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemma-3-27b-it")
runtime = ToolRuntime(llm, verbose=True)

@runtime.tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

result = runtime.run("What is 15 + 27?")
print(result)  # "The result of 15 + 27 is 42."
```

See [example.py](https://github.com/Anky9972/llm_tool_runtime/blob/main/example.py) and [example_chain.py](https://github.com/Anky9972/llm_tool_runtime/blob/main/example_chain.py) for usage examples.

## Features
- Register and call tools with LLMs
- Chain tools together for complex workflows
- Extensible registry for custom tools
- Error handling and type safety

## Documentation
- [Project README and Contribution Guide](https://github.com/Anky9972/llm_tool_runtime/blob/main/README.md)
- [PyPI Project Page](https://pypi.org/project/llm-tool-runtime/)

## Contributing

Contributions are welcome! Please see the [main README](https://github.com/Anky9972/llm_tool_runtime/blob/main/README.md) for guidelines.

## License

This project is licensed under the terms of the MIT License. See the LICENSE file for details.

## Support

For issues, suggestions, or questions, please open an issue on the [GitHub repository](https://github.com/Anky9972/llm_tool_runtime).
