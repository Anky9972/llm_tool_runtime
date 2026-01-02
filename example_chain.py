"""
example_chain.py

An example of using ToolRuntime with LangChain and Google Generative AI.
Demonstrates registering tools and making chain calls to them using natural language.
"""
###############################################################
# Imports and Environment Setup
###############################################################
import os
from llm_tool_runtime import ToolRuntime
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check for Google API key in environment variables
if not os.getenv("GOOGLE_API_KEY"):
    print("❌ ERROR: GOOGLE_API_KEY not found!")
    print("   Create a .env file with: GOOGLE_API_KEY=your-key-here")
    exit(1)

# Notify user that API key is found and runtime is initializing
print("✅ API Key found!")
print("🚀 Initializing LLM Tool Runtime...\n")

# Initialize the language model and ToolRuntime
llm = ChatGoogleGenerativeAI(model="gemma-3-27b-it")
runtime = ToolRuntime(llm, max_retries=20, verbose=True)

# Register tools using the @runtime.tool decorator
@runtime.tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

# Example tool: get_weather
@runtime.tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    # Your weather API logic here
    return f"Weather in {city}: Sunny, 25°C"

# Example: Run with natural language (uncomment to test basic addition)
# result = runtime.run("What is 15 + 27?")
# print(result)  # "The result of 15 + 27 is 42."

# Example: Chain multiple tool calls using natural language
result = runtime.run("call the get_weather tool for Tokyo then call the weather tool for New York then add temperatures of these two cities using the add tool")
print(result)  # "The combined temperature of Tokyo and New York is 50°C."