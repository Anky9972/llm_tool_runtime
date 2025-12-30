"""
Manual test script for llm-tool-runtime package.

Instructions:
1. Create a .env file with your API key (copy from .env.example)
2. Install Google GenAI: python -m uv pip install langchain-google-genai --python .venv/Scripts/python.exe
3. Run: .venv/Scripts/python example.py
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from llm_tool_runtime import ToolRuntime
from langchain_google_genai import ChatGoogleGenerativeAI

# Check for API key
if not os.getenv("GOOGLE_API_KEY"):
    print("❌ ERROR: GOOGLE_API_KEY not found!")
    print("   Create a .env file with: GOOGLE_API_KEY=your-key-here")
    exit(1)

print("✅ API Key found!")
print("🚀 Initializing LLM Tool Runtime...\n")

# Initialize with Google Gemini
llm = ChatGoogleGenerativeAI(model="gemma-3-27b-it")
runtime = ToolRuntime(llm, verbose=True)

# Register tools
@runtime.tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

@runtime.tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b

@runtime.tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    # Simulated weather data
    weather_data = {
        "mumbai": "Sunny, 32°C, Humidity: 75%",
        "delhi": "Hazy, 28°C, AQI: 180",
        "bangalore": "Cloudy, 24°C, Light rain expected",
        "new york": "Cold, 5°C, Snow expected",
        "london": "Rainy, 8°C, Winds: 20 km/h",
    }
    city_lower = city.lower()
    return weather_data.get(city_lower, f"Weather data not available for {city}")

@runtime.tool
def calculate_bmi(weight_kg: float, height_m: float) -> str:
    """Calculate BMI given weight in kg and height in meters."""
    bmi = weight_kg / (height_m ** 2)
    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal weight"
    elif bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"
    return f"BMI: {bmi:.1f} ({category})"


print("=" * 50)
print("🛠️  Registered Tools:")
for tool_name in runtime.registry.list_tools():
    tool = runtime.registry.get(tool_name)
    print(f"   • {tool_name}: {tool.description}")
print("=" * 50)
print()

# Test prompts
test_prompts = [
    "What is 25 + 17?",
    "Multiply 8 and 9",
    "What's the weather in Mumbai?",
    "Calculate BMI for someone who weighs 70 kg and is 1.75 meters tall",
]

print("🧪 Running test prompts...\n")

for i, prompt in enumerate(test_prompts, 1):
    print(f"─── Test {i} ───")
    print(f"📝 Prompt: {prompt}")
    
    # Use run_safe for production-grade error handling
    result = runtime.run_safe(prompt, default="⚠️  I couldn't process that request right now.")
    
    if "⚠️" in result:
        print(f"❌ Error handled gracefully: {result}")
    else:
        print(f"✅ Response: {result}")
    print()

print("=" * 50)
print("🎉 Manual testing complete!")
print("=" * 50)
