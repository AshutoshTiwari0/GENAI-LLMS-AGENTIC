import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor



# Load env
load_dotenv()


# Tool
@tool
def get_weather(city: str) -> str:
    """Get the weather of a city"""
    return f"It is always sunny in {city}"


# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash"
)




# Agent
agent = create_tool_calling_agent(
    llm=llm,
    tools=[get_weather],
)


executor = AgentExecutor(
    agent=agent,
    tools=[get_weather],
    verbose=True
)


# Run
result = executor.invoke({
    "input": "What is the weather in SF?"
})

print(result["output"])
