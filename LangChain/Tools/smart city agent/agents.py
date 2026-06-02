"""
USE CASE 

DESIGN A TOOL WHICH CAN GET LATEST WEATHER NEWS OF A GIVEN CITY"""

"""
FLOW

MAKE 2 TOOLS ONE FOR GETTING CITY NEWS ONE FOR GETTING WEATHER OF GIVEN CITY

"""
import openmeteo_requests

import pandas as pd
import requests_cache
from retry_requests import retry
from dotenv import load_dotenv

load_dotenv()

import os
import requests

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage


from tavily import TavilyClient
#Now lets make some tools



#WEATHER TOOL
@tool
def get_weather(latitude: float, longitude: float) -> str:
    """
    Get weather info 
    """
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,        # 🔹 changed
        "longitude": longitude,      # 🔹 changed
        "hourly": "temperature_2m",
        "past_days": 0,
        "forecast_days": 7,
    }

    responses = openmeteo.weather_api(url, params = params)

    response = responses[0]
    print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")

    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end =  pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}

    hourly_data["temperature_2m"] = hourly_temperature_2m

    hourly_dataframe = pd.DataFrame(data = hourly_data)

    #return hourly_dataframe.to_string()   # 🔹 better than print
    #get latest temp
    return hourly_data["temperature_2m"][-1]


print(get_weather.invoke({
    "latitude": 52.52,
    "longitude": 13.41
}))


#News Search Tool using Tavily news tool

tavily=TavilyClient(os.getenv("TAVILY_API_KEY"))


@tool 
def get_news(city: str) -> str:
    """
    Get latest news about the city
    """
    response = tavily.search(
        query=f"latest news in {city}",
        search_depth="basic",
        topic="news",              # important for news
        max_results=5
    )

    results = response.get("results", [])

    if not results:
        return "No news found."

    news_list = []
    for r in results:
        news_list.append(f"{r['title']} - {r['url']}")

    return "\n".join(news_list)

print(get_news.invoke("Lucknow"))


#LLM AND BINDING TOOLS WITH IT

llm=ChatMistralAI(model="mistral-small-2506")

tools={
    "get_weather":get_weather,
    "get_news":get_news
}

llm_with_tool=llm.bind_tools([get_weather,get_news])


#making an agent loop

messages=[]
print("=================City Intelligence System=================")
print("Type EXIT to quit")

while True:
    user_input = input("YOU: ")
    if user_input.lower() == "exit":
        break

    messages.append(HumanMessage(content=user_input))

    while True:  # inner loop to handle tool calls
        result = llm_with_tool.invoke(messages)
        messages.append(result)

        if not result.tool_calls:
            # No tools needed — final answer
            print(f"AGENT: {result.content}")
            break

        # Handle each tool call
        for tool_call in result.tool_calls:
            tool_name = tool_call['name']

            confirm = input(f"Agent wants to call '{tool_name}'. Approve? (YES/NO): ")

            if confirm.strip().lower() == "no":
                print(f"Tool call '{tool_name}' denied.")
                # Must still append ToolMessage to keep message order valid
                messages.append(ToolMessage(
                    content="Tool call was denied by the user.",
                    tool_call_id=tool_call['id']
                ))
            else:
                tool_result = str(tools[tool_name].invoke(tool_call))
                messages.append(ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call['id']
                ))
        # After all ToolMessages appended, loop again to get LLM's final answer
        