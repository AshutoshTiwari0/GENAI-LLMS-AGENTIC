from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

#USING SEARCH TOOL HERE
search_tool=TavilySearch(max_results=5) #get 5 articles

llm=ChatMistralAI(model="mistral-small-2506")

prompt=ChatPromptTemplate.from_template(
"""
You are a news summarizer. Take the following news and summarize it in points.
{news}
"""
)

chain=prompt|llm| StrOutputParser()

#get the news

news=search_tool.run("Iran War News")

result=chain.invoke({"news":news})

print(result) 