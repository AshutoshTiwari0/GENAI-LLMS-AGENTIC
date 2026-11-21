from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI # type: ignore


load_dotenv()

model=ChatMistralAI(model="mistral-small-2506")

res=model.invoke("What is AI?")

print(res.content)