import os
from dotenv import load_dotenv

#OPENAI

# from langchain.chat_models import init_chat_model
load_dotenv()
# model = init_chat_model("google_genai:gemini-2.5-flash-lite")

# response = model.invoke("What is LR(0) in compiler?")
# print(response)
# print("===========================")
# print(response.content) # we only need content


#Google Gemini

from langchain.chat_models import init_chat_model
model = init_chat_model("google_genai:gemini-2.5-flash-lite")
response = model.invoke("What is LR(0) in compiler?")
print(response.content)

print()

#Groq
print("GROQ RESPONSE")
from langchain.chat_models import init_chat_model
model = init_chat_model(
    "meta-llama/llama-4-scout-17b-16e-instruct",
    model_provider="groq"  #this is very imp to mention
)
response = model.invoke("What is LR(0) in compiler?")
print(response.content)

#Mistral->It is free
print()


print("MISTRAL RESPONSE")
from langchain.chat_models import init_chat_model
#we can chat init too but there is this other method too
from langchain_mistralai import ChatMistralAI

model=init_chat_model(
    "mistral-small-2506",
    temperature=1, #tells creativeness of model
    max_tokens=50
)
response = model.invoke("What is LR(0) in compiler?")
print(response.content)
