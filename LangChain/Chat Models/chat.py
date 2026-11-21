import os
from dotenv import load_dotenv

# #OPENAI

# # from langchain.chat_models import init_chat_model
# load_dotenv()
# # model = init_chat_model("google_genai:gemini-2.5-flash-lite")

# # response = model.invoke("What is LR(0) in compiler?")
# # print(response)
# # print("===========================")
# # print(response.content) # we only need content


# #Google Gemini

# from langchain.chat_models import init_chat_model
# model = init_chat_model("google_genai:gemini-2.5-flash-lite")
# response = model.invoke("What is LR(0) in compiler?")
# print(response.content)

# print()

# #Groq
# print("GROQ RESPONSE")
# from langchain.chat_models import init_chat_model
# model = init_chat_model(
#     "meta-llama/llama-4-scout-17b-16e-instruct",
#     model_provider="groq"  #this is very imp to mention
# )
# response = model.invoke("What is LR(0) in compiler?")
# print(response.content)

# #Mistral->It is free
# print()


# print("MISTRAL RESPONSE")
# from langchain.chat_models import init_chat_model
# #we can chat init too but there is this other method too
# from langchain_mistralai import ChatMistralAI

# model=init_chat_model(
#     "mistral-small-2506",
#     temperature=1, #tells creativeness of model
#     max_tokens=50
# )
# response = model.invoke("What is LR(0) in compiler?")
# print(response.content)


#USING HUGGING FACE MODELS
load_dotenv()
print("*********HUGGING FACE MODEL RESPONSE*********")
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# #remember to update the permissions in huugingface settings->tokens
# llm = HuggingFaceEndpoint(
#     repo_id="deepseek-ai/DeepSeek-R1"
# )
# model = ChatHuggingFace(llm=llm)
# response = model.invoke("Why do parrots talk?")
# print(response.content)

#using hugging face pipeline we can also download the model locally. but use carefully bcoz some times size can be huge
from langchain_huggingface import HuggingFacePipeline

#here in this example we are using a smaller model
llm = HuggingFacePipeline.from_model_id(
    model_id="distilgpt2",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 100,
        "top_k": 50,
        "temperature": 0.1,
    },
)

res=llm.invoke("What is AI?")
print(res)

