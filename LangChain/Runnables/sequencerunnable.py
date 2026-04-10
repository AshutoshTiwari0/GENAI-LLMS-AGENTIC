from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

#WIHTOUT CHAINS
# prompt=ChatPromptTemplate.from_template("Explain {topic} to a 5 year old")

# model = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     temperature=1.0)

# parser=StrOutputParser()

# formatted_prompt=prompt.format_messages(topic="Machine Learning")

# #call model
# response=model.invoke(formatted_prompt)

# final_output=parser.parse(response.content)

# print(final_output)


#WITH CHAINS[DEPRECATED] (AB TOH SAB RUNNABLE HAI)
# from langchain.chains import LLMChains

# LLMChains(prompt,model,parser)


#USING VIA RUNNABLE
prompt=ChatPromptTemplate.from_template("Explain {topic} to a 5 year old")

model = ChatMistralAI(
    model="mistral-large-latest",
    temperature=1.0)

parser=StrOutputParser()

chain=prompt | model | parser

result=chain.invoke("Machine Learning")

print(result)

