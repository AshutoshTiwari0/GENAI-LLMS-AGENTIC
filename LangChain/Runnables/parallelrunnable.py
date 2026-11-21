from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
load_dotenv()


short_prompt=ChatPromptTemplate.from_template("Explain {topic} in simple terms")

detailed_prompt=ChatPromptTemplate.from_template("Explain {topic} in detailed manner and terms")


model = ChatMistralAI(
    model="mistral-large-latest",
    temperature=1.0)

parser=StrOutputParser()

short_prompt_template=short_prompt|model|parser
detailed_prompt_template=detailed_prompt|model|parser


final_prompt={"short":short_prompt_template, 
              "detailed":detailed_prompt_template}

#final_prompt is a dict currently. make it a runnable
parallel_runnable=RunnableParallel(final_prompt)

result=parallel_runnable.invoke({"topic":"Deep Learning"})

print(result)
print()
print("SHORT ANSWER")
print(result['short'])
print()
print("DETAILED ANSWER")
print()
print(result['detailed'])