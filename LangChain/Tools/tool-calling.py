from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage
#just for good formatting of output. Not releated to GENAI
from rich import print

@tool
def get_text_length(text:str)-> int:
    """Returns number of characters in a given text"""
    return len(text)

tools={
    "get_text_length":get_text_length
}

llm=ChatMistralAI(model="mistral-small-2506")

result_no_tool=llm.invoke("Return number of charcaters in given text:'Hello how are you?'  ")
print("NO TOOL")
print()
print(result_no_tool)

#tool binding -> connect tool with LLM
llm_with_tool=llm.bind_tools([get_text_length])

result_with_tool=llm_with_tool.invoke("Return number of charcaters in given text:'Hello how are you?' ")
print("WITH TOOL")
print()
print(result_with_tool)

#tool calling-> LLM chooses to use the tool
#iska code nhi hota. ye llm khud krta hai

#tool execution->LLM execute the tool
# print(result_with_tool.tool_calls[0])
if result_with_tool.tool_calls:
    tool_call=result_with_tool.tool_calls[0]

tool_name=tool_call['name']
tool_args=tool_call['args']

result_with_tool=get_text_length.invoke(tool_args)

final_response=llm_with_tool.invoke(f"The length of text is {result_with_tool}")
print(final_response)

message=[]
query=HumanMessage("Return the number of characters in given text: 'Hello How are you?'")
message.append(query)

result=llm_with_tool.invoke(message)

message.append(result)

# print(message)

if result.tool_calls:
    #print(result.tool_calls[0])
    tool_name=result.tool_calls[0]["name"]
    tool_message=tools[tool_name].invoke(result.tool_calls[0])
    # tool_name.invoke()
    # print(tool_message)
    message.append(tool_message)

res=llm_with_tool.invoke(message)
print(res.content)