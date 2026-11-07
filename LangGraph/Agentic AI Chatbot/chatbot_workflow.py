# %%
from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Literal,Annotated
import operator
from langchain_core.messages import SystemMessage,HumanMessage,BaseMessage #we will be using LLM for every node so for that we need to set prompt
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


# %%
from langgraph.checkpoint.memory import MemorySaver  #used to store messages in side lang graph. It will only be erased when we will reload the kernel

# %%
load_dotenv()

# %%
llm=ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=1.0
)

# %%
from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]  #base message is like chat history [user prompt +response]

# %%
def chat_node(state:ChatState):
    response=llm.invoke(state['messages']) #semd to LLM
    return {'messages':[response]}


# %%
#add checkpoin to save the messages
checkpoint=MemorySaver()

graph=StateGraph(ChatState)

#add nodes
graph.add_node('chat_node',chat_node)

#add edges 
graph.add_edge(START,'chat_node')

#add edges 
graph.add_edge('chat_node',END)

#execute graph
chatbot=graph.compile(checkpointer=checkpoint)

# %%
chatbot

# %%
initial_state={
    'messages':[HumanMessage(content='What is Object Oriented Programming?')]
}

# %%
response=chatbot.invoke(initial_state, config={"configurable": {"thread_id": "1"}})

# %%
response['messages'][-1].content

# %%
#making a loop so that user can keep on chatting

# %% [markdown]
# using thread we can make different chat sections 

# %%
thread_id="1"   #make different chats like thread 1 is of x topic thread 2 is of y topic

while True:
    user_message=input("type here...")
    print('user: ',user_message)

    if user_message.strip().lower() in ['exit','quit','bye']:
        break

    else: 
        config={'configurable':{'thread_id':thread_id}}  
        response=chatbot.invoke({'messages':[HumanMessage(content=user_message)]},config=config)
        print('AI:',response['messages'][-1].content)

# %%
chatbot.get_state(config=config)

# %% [markdown]
# Streaming Responses in LangGraph

# %% [markdown]
# Generating output token by token instead of all together

# %%
config={'configurable':{'thread_id':'thread-1'}}

# %%
res=chatbot.invoke({'messages':[HumanMessage(content="Generate a blog about Capital of India?")]},config=config)
print(res['messages'][-1].content)

# %% [markdown]
# using streaming 

# %%
response=chatbot.stream(
    {'messages':[HumanMessage(content="Generate a blog about Capital of India?")]},
    config=config,
    stream_mode='messages'
)

# %%
print(response)

# %%
for message_chunk, metadata in chatbot.stream(
    {'messages':[HumanMessage(content="Generate a blog about Capital of India?")]},
    config=config,
    stream_mode='messages'
):
    if message_chunk.content:
        print(message_chunk.content,end='',flush=True)

# %% [markdown]
# Chat Threading

# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%



