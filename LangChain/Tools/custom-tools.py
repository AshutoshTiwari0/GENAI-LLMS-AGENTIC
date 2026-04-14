from langchain.tools import tool

@tool
def get_greeting(name:str)->str:
    """
    Generate a greeting message for user
    """
    return f"Hello {name}! How are you?"

#Our function get_greeting() is now a tool and a tool is a runnable so our function is a runnbale so we can use .invoke() directly.

result=get_greeting.invoke({"name":"Ashutosh"})

print(result)

#getting info about our tool

print(get_greeting.name) #print name of tool
print(get_greeting.description) #get the doc string(explanation) of the function
print(get_greeting.args)#arguments of the function
