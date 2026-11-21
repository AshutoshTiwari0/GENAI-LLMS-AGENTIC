"""
A company wants to build an AI powered tool which does the following:
1. Takes a raw paragraph about a movie
2. Extracts important structered information
3. Generates a clean summary
4. Returns it in JSON format
5.Stores it in their database
"""

from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI


load_dotenv()

#defining schema
class Movie(BaseModel):
    title:str 
    release_year:Optional[int] #may give or not but if given give in integer
    genre:List[str] #one movie can have many genres
    director:Optional[str]
    cast:List[str]
    rating:Optional[float]
    summary: str

#we need to check via parsers so that data type is correct or not

parser=PydanticOutputParser(pydantic_object=Movie)


model=ChatMistralAI(model="mistral-small-2506")


# prompt=ChatPromptTemplate.from_messages([
# ("system","""Give a structured summary of the movie "<Movie Name> (<Year>)".

# Include the following sections:

# 1. Overview:
# Provide a concise 3–4 line summary of the movie’s premise.

# 2. Key Plot Points:
# - Beginning (setup of the story)
# - Main journey / conflict
# - Major turning points (twists, challenges)
# - Climax and resolution

# 3. Main Characters:
# List important characters with a brief role description.

# 4. Key Themes:
# Highlight 3–5 major themes explored in the movie.

# 5. Notable Concepts (if applicable):
# Explain any important scientific, philosophical, or unique ideas used in the movie.

# 6. Ending Explanation (optional):
# Briefly explain the ending in simple terms.

# Keep the explanation clear, structured, and easy to understand."""),
# (
#     "human",
#     """Give summary of below paragraph:
#     {paragraph}
#     """
# )
# ])

#now since we have made pydantic we need a new prompt template

prompt=ChatPromptTemplate.from_messages([
    ("system","""
Extract movie information from paragraphs 
{format_instructions}"""
),("human","{paragraph}")]
)   #ham ek instruction dege then para dege



paragraph=input("Provide info about the paragraph:")


final_prompt=prompt.invoke({
    {"paragraph": paragraph,
     "format_instructions":parser.get_format_instructions()}
})

response=model.invoke(final_prompt)

print(response.content)