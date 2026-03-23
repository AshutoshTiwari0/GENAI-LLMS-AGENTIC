"""
A company wants to build an AI powered tool which does the following:
1. Takes a raw paragraph about a movie
2. Extracts important structered information
3. Generates a clean summary
4. Returns it in JSON format
5.Stores it in their database
"""

from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

from langchain_mistralai import ChatMistralAI

model=ChatMistralAI(model="mistral-small-2506")

# response=model.invoke("""Interstellar (2014) is a sci-fi epic where ex-pilot Cooper joins a NASA mission to find a new planet through a wormhole, as Earth is dying from blight. Due to time dilation near a black hole, Cooper loses decades while his daughter helps solve the gravity equation needed for humanity's survival, eventually reuniting in the future.Key Plot Points
# Dying Earth (2067): Humanity faces extinction due to crop failure. Cooper is recruited to lead a team through a wormhole near Saturn to survey potential habitats.
# The Mission (Endurance): Cooper and his crew (Amelia Brand, Romilly, Doyle, TARS) explore planets near a supermassive black hole named Gargantua, which causes extreme time dilation.
# Time Dilation: Due to gravity, hours on Miller’s planet equate to years on Earth, causing Cooper to miss his children growing up.
# Betrayal & Sacrifice: Dr. Mann, a previous explorer, fakes data and tries to steal the ship. Cooper later plunges into Gargantua to send quantum data back to his daughter, Murph, on Earth.
# The Tesseract: Inside the black hole, Cooper enters a 5D Tesseract built by future humans, realizing he is the "ghost" who guided his past self and sent data to his daughter.
# Conclusion: Humanity, enabled by Murph solving the gravity equation, escapes Earth. A revived Cooper reunites with his elderly daughter in space. 

# Key Themes
# Time and Relativity: The film highlights how time is non-linear and affected by gravity.
# Love as a Physical Force: The connection between Cooper and his daughter is portrayed as a force that transcends time and space.
# Human Survival: The imperative to adapt and find a new home, sacrificing the present for the future.
                      
# Give key information about this movie?
# """)
# print(response.content)


#this above is not a good design. Use prompt template


prompt=ChatPromptTemplate.from_messages([
("system","""Give a structured summary of the movie "<Movie Name> (<Year>)".

Include the following sections:

1. Overview:
Provide a concise 3–4 line summary of the movie’s premise.

2. Key Plot Points:
- Beginning (setup of the story)
- Main journey / conflict
- Major turning points (twists, challenges)
- Climax and resolution

3. Main Characters:
List important characters with a brief role description.

4. Key Themes:
Highlight 3–5 major themes explored in the movie.

5. Notable Concepts (if applicable):
Explain any important scientific, philosophical, or unique ideas used in the movie.

6. Ending Explanation (optional):
Briefly explain the ending in simple terms.

Keep the explanation clear, structured, and easy to understand."""),
(
    "human",
    """Give summary of below paragraph:
    {paragraph}
    """
)
])
paragraph=input("Provide info about the paragraph:")


final_prompt=prompt.invoke({
    {"paragraph": paragraph}
})

response=model.invoke(final_prompt)

print(response.content)