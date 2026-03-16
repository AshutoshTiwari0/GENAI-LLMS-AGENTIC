from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

# #create an embedding now
# vector=embeddings.embed_query("My name is Ashutosh Tiwari")


# print(vector)

texts=["Hello brother how are you",
       "I am good brother how are you",
       "I am also good"]

vector2=embeddings.embed_query(texts)

print(vector2)