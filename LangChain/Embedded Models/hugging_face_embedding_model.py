from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
load_dotenv()

embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


texts=["Hello brother how are you",
       "I am good brother how are you",
       "I am also good"]

vector=embeddings.embed_documents(texts)

print(vector)