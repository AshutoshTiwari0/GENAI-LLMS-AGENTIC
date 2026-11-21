#ye file embeddings bana ke seedhe db mai dal degi

from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings #we can use any other model embedding too like openAi
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()


#load pdf
data=PyPDFLoader("C:\\Users\\Dell\\Desktop\\Generative AI\\LangChain\\RAG\\Document Loaders\\dlbook.pdf")
docs=data.load()

#split into chunks

splitter=RecursiveCharacterTextSplitter(chunk_size=1000,
                                        chunk_overlap=200)


chunks=splitter.split_documents(docs)

#create embeddings
embedding_model=MistralAIEmbeddings()

vectorstore=Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)
#store in chroma db
