from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()


# data=PyPDFLoader("C:\\Users\\Dell\\Desktop\\Generative AI\\LangChain\\RAG\\Document Loaders\\dlbook.pdf")
# docs=data.load()

# splitter=RecursiveCharacterTextSplitter(chunk_size=1000,
#                                         chunk_overlap=200)


# chunks=splitter.split_documents(docs)

template=ChatPromptTemplate.from_messages(
    [("system","you are an ai that summarizes text/pdf"),
     ("human","Summarize the following {data}")]
)

model=ChatMistralAI(model="mistral-small-2506")


