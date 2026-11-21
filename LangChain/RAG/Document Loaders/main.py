from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_core.documents import Document
load_dotenv()


# data=PyPDFLoader("C:\\Users\\Dell\\Desktop\\Generative AI\\LangChain\\RAG\\Document Loaders\\dlbook.pdf")
# docs=data.load()

# splitter=RecursiveCharacterTextSplitter(chunk_size=1000,
#                                         chunk_overlap=200)


# chunks=splitter.split_documents(docs)

embedding_model=MistralAIEmbeddings()

vectorstore=Chroma(
    persist_directory="RAG_embeddings",
    embedding_function=embedding_model
)

retriever=vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":4, #get top 4 elements
        "fetch_k":10, #intially get 10 thne from mmr find 4
        "lambda_mult":0.5 #controlls diversity 
    }
)



template=ChatPromptTemplate.from_messages(
    [("system","you are an ai that answers questions from document. If answer is not present you tell i do not know."),
     ("human","Context:{context} Question:{question}")]
)

model=ChatMistralAI(model="mistral-small-2506")

print("press 0 to exit")

while True:
    query=input("You :")
    if query=="0":
        break

    docs=retriever.invoke(query)

    context="\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt=template.invoke({
        "context":context,
        "question":query
    })

    response=model.invoke(final_prompt)

    print(f"\n AI: {response.content}")