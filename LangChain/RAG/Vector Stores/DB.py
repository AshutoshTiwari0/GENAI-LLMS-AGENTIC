from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()


#here we are making our own data. but we can aslo use pdf, web pages etc.
docs=[Document(page_content="Python is widely used in Artificial Intelligence.", metadata={"source": "AI_book"}),
    Document(page_content="Pandas is used for data analysis in Python.", metadata={"source": "DataScience_book"}),
    Document(page_content="Neural networks are used in deep learning.", metadata={"source": "DL_book"}),
]

embeddings = MistralAIEmbeddings(
    model="mistral-embed"
)

#make a vector store

vector_store=Chroma.from_documents(documents=docs,embedding=embeddings,persist_directory="chroma-db") #persist directory se ham apne local storgae mai store kr rhe embeddings


#kuch files bani hai extra jisme embeddings stored hai. data_level0.bin, header.bin, length.bin, link_lists.bin....... etc


#now we need to get info from DB
result=vector_store.similarity_search("What is neural network?",k=2) #return top 2(k) most similar embeddings


#vector store sirf retrieve krte hai info. Answer nhi krte hai. Answer LLM krte hai

for r in result:
    print(r.page_content)
    print()
    print(r.metadata)


retriever=vector_store.as_retriever()

docs=retriever.invoke("Explain deep learning?")

for d in docs:
    print(d.page_content)