"""
it is a retrieval algorithm used to retrieve documents.

Iske khas baat ye hai ki agar  2 docs jo ki almost same baat kah rhe hai toh ye dono ko na return krke ek hi retun krti hai.

MAXIMUM MARGINAL RELEVANCE-> JINME MARGIN KA GAP BADA HOGA VOH HOGE MORE RELEVANT



EXAMPLE->
CHUNK 1 GRADIENT DESCENT IS A OPTIMISATION TECHNIQUE

CHUNK 2 GRADIENT DESCENT MINIMISES THE LOSS FUNCTION

CHUNK 3 GRADIENT DESCENT IS A MINISATION TECHNQIUE USED FOR MINIMIXING THE LOSS FUNCTION


1,2 ARE AS SAME AS 3. SO YE ALGO 1 2 3 NA RETURN KRKE SIRF 1,2 YA SIRF 3 RETURN KREGI 


MAIN AIM-> "QUERY SE MATCH KRE BUT APAS MAI DIVERSIFIED HO"
"""


from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings



docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization that minimizes the loss function."),
    Document(page_content="Neural networks use gradient descent for training."),
    Document(page_content="Support Vector Machines are supervised learning algorithms.")
]


embeddings = HuggingFaceEmbeddings()


vectorstore = Chroma.from_documents(docs, embeddings)

#===================USING SIMILARITY SEARCH======================
similarity_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3}
)

print("\n===== Similarity Search Results =====\n")

similarity_docs = similarity_retriever.invoke("What is gradient descent?")

for doc in similarity_docs:
    print(doc.page_content)

# ===================USING MMR===============================

mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k":3}
)

print("\n===== MMR Results =====\n") #iske outputs diversified hote hai

mmr_docs = mmr_retriever.invoke("What is gradient descent?")

for doc in mmr_docs:
    print(doc.page_content)