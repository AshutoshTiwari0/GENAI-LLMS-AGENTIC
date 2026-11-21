#arxiv pe bahut hard hoga paper dhundhna. so we will use from here
from langchain_community.retrievers import ArxivRetriever

retriever = ArxivRetriever(
    load_max_docs=2, #number of papers to retrieve. exactly nhi but "try to return up to k docs"
    get_full_documents=True,
)

# docs = retriever.invoke("2604.02330") #ye paper ki id hia jisse retrieve krna hai

#if we want multiple paper we need to specify the topic
docs = retriever.invoke("Logisitc Regression")
# print(docs[0].metadata.get("Title"))

for i,doc in enumerate(docs):
    print(f"\nResult {i+1}")
    print("Title:", doc.metadata.get("Title"))
    print("Author:", doc.metadata.get("Author"))
    print("Summary:", doc.page_content[:500])
    print()

