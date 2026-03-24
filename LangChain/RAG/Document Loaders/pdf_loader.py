from langchain_community.document_loaders import PyPDFLoader

data=PyPDFLoader("oosd.pdf")

docs=data.load()

print(len(docs)) #we have 70 docs(pages). each doc will have meta data and page content

print(docs[-1]) #seeing last page or doc