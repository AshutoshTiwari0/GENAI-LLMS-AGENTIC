from langchain_community.document_loaders import WebBaseLoader

url="https://en.wikipedia.org/wiki/Ferrari"

data=WebBaseLoader(url)

doc=data.load()

print(len(doc))
