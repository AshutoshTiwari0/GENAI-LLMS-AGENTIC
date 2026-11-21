from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load an example document
with open("sample.txt",'r', encoding='utf-8') as f:
    sample = f.read()

splitter=RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10
)

chunks=splitter.split_text(sample)

print(len(chunks))

print(chunks[0])