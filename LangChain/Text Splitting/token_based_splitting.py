from langchain_text_splitters import TokenTextSplitter

# Load an example document
with open("sample.txt",'r', encoding='utf-8') as f:
    sample = f.read()

splitter=TokenTextSplitter(
    chunk_size=100,
    chunk_overlap=10
)

chunks=splitter.split_text(sample)

print(len(chunks)) #7

print(chunks[0])