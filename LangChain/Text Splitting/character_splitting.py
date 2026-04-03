from langchain_text_splitters import CharacterTextSplitter

# Load an example document
with open("sample.txt",'r', encoding='utf-8') as f:
    sample = f.read()

text_splitter = CharacterTextSplitter(
    separator=".",
    chunk_size=10, #isse kuch nhi hota. Main chunking seperator ke basis pe hoti hai
    chunk_overlap=2,
)
chunks = text_splitter.create_documents([sample])
print(len(chunks))

a=1
for i in chunks:
    print(f'{a} Chunk: {i.page_content}')
    a+=1