import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
# from langchain_ollama.llms import OllamaLLM
from langchain_chroma import Chroma
from dotenv import load_dotenv
from pydoc import doc


#retrieve files
def load_documents(docs_path="docs"):
    print(f"loading documents from {docs_path}")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Directory {docs_path} does not exist.")

    loader = DirectoryLoader(
        path = docs_path,
        glob = "*.txt",
        loader_cls = TextLoader,
        loader_kwargs = {"encoding": "utf-8"}
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in directory {docs_path}.")

    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i + 1}:")
        print(f"Source: {doc.metadata['source']}")
        print(f"Content length: {len(doc.page_content)} characters")
        print(f"content preview: {doc.page_content[:100]}...")
        print(f"metadata: {doc.metadata}")

    return documents



#chunking files
def split_documents(documents, chunk_size=800, chunk_overlap=0):
    print("splitting documents into chunks")

    #CharacterTextSplitter
    # text_splitter = CharacterTextSplitter(
    #     chunk_size=chunk_size,
    #     chunk_overlap=chunk_overlap,
    # )
    
    #RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        separators = ["\n\n", "\n", "."]
    )

    # #SemanticChunker
    # text_splitter = SemanticChunker(
    #     embedding=OllamaEmbeddings(model="mxbai-embed-large:latest"),
    #     breakpoint_threshold_type="percentile", #or "standard_deviation"
    #     breakpoint_threshold_amount=70 #70 is best because it finds the best in finding the weakest relative connection between sentences.
    # )

    #----
    chunks = text_splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n-- chunk {i+1} --")
            print("source:", chunk.metadata['source'])
            print("content length:", len(chunk.page_content), "characters")
            print("content preview:", chunk.page_content)
            print("-" * 50)

        if len(chunks) > 5:
            print(f"and {len(chunks)-5} more chunks")

        sum=0
        for i, chunk in enumerate(chunks):
            if len(chunk.page_content) > chunk_size:
                sum += 1
                print(f"Warning: Chunk {i+1} exceeds the specified chunk size of {chunk_size} characters.")

        print(f"\n-----\n{sum} chunks exceeds the specified chunk size of {chunk_size} characters.")
    return chunks


#creating db
def create_vector_store(chunks, persist_directory="db/chroma_db", batch_size=25):
    print("creating database")

    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

    embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vectorstore = Chroma.from_documents(
            documents=batch,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_metadata={"hnsw:space": "cosine"}
        )
        print(f"Processed batch {i // batch_size + 1} of {len(chunks) // batch_size + 1}")

    print("vector store created successfully")
    return vectorstore


#embedding files



#storing files

def main():
    print("main functions")
    documents = load_documents(docs_path="docs")
    chunks = split_documents(documents)
    vector_store = create_vector_store(chunks)

if __name__ == "__main__":
    main()