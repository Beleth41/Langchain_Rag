from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
from langchain_ollama.llms import OllamaLLM
from langchain_core.messages import HumanMessage, SystemMessage

persist_directory = "db/chroma_db"

embedding_model = OllamaEmbeddings(model="mxbai-embed-large:latest")

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)


query = "who is the founder of Microsoft?"

# retriever = db.as_retriever(search_kwargs={"k": 5})

retriever = db.as_retriever(
search_type="similarity_score_threshold",
search_kwargs={"score_threshold": 0.3, "k": 5}
)

relevant_docs = retriever.invoke(query)
print(f"user query: {query}")

# print("___context___")
# for i, doc in enumerate(relevant_docs, 1):
#     print(f"Document {i}:\n{doc.page_content}")


#LLM part

#combine query + retrieval
combined_input = f"""based on the following documents, please answer this question {query}

Documents:
{chr(10).join([f"-{doc.page_content}" for doc in relevant_docs])}

   please provide a clear, helpful answer using only the information from these documents. if you can't find the answer in the documents say "I don't have enough information to answer that question based on the provided."

"""

model = OllamaLLM(model="llama3.2:latest")


#define the messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content=combined_input)
]

result = model.invoke(messages)

print("content only")
print(result)

