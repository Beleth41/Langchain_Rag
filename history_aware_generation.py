from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM

persistent_directory = "db/chroma_db"
embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

model = OllamaLLM(model="llama3.2:latest")

chat_history = []

def ask_question(user_question):
    if chat_history:
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question"),
        ] + chat_history + [
            HumanMessage(content=user_question)
        ]

        result = model.invoke(messages)
        search_question = result.strip()
        print(f"search question: {search_question}")

    else:
        search_question = user_question

    retriever = db.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(search_question)

    combined_input = f"""Based on the following documents, please answer this question: {user_question}

    Documents:
    {chr(10).join([f" -{doc.page_content}" for doc in docs])}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
    """

    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on the provided documents and conversation history."),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]

    result = model.invoke(messages)
    answer = result

    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print(answer)

def start_chat():
    print("Welcome to the chat! Type 'quit' to quit.")
    while True:
        question = input("\nYour question: ")
        if question.lower() == "quit":
            print("Exiting the chat. Goodbye!")
            break
        ask_question(question)

if __name__ == "__main__":
    start_chat()