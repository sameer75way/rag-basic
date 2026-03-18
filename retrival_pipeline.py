import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

load_dotenv()


# ✅ 1. Load Vector Store
def load_vector_store(persist_directory="db/chroma_db"):
    print("📂 Loading vector database...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    print("✅ Vector DB loaded")

    return db


# ✅ 2. Create Retriever (IMPORTANT)
def create_retriever(db):
    print("🔍 Creating retriever...")

    retriever = db.as_retriever(
        search_type="mmr",  # better than similarity
        search_kwargs={
            "k": 3,          # top results
            "fetch_k": 10    # candidates before rerank
        }
    )

    return retriever


# ✅ 3. Load LLM (FREE LOCAL)
def create_llm():
    print("🤖 Loading local LLM...")

    pipe = pipeline(
        "text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    return llm


# ✅ 4. Format Context
def format_context(docs):
    return "\n\n".join([doc.page_content for doc in docs])


# ✅ 5. RAG QA Function
def ask_question(retriever, llm, query):
    print(f"\n🧠 Query: {query}")

    # Step 1: Retrieve
    docs = retriever.invoke(query)

    print(f"📄 Retrieved {len(docs)} chunks")

    # Step 2: Build context
    context = format_context(docs)

    # Step 3: Prompt
    prompt = f"""
You are a helpful AI assistant.

Answer ONLY from the provided context.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{query}

Answer:
"""

    # Step 4: Generate answer
    response = llm.invoke(prompt)

    return response


# ✅ 6. CLI Loop
def main():
    db = load_vector_store()
    retriever = create_retriever(db)
    llm = create_llm()

    print("\n💬 Ask questions (type 'exit' to quit)\n")

    while True:
        query = input("👉 You: ")

        if query.lower() == "exit":
            break

        answer = ask_question(retriever, llm, query)

        print(f"\n🤖 AI: {answer}\n")


if __name__ == "__main__":
    main()