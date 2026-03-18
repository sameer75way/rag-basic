import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()


# ✅ Clean unwanted text (important for Wikipedia data)
def clean_text(text: str) -> str:
    unwanted = [
        "WikipediaThe Free Encyclopedia",
        "Donate",
        "Create account",
        "Log in"
    ]

    for word in unwanted:
        text = text.replace(word, "")

    return text


# ✅ Load documents
def load_documents(docs_path="docs"):
    print(f"📂 Loading documents from: {docs_path}")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Path '{docs_path}' does not exist.")

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader
    )

    documents = loader.load()

    if not documents:
        raise ValueError("No .txt files found in docs folder")

    # Clean text
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    print(f"✅ Loaded {len(documents)} documents")

    return documents


# ✅ Split documents (FIXED)
def split_documents(documents, chunk_size=500, chunk_overlap=50):
    print(f"✂️ Splitting into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    print(f"✅ Created {len(chunks)} chunks")

    return chunks


# ✅ Create vector DB
def create_vector_store(chunks, persist_directory="db/chroma_db"):
    print(f"📦 Creating vector DB at: {persist_directory}")

    os.makedirs(persist_directory, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )


    print("✅ Vector DB created and saved")

    return vector_store


# ✅ Main pipeline
def main():
    print("🚀 Starting ingestion pipeline...\n")

    documents = load_documents()
    chunks = split_documents(documents)
    create_vector_store(chunks)

    print("\n🎉 Ingestion complete!")


if __name__ == "__main__":
    main()