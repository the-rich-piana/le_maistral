import os
from uuid import uuid4
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pdfplumber  # Using pdfplumber for PDF extraction

# Path to your directory containing documents
directory_path = "/Users/ysindi/personal_projects/le_maistral/uploaded_groups/Computer Science"

# Initialize Mistral embeddings
embeddings = MistralAIEmbeddings(
    model="mistral-embed",
)

# Initialize Chroma vector store
vector_store = Chroma(
    collection_name="computer_science_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Specify the directory where to persist vector data
)

# Function to read documents from the specified directory, including PDFs
def load_documents_from_directory(directory_path):
    documents = []
    for root, _, files in os.walk(directory_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if file_name.lower().endswith(('.txt', '.md')):  # For text files
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        if content.strip():  # Ensure the content is not empty
                            documents.append({"content": content, "source": file_path})
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
            elif file_name.lower().endswith('.pdf'):  # For PDF files
                try:
                    content = ""
                    with pdfplumber.open(file_path) as pdf:
                        for page in pdf.pages:
                            content += page.extract_text() or ""
                    if content.strip():  # Ensure the content is not empty
                        documents.append({"content": content, "source": file_path})
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return documents

# Function to chunk documents and convert to Document objects
def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ".", ","]
    )

    chunked_documents = []
    for doc in documents:
        chunks = text_splitter.split_text(doc["content"])
        for chunk in chunks:
            chunked_documents.append(Document(
                page_content=chunk,
                metadata={"source": doc["source"]}
            ))
    return chunked_documents

# Load and chunk documents
raw_documents = load_documents_from_directory(directory_path)
chunked_documents = chunk_documents(raw_documents)

# Ensure there are documents to add to the vector store
if not chunked_documents:
    print("No documents were found or processed. Exiting.")
else:
    # Generate UUIDs for the documents
    uuids = [str(uuid4()) for _ in range(len(chunked_documents))]

    # Add documents to the vector store
    vector_store.add_documents(documents=chunked_documents, ids=uuids)

    # Test query for similarity search
    test_query = "What are the basics of computer algorithms?"

    # Perform a similarity search in the vector database
    results = vector_store.similarity_search(
        test_query,
        k=3  # Number of results to return
    )

    # Print the results
    print(f"Query: {test_query}\n")
    print("Most relevant documents:")
    for i, res in enumerate(results):
        print(f"{i+1}. {res.page_content} [Source: {res.metadata['source']}]")
