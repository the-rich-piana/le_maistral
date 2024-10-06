import streamlit as st
import os
import json
from utils import process_files_recursively, process_text_files_recursively, process_youtube_links
from uuid import uuid4
from mistralai import Mistral
from langchain.output_parsers import PydanticOutputParser
from langchain import PromptTemplate
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

# Load environment variables
load_dotenv()

# Set a base directory for storing groups and files
BASE_DIR = 'uploaded_groups'

# Create the base directory if it does not exist
if not os.path.exists(BASE_DIR):
    os.mkdir(BASE_DIR)

# Initialize Mistral model
@st.cache_resource  # Cache the model for faster load times
def load_mistral_model():
    return ChatMistralAI(
        model="mistral-large-latest",
        temperature=0.7,
        max_retries=2,
    )

# Initialize Chroma vector store
@st.cache_resource
def initialize_vector_store():
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    return Chroma(
        collection_name="document_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",
    )

# Load and chunk documents
def load_and_chunk_documents(directory_path, chunk_size=500, chunk_overlap=50):
    documents = []
    for root, _, files in os.walk(directory_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            # Only process .txt files that don't end in 'summary.txt'
            if file_name.lower().endswith('.txt') and not file_name.lower().endswith('summary.txt'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        if content.strip():
                            documents.append({"content": content, "source": file_path})
                except Exception as e:
                    st.write(f"Error reading {file_path}: {e}")

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

# Function to scan directory and add documents to the vector store
def add_existing_txt_documents_to_vector_store(group_path):
    # Load and chunk documents in the specified group directory
    chunked_documents = load_and_chunk_documents(group_path)
    if chunked_documents:
        uuids = [str(uuid4()) for _ in range(len(chunked_documents))]
        vector_store.add_documents(documents=chunked_documents, ids=uuids)

# Function to get a list of existing groups (folders)
def get_existing_groups():
    return [name for name in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, name))]

# Initialize models
mistral_model = load_mistral_model()
vector_store = initialize_vector_store()

def generate_response_with_context(query, group_path):
    # Ensure the latest documents are added to the vector store
    add_existing_txt_documents_to_vector_store(group_path)

    # Retrieve relevant document chunks
    results = vector_store.similarity_search(query, k=2)
    context = "\n\n".join([res.page_content for res in results])
    
    # Create the prompt with context
    prompt_template = PromptTemplate(
        template="You are a tutor. Answer the user's query based on the context provided, try to give your answer as close as possible to what is in the context:\n\nContext:\n{context}\n\nUser Query: {query}",
        input_variables=["context", "query"]
    )
    prompt_text = prompt_template.format(context=context, query=query)

    print(prompt_text)
    
    # Invoke the model
    response = mistral_model.invoke(prompt_text)
    response_text = response.content if hasattr(response, 'content') else str(response)
    
    return response_text.strip()

# Streamlit Interface
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for group management
st.sidebar.title("Group Management")
group_name = st.sidebar.text_input("Create New Group", "")
create_group_button = st.sidebar.button("Create Group")

# Create a new group if the button is clicked
if create_group_button:
    if group_name:
        group_path = os.path.join(BASE_DIR, group_name)
        if not os.path.exists(group_path):
            os.mkdir(group_path)
            st.sidebar.success(f"Group '{group_name}' created!")
        else:
            st.sidebar.error(f"Group '{group_name}' already exists!")
    else:
        st.sidebar.error("Please enter a group name.")

# Select an existing group
existing_groups = get_existing_groups()
selected_group = st.sidebar.selectbox("Select Group", existing_groups)

@st.cache_data
# New function to load processed data
def load_processed_data():
    process_data_path = os.path.join(BASE_DIR, 'process_data.txt')
    if os.path.exists(process_data_path):
        with open(process_data_path, 'r') as f:
            return json.load(f)
    return []

# New function to save processed data
def save_processed_data(processed_data):
    process_data_path = os.path.join(BASE_DIR, 'process_data.txt')
    with open(process_data_path, 'w') as f:
        json.dump(processed_data, f)

# New function to process input (placeholder)
def process_input(uploaded_groups_path: str, processed_data: List[str], language: Optional[str] = "English"):
    # Placeholder for processing logic
    process_files_recursively(uploaded_groups_path, processed_data)
    process_youtube_links(uploaded_groups_path, processed_data)
    process_text_files_recursively(uploaded_groups_path, processed_data, language)

# Load processed data at the start
processed_data = load_processed_data()

# New function to load JSON file
def load_json(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return {}

# New function to map keys
def map_keys(mapping, url_mapping):
    mapped_mapping = {}
    for key, value in mapping.items():
        mapped_key = url_mapping.get(key, key)
        mapped_mapping[mapped_key] = value
    
    return mapped_mapping

# Load and map the keys
mapping = load_json(os.path.join(BASE_DIR, 'mapping.json'))
url_mapping = load_json(os.path.join(BASE_DIR, 'url_mapping.json'))
mapped_mapping = map_keys(mapping, url_mapping)

# Main section for file uploads and link management
st.title(f"File and Link Management in Group: {selected_group}")

if selected_group:
    processed_data = load_processed_data()
    mapping = load_json(os.path.join(BASE_DIR, 'mapping.json'))
    url_mapping = load_json(os.path.join(BASE_DIR, 'url_mapping.json'))
    mapped_mapping = map_keys(mapping, url_mapping)
    # Add a text input for language selection
    language = st.text_input("Select language (default: English)", value="English")

    # File upload section
    st.header("File Upload")
    group_path = os.path.join(BASE_DIR, selected_group)
    
    uploaded_file = st.file_uploader(f"Upload a file to '{selected_group}'", type=["csv", "txt", "pdf", "jpg", "png"])
    
    if uploaded_file:
        file_path = os.path.join(group_path, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File '{uploaded_file.name}' uploaded to group '{selected_group}'!")

        # Add the new file to processed_data
        file_path_no_ext = file_path.split('.')[0]
        
        # Call process_input function
        process_input(BASE_DIR, processed_data, language)
        processed_data.append(file_path_no_ext)
        save_processed_data(processed_data)

    # Link management section
    st.header("Link Management")
    new_link = st.text_input("Add a new link:")
    add_link_button = st.button("Add Link")

    if add_link_button and new_link:
        group_path = os.path.join(BASE_DIR, selected_group)
        links_file_path = os.path.join(group_path, "links.txt")
        
        # Append the new link to links.txt
        with open(links_file_path, "a") as f:
            f.write(new_link + "\n")
        st.success(f"Link added to '{selected_group}'!")

        # Add the new link to processed_data

        
        # Call process_input function with the selected language
        process_input(BASE_DIR, processed_data, language)
        processed_data.append(new_link)
        save_processed_data(processed_data)

    # List files and links in the selected group
    st.subheader(f"Files and Links in '{selected_group}'")
    group_path = os.path.join(BASE_DIR, selected_group)
    group_files = os.listdir(group_path)
    
    if group_files:
        for file_name in group_files:
            if file_name == "links.txt":
                st.write("Links:")
                with open(os.path.join(group_path, file_name), "r") as f:
                    links = f.readlines()
                for link in links:
                    link = link.strip()
                    if link in mapped_mapping:
                        st.write(f"- {link.strip('.txt')}")
            elif file_name in mapped_mapping:
                print(file_name)
                st.write(f"File: {file_name.strip('.txt')}")
                
                # Display summary for the file
                st.write("Summary:")
                file_path = os.path.join(os.sep.join(group_path.split('/')[:-1]), mapped_mapping[file_name])
                print(group_path, file_path)
                with open(file_path, "r") as f:
                    content = f.read()
                
                # Truncate content if it's too long
                max_chars = 500
                truncated_content = content[:max_chars] + "..." if len(content) > max_chars else content
                
                # Display truncated content in a text area
                st.text_area("", value=truncated_content, height=150, key=f"summary_{file_name}")
                
                # Add an expander for full content
                with st.expander("Show full content"):
                    st.text_area("", value=content, height=300, key=f"full_{file_name}")
    else:
        st.write("No files or links in this group.")
        st.write("No files in this group.")

    # Chat-like interface for user interaction with documents
    st.subheader("Chat with a Tutor")
    if group_files:
        user_input = st.chat_input("Ask a question about the uploaded documents:")

        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Generate response using Mistral model with context
            response = generate_response_with_context(user_input, group_path)
            
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Display chat history
        for chat in st.session_state.chat_history:
            st.chat_message(chat["role"]).write(chat["content"])
