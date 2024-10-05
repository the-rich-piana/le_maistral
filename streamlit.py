import streamlit as st
import os
from mistralai import Mistral
from langchain.output_parsers import PydanticOutputParser
from langchain import PromptTemplate
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

# Set a base directory for storing groups and files
BASE_DIR = 'uploaded_groups'

# Create the base directory if it does not exist
if not os.path.exists(BASE_DIR):
    os.mkdir(BASE_DIR)

# Initialize Mistral model (adjust parameters as needed)
@st.cache_resource  # Cache the model for faster load times
def load_mistral_model():
    return ChatMistralAI(
        model="mistral-large-latest",
        temperature=0.7,
        max_retries=2,
        # Add other parameters if needed...
    )

mistral_model = load_mistral_model()

def generate_response_mistral(query):
    # Define a basic prompt template (adjust to your needs)
    prompt_template = PromptTemplate(
        template="Answer the user's query based on the given information:\n\nUser Query: {query}",
        input_variables=["query"]
    )
    
    # Format the prompt
    prompt_text = prompt_template.format(query=query)
    
    # Invoke the model
    response = mistral_model.invoke(prompt_text)
    
    # Extract the text content from the AIMessage object
    response_text = response.content if hasattr(response, 'content') else str(response)
    
    return response_text.strip()

# Function to get a list of existing groups (folders)
def get_existing_groups():
    return [name for name in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, name))]

# Initialize session state for chat history
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
selected_group = st.sidebar.selectbox("Select Group", [""] + existing_groups)

# Main section for file uploads
st.title(f"File Management in Group: {selected_group}")

if selected_group:
    uploaded_file = st.file_uploader(f"Upload a file to '{selected_group}'", type=["csv", "txt", "pdf", "jpg", "png"])

    if uploaded_file:
        # Save uploaded file to the selected group's folder
        group_path = os.path.join(BASE_DIR, selected_group)
        file_path = os.path.join(group_path, uploaded_file.name)
        
        # Write the file to the group's folder
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File '{uploaded_file.name}' uploaded to group '{selected_group}'!")

    # List files in the selected group
    st.subheader(f"Files in '{selected_group}'")
    group_files = os.listdir(os.path.join(BASE_DIR, selected_group))
    if group_files:
        for file_name in group_files:
            st.write(file_name)
    else:
        st.write("No files in this group.")

    # Chat-like interface for user interaction with documents
    st.subheader("Chat with a Tutor")
    if group_files:
        # Chat input
        user_input = st.chat_input("Ask a question about the uploaded documents:")

        if user_input:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Generate response using Mistral model
            response = generate_response_mistral(user_input)
            
            # Add response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Display chat history
        for chat in st.session_state.chat_history:
            st.chat_message(chat["role"]).write(chat["content"])
