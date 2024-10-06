import streamlit as st
import os
import json
from utils import process_files_recursively, process_text_files_recursively, process_youtube_links

# Set a base directory for storing groups and files
BASE_DIR = 'uploaded_groups'

# Create the base directory if it does not exist
if not os.path.exists(BASE_DIR):
    os.mkdir(BASE_DIR)

# Function to get a list of existing groups (folders)
def get_existing_groups():
    return [name for name in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, name))]

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
def process_input(uploaded_groups_path, processed_data):
    # Placeholder for processing logic
    process_files_recursively(uploaded_groups_path, processed_data)
    process_youtube_links(uploaded_groups_path, processed_data)
    process_text_files_recursively(uploaded_groups_path, processed_data)

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
    # File upload section
    st.header("File Upload")
    uploaded_file = st.file_uploader(f"Upload a file to '{selected_group}'", type=["csv", "txt", "pdf", "jpg", "png"])

    if uploaded_file:
        # Save uploaded file to the selected group's folder
        group_path = os.path.join(BASE_DIR, selected_group)
        file_path = os.path.join(group_path, uploaded_file.name)
        
        # Write the file to the group's folder
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File '{uploaded_file.name}' uploaded to group '{selected_group}'!")

        # Add the new file to processed_data
        file_path_no_ext = file_path.split('.')[0]
        processed_data.append(file_path_no_ext)
        save_processed_data(processed_data)
        
        # Call process_input function
        process_input(BASE_DIR, processed_data)

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
        processed_data.append(new_link)
        save_processed_data(processed_data)
        
        # Call process_input function
        process_input(BASE_DIR, processed_data)

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
