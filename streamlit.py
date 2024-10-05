import streamlit as st
import os

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
