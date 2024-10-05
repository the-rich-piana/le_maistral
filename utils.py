from os.path import exists
from pdf2image import convert_from_path
import os 
import base64
import requests
import os
from mistralai import Mistral
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_mistralai import ChatMistralAI

load_dotenv()

# Define the data structure for the model's output
class PageProcessorOutput(BaseModel):
    Index: List[str] = Field(description="Table of Contents: List of headings/sub-headings")
    FilledContent: Dict[str, str] = Field(description="Filled content based on the generated table of contents")
    UnassignedText: str = Field(description="Text that could not be assigned to any table of contents section")

def encode_image(image_path):
    """Encode the image to base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: The file {image_path} was not found.")
        return None
    except Exception as e:  # Added general exception handling
        print(f"Error: {e}")
        return None
    
# Function to process a list of content chunks
def process_pages(contents: List[str]) -> Dict[str, Any]:
    model = ChatMistralAI(
        model="mistral-large-latest",
        temperature=0,
        max_retries=2,
        # other params...
    )
    # Parser for the model's output
    parser = PydanticOutputParser(pydantic_object=PageProcessorOutput)

    # Prompt template for the language model
    prompt = PromptTemplate(
        template="""
    Given the current content chunk and the previous state, perform the following steps:

    1. **Update the Index (Table of Contents):**
    - **Append** new significant headings or sub-headings to the existing 'Index' based on the current content and unassigned text.
    - **Do not remove or modify** existing entries in the 'Index'.

    2. **Update FilledContent:**
    - **Add new content** to 'FilledContent' under the appropriate headings from the 'Index'.
    - **Do not overwrite** existing entries in 'FilledContent'.

    3. **Handle Unassigned Text:**
    - Identify any part of the content that could not be assigned to any section in the 'Index'.
    - Combine it with 'UnassignedText' for the next iteration.

    4. **Consider Next Content:**
    - Use 'next_content' to anticipate upcoming topics and adjust the 'Index' if necessary.

    {format_instructions}

    Previous Index (if any):
    {last_index}

    Unassigned Text (if any):
    {unassigned_text}

    Current Content Chunk:
    {page_content}

    Next Content Chunk (for context):
    {next_content}
    """,
        input_variables=["page_content", "last_index", "unassigned_text", "next_content"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    last_index = []  # Initialize the last index as an empty list
    unassigned_text = ""  # Initialize unassigned_text as an empty string
    final_filled_content = {}

    for i, content in enumerate(contents):
        next_content = contents[i + 1] if i + 1 < len(contents) else ""

        # Prepare the inputs for the prompt
        inputs = {
            "page_content": content,
            "last_index": '\n'.join(last_index),
            "unassigned_text": unassigned_text,
            "next_content": next_content
        }

        # Generate the prompt text
        prompt_text = prompt.format(**inputs)

        # Invoke the language model
        output = model.invoke(prompt_text)

        # Parse the model's output
        result = parser.invoke(output)

        # Merge the new Index with the last Index, avoiding duplicates
        combined_index = last_index + [item for item in result.Index if item not in last_index]
        last_index = combined_index

        # Merge the FilledContent, adding new entries without overwriting existing ones
        for key, value in result.FilledContent.items():
            if key in final_filled_content:
                # Append new content to existing content under the same heading
                final_filled_content[key] += "\n" + value
            else:
                final_filled_content[key] = value

        # Update unassigned_text for the next iteration
        unassigned_text = result.UnassignedText.strip()
        print("Index: ", result.Index)
    print("Index: ", last_index)
    print("Filled Content: ", final_filled_content)
    print('\n')
    # Return the final index and filled content
    return {"Index": last_index, "FilledContent": final_filled_content}
    
def convert_pdf_to_images(pdf_path):
    folder_path = pdf_path.replace('.pdf', '_pdf')
    os.makedirs(folder_path, exist_ok=True)
    
    # Store Pdf with convert_from_path function
    images = convert_from_path(pdf_path)

    for i, image in enumerate(images):
        # Save pages as images in the pdf
        image.save(os.path.join(folder_path, f'page{i}.jpg'), 'JPEG')

    return folder_path

def convert_image_to_text(image_path, model="pixtral-12b-2409"):
    # Getting the base64 string
    base64_image = encode_image(image_path)
    api_key = os.environ["MISTRAL_API_KEY"]
    client = Mistral(api_key=api_key)

    if base64_image is None:
        return "Error: Failed to encode image."

    # Define the messages for the chat
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Convert the image into text, do not miss on anything."
                },
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}" 
                }
            ]
        }
    ]

    # Get the chat response
    chat_response = client.chat.complete(
        model=model,
        messages=messages
    )

    # Write the content of the response to a .txt file
    output_path = image_path.rsplit('.', 1)[0] + '.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(chat_response.choices[0].message.content)

    return f"Text written to {output_path}"

def combine_text_files(folder_path):
    """
    Combines all text files in a given folder into one large text file.
    
    Args:
    folder_path (str): Path to the folder containing text files.
    
    Returns:
    str: Path to the combined text file.
    """
    # Get all files in the folder and sort them by name
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.txt')])
    
    # Get the name of the current folder
    current_folder_name = os.path.basename(folder_path).rstrip('_pdf')
    
    # Create a new file to store the combined text in the parent directory
    parent_directory = os.path.dirname(folder_path)
    combined_file_path = os.path.join(parent_directory, f'{current_folder_name}.txt')
    
    with open(combined_file_path, 'w', encoding='utf-8') as outfile:
        for filename in files:
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read() + '\n')
    
    # Recursively remove the folder_path
    import shutil
    shutil.rmtree(folder_path)
    
    return combined_file_path

def process_files_recursively(root_dir):
    """
    Recursively process files in the given directory.
    
    Args:
    root_dir (str): Root directory to start processing from.
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            file_extension = os.path.splitext(filename)[1].lower()

            if file_extension == '.txt':
                # Leave text files as they are
                continue
            elif file_extension in ['.png', '.jpeg', '.jpg', '.webp', '.gif']:
                # Process image files
                convert_image_to_text(file_path)
            elif file_extension == '.pdf':
                # Process PDF files
                pdf_images_folder = convert_pdf_to_images(file_path)
                for image_file in os.listdir(pdf_images_folder):
                    image_path = os.path.join(pdf_images_folder, image_file)
                    if image_path.lower().endswith(('.png', '.jpeg', '.jpg', '.webp', '.gif')):
                        convert_image_to_text(image_path)
                combine_text_files(pdf_images_folder)

def process_text_files_recursively(root_dir: str):
    """
    Recursively process text files in the given directory.
    
    Args:
    root_dir (str): Root directory to start processing from.
    model: The language model to use for processing.
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.txt'):
                file_path = os.path.join(dirpath, filename)
                
                # Read the content of the text file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Create text splitter
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=5000,
                    chunk_overlap=0,
                    length_function=len,
                    is_separator_regex=False,
                    separators=[
                        "\n\n",
                        "\n",
                        " ",
                        ".",
                        ",",
                        "\u200b",  # Zero-width space
                        "\uff0c",  # Fullwidth comma
                        "\u3001",  # Ideographic comma
                        "\uff0e",  # Fullwidth full stop
                        "\u3002",  # Ideographic full stop
                        "",
                    ],
                )
                
                # Split the content into chunks
                chunks = text_splitter.create_documents([content])
                chunk_texts = [chunk.page_content for chunk in chunks]
                
                # Process the chunks
                result = process_pages(chunk_texts)
                
                # Generate the summary content
                summary_content = "Index:\n"
                for item in result["Index"]:
                    summary_content += f"- {item}\n"
                summary_content += "\nFilled Content:\n"
                for key, value in result["FilledContent"].items():
                    summary_content += f"{key}:\n{value}\n\n"
                
                # Write the summary to a new file
                summary_file_path = file_path.rsplit('.', 1)[0] + '_summary.txt'
                with open(summary_file_path, 'w', encoding='utf-8') as f:
                    f.write(summary_content)
                
                print(f"Summary written to: {summary_file_path}")

process_files_recursively('/Users/qazisaad/Projects/le_maistral/uploaded_groups')
process_text_files_recursively('/Users/qazisaad/Projects/le_maistral/uploaded_groups')