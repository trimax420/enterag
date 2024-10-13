import streamlit as st
import os
import openai
import chromadb
import pandas as pd
import numpy as np
import PyPDF2
import json
from typing import List, Dict
from chromadb.utils import embedding_functions
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_option_menu import option_menu
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from bson import ObjectId
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB connection details
MONGODB_URI = "mongodb url"

def get_mongodb_connection():
    client = MongoClient(MONGODB_URI, server_api=ServerApi('1'))
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
        return client
    except Exception as e:
        print(f"An error occurred while connecting to MongoDB: {e}")
        return None

# Placeholder for OpenAI API Key
OPENAI_API_KEY = "openai api"
openai.api_key = OPENAI_API_KEY

# ChromaDB setup
chroma_client = chromadb.Client()
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI embedding model

# Helper function to extract text from a PDF file
def extract_text_from_pdf(file) -> str:
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text

# Helper function to chunk text into smaller parts
def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Updated function to get embeddings for a list of texts
def get_embeddings(texts: List[str]) -> List[np.ndarray]:
    response = openai.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    # Correctly access embeddings from the response
    return [np.array(embedding.embedding) for embedding in response.data]

# AIChatbot class that interacts with ChromaDB
class AIChatbot:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.collection = chroma_client.get_or_create_collection(name=collection_name)

    def generate_response(self, user_input: str):
        # Embed the user query
        query_embedding = get_embeddings([user_input])[0]

        # Retrieve the most relevant chunks from the collection
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=3  # Get top 3 most similar chunks
        )

        # Build context from the retrieved documents
        context = " ".join(results['documents'][0])

        # Generate a streaming response using GPT-4 with context
        response = openai.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {user_input}"}
            ],
            stream=True  # Enable streaming
        )
        return response

# Helper functions for managing collections in ChromaDB
def add_files_to_collection(collection_name: str, files: List):
    collection = chroma_client.get_or_create_collection(name=collection_name)

    for file in files:
        # Extract text from the PDF file
        text = extract_text_from_pdf(file)
        
        # Chunk the extracted text
        chunks = chunk_text(text)
        
        # Get embeddings for each chunk
        embeddings = get_embeddings(chunks)

        # Add chunks and embeddings to the collection
        collection.add(
            documents=chunks,
            embeddings=[emb.tolist() for emb in embeddings],
            ids=[f"{file.name}_{i}" for i in range(len(chunks))]
        )

    st.success(f"Added {len(files)} PDF files to collection '{collection_name}'.")

def delete_files_from_collection(collection_name: str, file_names: List[str]):
    collection = chroma_client.get_or_create_collection(name=collection_name)
    for file_name in file_names:
        # Get all chunk IDs for this file
        chunk_ids = [id for id in collection.get()['ids'] if id.startswith(f"{file_name}_")]
        if chunk_ids:
            # Delete chunks
            collection.delete(ids=chunk_ids)
            st.success(f"Deleted file '{file_name}' from collection '{collection_name}'.")
        else:
            st.warning(f"No chunks found for file '{file_name}' in collection '{collection_name}'.")

def delete_collection(collection_name: str):
    chroma_client.delete_collection(name=collection_name)
    st.success(f"Deleted collection '{collection_name}'.")

def list_collections() -> List[str]:
    return [collection.name for collection in chroma_client.list_collections()]

def list_files_in_collection(collection_name: str) -> List[str]:
    collection = chroma_client.get_or_create_collection(name=collection_name)
    all_ids = collection.get()['ids']
    return list(set([id.split('_')[0] for id in all_ids]))

# Manage Collections Page
def manage_collections():
    st.header("Manage Collections")
    collections = list_collections()

    # Use radio buttons for managing collections
    task = st.radio("What would you like to do?", ("Add Collection", "Modify Collection", "Delete Collection"))

    if task == "Add Collection":
        st.subheader("Add New Collection")
        new_collection_name = st.text_input("Collection Name")
        uploaded_files = st.file_uploader("Upload PDF Files", type="pdf", accept_multiple_files=True)

        if st.button("Add Collection"):
            if new_collection_name and uploaded_files:
                add_files_to_collection(new_collection_name, uploaded_files)

    elif task == "Modify Collection":
        st.subheader("Modify Existing Collection")
        selected_collection = st.selectbox("Select a Collection", collections)

        if selected_collection:
            st.write("Add new files:")
            new_files = st.file_uploader(f"Add PDF Files to {selected_collection}", type="pdf", accept_multiple_files=True)
            if st.button("Add Files"):
                if new_files:
                    add_files_to_collection(selected_collection, new_files)
            
            st.write("Delete existing files:")
            existing_files = list_files_in_collection(selected_collection)
            if existing_files:
                files_to_delete = st.multiselect("Select files to delete", existing_files)
                if st.button("Delete Selected Files"):
                    if files_to_delete:
                        delete_files_from_collection(selected_collection, files_to_delete)
                        st.success("Files deleted. Please refresh the page to see the updated file list.")
                        st.button("Refresh")  # Add a refresh button
            else:
                st.info("No files found in this collection.")

    elif task == "Delete Collection":
        st.subheader("Delete Collection")
        selected_collection = st.selectbox("Select a Collection to Delete", collections)

        if st.button("Delete Collection"):
            if selected_collection:
                delete_collection(selected_collection)

# AI Chatbot Page with RAG Pipeline
def chatbot_interface():
    st.header("AI Chatbot Interface")

    collections = list_collections()
    selected_collection = st.selectbox("Select a Collection", collections)

    if selected_collection:
        st.write(f"Using collection: {selected_collection}")

        # Initialize chatbot for the selected collection
        chatbot = AIChatbot(selected_collection)

        # Chat Interface using Streamlit's chat elements
        if 'messages' not in st.session_state:
            st.session_state['messages'] = []

        # Display chat history
        for message in st.session_state['messages']:
            with st.chat_message(message["role"]):
                st.write(message['content'])

        # User input
        user_input = st.chat_input("Type your message here...")

        if user_input:
            # Display user message
            with st.chat_message("user"):
                st.write(user_input)
            st.session_state['messages'].append({"role": "user", "content": user_input})

            # Get and display AI response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                for response in chatbot.generate_response(user_input):
                    if response.choices[0].delta.content is not None:
                        full_response += response.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            st.session_state['messages'].append({"role": "assistant", "content": full_response})

def edit_mongodb_document():
    st.subheader("Edit MongoDB Document")

    # Fetch and display data
    data = fetch_mongodb_data()
    if not data:
        st.info("No data available in MongoDB.")
        return

    # Display data in a table
    df = pd.DataFrame(data)
    st.dataframe(df)

    # Select a document to edit
    selected_index = st.selectbox("Select a document to edit", range(len(data)), format_func=lambda i: f"Document {i+1}")
    selected_doc = data[selected_index]

    # Display the selected document
    st.json(selected_doc)

    # User input for changes
    user_input = st.text_area("Describe the changes you want to make to this document:")

    if st.button("Apply Changes"):
        if user_input:
            try:
                # Generate modified JSON using LLM
                modified_json = generate_modified_json(selected_doc, user_input)
                
                # Display the modified JSON
                st.subheader("Modified JSON:")
                st.json(modified_json)

                # Confirm changes
                if st.button("Confirm and Update MongoDB"):
                    # Update document
                    update_result = update_mongodb_document(str(selected_doc['_id']), modified_json)
                    if update_result:
                        st.success("Document updated successfully!")
                        st.experimental_rerun()
                    else:
                        st.error("Failed to update the document in MongoDB.")
            except Exception as e:
                logger.error(f"Error in edit_mongodb_document: {str(e)}", exc_info=True)
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please describe the changes you want to make.")

def generate_modified_json(document, user_input):
    prompt = f"""
    Given the following MongoDB document:
    {json.dumps(document, indent=2)}

    And the user's request to modify it:
    {user_input}

    Generate the modified JSON document. Return only the modified JSON, without any explanation or additional text.
    Ensure that you're modifying the correct fields and using appropriate data types.
    If a field doesn't exist and needs to be added, add it to the appropriate place in the JSON structure.
    If a field needs to be removed, remove it entirely.
    Preserve any fields that don't need modification.
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that modifies JSON documents based on user instructions."},
                {"role": "user", "content": prompt}
            ]
        )

        modified_json = json.loads(response.choices[0].message.content.strip())
        return modified_json
    except Exception as e:
        logger.error(f"Error in generate_modified_json: {str(e)}", exc_info=True)
        raise

def delete_mongodb_document(document_id):
    client = get_mongodb_connection()
    if client:
        try:
            db = client.enterrag_db
            collection = db.pdf_data
            result = collection.delete_one({'_id': ObjectId(document_id)})
            return result.deleted_count
        except Exception as e:
            st.error(f"Error deleting from MongoDB: {str(e)}")
            return 0
        finally:
            client.close()
    else:
        return 0

def insert_mongodb_document(document):
    client = get_mongodb_connection()
    if client:
        try:
            db = client.enterrag_db
            collection = db.pdf_data
            if '_id' in document:
                del document['_id']  # Remove _id if present to let MongoDB generate a new one
            result = collection.insert_one(document)
            return result.inserted_id
        except Exception as e:
            st.error(f"Error inserting into MongoDB: {str(e)}")
            return None
        finally:
            client.close()
    else:
        return None

def generate_edit_command(document, user_input):
    prompt = f"""
    Given the following MongoDB document:
    {json.dumps(document, indent=2)}

    And the user's request to modify it:
    {user_input}

    Generate a command to make the requested changes to the document. It should be a MongoDB command to modify a certain .json record in context.
    The command should modify the 'document' variable directly.
    Do not include any explanations or comments, just the command to make the changes.
    Ensure that you're modifying the correct fields and using appropriate data types.
    """

    response = openai.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates Python commands to edit MongoDB documents."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()

def update_mongodb_document(document_id, updated_document):
    client = get_mongodb_connection()
    if client:
        try:
            db = client.enterrag_db
            collection = db.pdf_data
            
            # Remove the '_id' field from the update operation
            if '_id' in updated_document:
                del updated_document['_id']
            
            result = collection.replace_one({'_id': ObjectId(document_id)}, updated_document)
            logger.info(f"MongoDB update result: {result.modified_count} document(s) modified")
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating MongoDB: {str(e)}", exc_info=True)
            return False
        finally:
            client.close()
    else:
        logger.error("Failed to establish MongoDB connection")
        return False

def extract_text_from_pdf(file) -> str:
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_important_info(text: str) -> Dict:
    prompt = f"""
    Extract important information from the following text and organize it into a structured format:

    {text[:2000]}  # Limit to first 2000 characters to avoid token limits

    Provide the output as a JSON object with appropriate keys and values.
    """

    response = openai.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that extracts and structures information."},
            {"role": "user", "content": prompt}
        ]
    )

    content = response.choices[0].message.content

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # If JSON parsing fails, return a dictionary with the raw content
        return {"raw_content": content}

def insert_data_to_mongodb(data: Dict):
    client = get_mongodb_connection()
    if client:
        db = client.enterrag_db
        collection = db.pdf_data
        result = collection.insert_one(data)
        client.close()
        return result.inserted_id
    else:
        return None

def fetch_mongodb_data() -> List[Dict]:
    client = get_mongodb_connection()
    if client:
        db = client.enterrag_db
        collection = db.pdf_data
        data = list(collection.find({}, {'_id': 0}).limit(100))  # Limiting to 100 documents
        client.close()
        return data
    else:
        return []

def chat_with_mongodb(user_input: str) -> str:
    prompt = f"""
    Given the following user request:
    {user_input}

    Generate a response based on the data stored in the MongoDB collection 'pdf_data'.
    The data in this collection is structured JSON extracted from PDF files.
    Provide a concise and relevant answer based on the information available in the database.

    If the request involves retrieving specific information, mention that in your response.
    If the request is for analysis or insights, provide a thoughtful answer based on the available data.

    Provide only the response, without any additional explanation of the process.
    """

    response = openai.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides information based on MongoDB data."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()

def pdf_to_mongodb_page():
    st.header("PDF to MongoDB Converter")

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        if st.button("Process PDF and Store in MongoDB"):
            with st.spinner("Processing PDF and storing data..."):
                # Extract text from PDF
                pdf_text = extract_text_from_pdf(uploaded_file)

                # Extract important information
                important_info = extract_important_info(pdf_text)

                # Insert data into MongoDB
                inserted_id = insert_data_to_mongodb(important_info)
                if inserted_id:
                    st.success(f"Data inserted successfully! Document ID: {inserted_id}")
                else:
                    st.error("Failed to insert data into MongoDB.")

                # Fetch and display data
                data = fetch_mongodb_data()
                if data:
                    st.subheader("Sample Data from MongoDB")
                    st.dataframe(pd.DataFrame(data))
                else:
                    st.info("No data to display from MongoDB.")

# Main App
def main():
    # --- PAGE CONFIGURATION ---
    st.set_page_config(page_title="EnterRAG", page_icon="💼", layout="centered")

    # --- MAIN PAGE CONFIGURATION ---
    st.title("EnterRAG 💼")
    st.write("*Talk to your Enterprise Documents 📄, Receive Insights: AI. Unlocked.*")
    st.write(':blue[***Powered by LangChain***]')

    # ---- NAVIGATION MENU -----
    selected = option_menu(
        menu_title=None,
        options=["Info", "About"],
        icons=["bi bi-info-square", "bi bi-globe"],  # https://icons.getbootstrap.com
        orientation="horizontal",
    )

    # Custom CSS for buttons
    st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #0E1117;
        border: none;
        color: white;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 12px;
        transition-duration: 0.4s;
    }
    .stButton>button:hover {
        background-color: #FF4B4B;
        color : black
    }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar navigation with styled buttons
    with st.sidebar:
        st.sidebar.title("Navigation")
        if st.button("AI Chatbot"):
            st.session_state.page = "AI Chatbot"
        if st.button("Manage Collections"):
            st.session_state.page = "Manage Collections"
        if st.button("Convert to MongoDB"):
            st.session_state.page = "Convert to MongoDB"
        if st.button("Audit Data on MongoDB"):
            st.session_state.page = "Audit Data on MongoDB"

    if selected == "About":
        with st.expander("What technologies power this application?"):
            st.markdown(''' 
    This application is built using several powerful technologies:
    
    - **Streamlit**: A Python framework for building interactive web apps with ease. It's used to create the user interface for this application.
    - **OpenAI**: We use OpenAI's models for natural language processing and embeddings. OpenAI's GPT-4 handles the conversational AI capabilities.
    - **ChromaDB**: A vector database for efficient storage and retrieval of document embeddings. Chroma helps in managing and querying large sets of documents, making it easy to retrieve relevant information.
    - **PyPDF2**: A Python library to extract text from PDF documents, which are uploaded by the user to create collections.
    - **NumPy**: A core library for numerical computations, used to handle embeddings and similarity calculations efficiently.
    - **MongoDB**: A NoSQL database used to store structured data extracted from PDFs.

    These technologies together provide a seamless way to upload, manage, and query documents, enabling users to chat with their documents and get insightful answers based on the contents.
    ''')
        with st.expander("How does the AI interact with documents?"):
            st.markdown(''' 
    The AI interacts with your documents using a combination of **document embeddings** and **semantic search**:

    - **Document Embedding**: Each document you upload is processed to create an embedding (a numerical representation) of its content. This is done using OpenAI's embeddings model, which converts the textual content into a vector.
    - **ChromaDB**: After embedding the documents, the vectors are stored in **ChromaDB**, a database optimized for storing and querying these embeddings. When you ask a question, the AI searches the embeddings for the most relevant documents or document chunks.
    - **Semantic Search**: Using cosine similarity, the AI compares your query's embedding to the stored document embeddings to find the closest matches. This allows the AI to understand context and return highly relevant answers.
    - **GPT-4**: Once the AI identifies relevant documents, it uses **OpenAI's GPT-4** to generate detailed responses by providing the context from these documents.
    - **MongoDB**: Structured data extracted from PDFs is stored in MongoDB, allowing for efficient querying and retrieval of specific information.

    This combination of techniques allows the AI to provide accurate, context-aware responses, making it easier to extract meaningful insights from your documents.
    ''')
        with st.expander("What are the core features of this application?"):
            st.markdown(''' 
    The application offers a range of features designed to help you interact with and manage your documents:

    - **Manage Collections**: You can create, modify, and delete collections of documents. Collections are groups of PDF files that you upload, which are then indexed for searching and querying.
    - **AI Chatbot Interface**: Once you have a collection, you can chat with your documents using the AI chatbot. The chatbot uses the uploaded documents to answer questions, extract insights, and assist with tasks like summarization, analysis, and more.
    - **Document Upload & Chunking**: The app allows you to upload multiple PDF documents, which are automatically processed, chunked into smaller pieces, and stored for efficient querying.
    - **Semantic Search**: By leveraging **OpenAI embeddings** and **ChromaDB**, the AI can search your documents semantically, not just through keyword matching, giving you context-aware responses.
    - **Real-Time Chatting**: Interact with the AI in real-time as it streams responses, allowing for a more engaging and conversational experience.
    - **MongoDB Integration**: Extract structured data from PDFs and store it in MongoDB for efficient retrieval and querying.

    These features make it simple to manage large volumes of documents and extract meaningful insights with minimal effort. Whether for research, business analysis, or personal use, this tool helps you maximize the value of your documents.
    ''')

    # Initialize session state for page if not exists
    if 'page' not in st.session_state:
        st.session_state.page = "AI Chatbot"

    if st.session_state.page == "AI Chatbot":
        if selected == "Info":
            with st.expander("What is this page?"):
                st.markdown('''
    The **AI Chatbot** page allows you to interact with your uploaded documents and ask questions about their contents. 

    Here's how it works:

    - Once you have uploaded your documents and created a collection in the **"Manage Collections"** page, you can use the AI Chatbot to **ask questions** related to those documents.
    - The chatbot uses **Embedding-based Retrieval-Augmented Generation (RAG)** techniques, where it searches for relevant content from the documents you uploaded and generates accurate answers using GPT-based models (like GPT-4).
    - The AI will try to find **relevant sections or chunks of text** from your documents that answer your query. It then combines these with its own knowledge to create a useful response.

    Whether you're looking for specific information, insights, or performing document-based queries, the AI Chatbot helps you quickly extract answers and knowledge from your documents.
    ''')
            with st.expander("What are the pre-requisites to chat?"):
                st.markdown('''
    In order to use the AI Chatbot, you must first **create a collection** in the **"Manage Collections"** page. Here's how you can do it:

    1. **Go to the "Manage Collections" page** from the Sidebar.
    2. **Click on "Add Collection"**, where you can upload your documents (PDF files).
    3. **Submit** the files, and they will be processed to create a collection.
    4. Once your collection is created, come back to the **"AI Chatbot"** page, select your collection from the dropdown, and you're ready to start chatting with your documents!

    You can interact with the chatbot, ask questions, and retrieve insights based on the content of your uploaded documents.

    **Important Notes:**
    - Ensure that your documents are in PDF format for easy upload.
    - The more diverse and rich your collection is, the better the AI will be at answering your questions.
    ''')
            with st.expander("Try a Demo Run Now"):
                st.markdown('''
    Here's how you can run a Demo of the application:

    1. **Download any one of the collections** from below: Google, Meta, or Netflix.
    2. **Unzip the collection**, revealing the appropriate PDF files.
    3. **Upload the files** in the "Add Collection" section of the **"Manage Collections"** page.
    4. Come back to the **"AI Chatbot"** page, select your collection, and **start chatting** with your documents!

    This demo allows you to quickly see how the AI can interact with your documents and provide insights. Just follow the simple steps and you're ready to start!

    Enjoy exploring the demo and interact with the content!
    ''')
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button('Google'):
                        st.write(f"[Download file]({"https://www.dropbox.com/scl/fi/cx2t9w5t4uapv9mkwb1uc/Google.zip?rlkey=mldpxflx01kpbi82b3j4y99zx&st=0ok4jawz&dl=1"})")
                with col2:
                    if st.button('Meta'):
                        st.write(f"[Download file]({"https://www.dropbox.com/scl/fi/h4145xifvhf39t5ojdx2i/Meta.zip?rlkey=v80pwcxm129b7sunnrzk5cula&st=oemj80a0&dl=1"})")
                with col3:
                    if st.button('Netflix'):
                        st.write(f"[Download file]({"https://www.dropbox.com/scl/fi/savy5gl5e5kc4ar0rxd9g/Netflix.zip?rlkey=ugotzhrucw6uflgz0leli0g9o&st=nk3p3ze5&dl=1"})")
                
        chatbot_interface()

    elif st.session_state.page == "Manage Collections":
        if selected == "Info":
            with st.expander("What is this page?"):
                st.markdown('''
    This page allows you to manage your document collections. 
    You can perform various actions such as:

    - **Add new collections**: Upload documents and create collections of your enterprise data.
    - **Modify collections**: Add new files to existing collections or remove files from them.
    - **Delete collections**: Remove entire collections from the system when they are no longer needed.

    Use this page to organize and manage your documents before interacting with the AI Chatbot for insights.
    ''')
            with st.expander("What is a collection?"):
                st.markdown('''
    A **collection** is a group of documents that you can upload and organize in the system. It acts as a container for your data and enables you to interact with the documents through the AI chatbot.

    In this app, we use **ChromaDB** to store and manage collections. Here's a brief overview of how collections work:

    1. **Embedding documents**: 
       - When you upload a document, we extract its text and break it down into smaller chunks (called *chunks*).
       - Each chunk is then converted into an **embedding** using OpenAI's embeddings model. An embedding is a numerical representation of the content in that chunk, which allows us to compare and retrieve relevant information efficiently.
    
    2. **ChromaDB**:
       - ChromaDB is a vector database that stores these embeddings and allows for **similarity search**. When you ask a question, the AI looks for the most relevant document chunks by comparing the embeddings of the question and the document chunks.
       - This process ensures that the AI provides you with the most contextually relevant information when responding to your queries.

    3. **Retrieving Context**:
       - When you interact with the AI chatbot, it uses these embeddings to find the most similar chunks of information from the collection. This is known as **retrieval-augmented generation** (RAG). The AI then combines the context from these chunks to generate accurate and insightful responses.
    
    By organizing your documents into collections, you make it easier for the AI to fetch relevant information and assist you in extracting insights from your data.
    ''')

        manage_collections()

    elif st.session_state.page == "Convert to MongoDB":
        if selected == "Info":
            with st.expander("What is this page?"):
                st.markdown('''
    This page allows you to convert your PDF documents to structured data stored in MongoDB. 
    Here's what you can do:

    - Upload a PDF file
    - Extract important information from the PDF using AI
    - Store the extracted structured data in MongoDB
    - View the stored data

    This feature enables you to convert unstructured PDF data into a structured format stored in a NoSQL database, 
    which can be useful for further analysis or integration with other data systems.
    ''')
        pdf_to_mongodb_page()
    
    elif st.session_state.page == "Audit Data on MongoDB":
        if selected == "Info":
            with st.expander("What is this page?"):
                st.markdown('''
This page allows you to view and edit the audit data stored in MongoDB. Here's what you can do:

1. **View Audit Data**: See a table of all the audit data extracted from your PDF documents and stored in MongoDB.

2. **Select and Edit Documents**: Choose a specific document from the table to view its details and make changes.

3. **AI-Assisted Editing**: Describe the changes you want to make in natural language, and our AI will generate the appropriate edit commands.

4. **Apply Changes**: Review the AI-generated edit commands and apply them directly to update the MongoDB document.

This feature enables you to:
- Easily review the structured data extracted from your PDFs
- Make precise adjustments to the stored information
- Correct any errors or update outdated information
- Maintain the accuracy and relevance of your audit data over time

By combining the power of MongoDB for data storage with AI-assisted editing, we provide a user-friendly interface for managing complex document structures without requiring deep technical knowledge of databases or JSON manipulation.

Remember to review any AI-generated changes carefully before applying them to ensure they match your intended modifications.
        ''')
        edit_mongodb_document()

if __name__ == "__main__":
    main()