import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import io
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
from pandas import json_normalize
import json2table
from flatten_json import flatten

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
OPENAI_API_KEY = "opwnai key"
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
                st.write(modified_json)

                # Confirm changes
                if st.button("Confirm and Update MongoDB"):
                    # Remove the triple backticks from the modified JSON
                    modified_json = modified_json.replace("```json", "").strip("'''").strip('"""')
                    if modified_json.endswith('```'):
                        modified_json = modified_json[:-3]

                    # Update document
                    update_result = update_mongodb_document(str(selected_doc['_id']), json.loads(modified_json))
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
            
            result = collection.update_one(
                {'_id': ObjectId(document_id)},
                {'$set': updated_document}
            )
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

def db_image_page():
    st.header("MongoDB Document Viewer")

    # Fetch data from MongoDB
    data = fetch_mongodb_data()

    if not data:
        st.info("No data available in MongoDB.")
        return

    # Create a list of document IDs
    document_ids = [f"Document {i + 1}" for i in range(len(data))]

    # Create a document selector
    selected_document = st.selectbox("Select a document to view:", document_ids)

    # Get the index of the selected document
    selected_index = document_ids.index(selected_document)

    # Display the selected document
    st.subheader(selected_document)

    # Check if the 'raw_content' key exists
    if 'raw_content' in data[selected_index]:
        # Get the JSON string
        json_string = data[selected_index]['raw_content'].replace("```json", "").strip("'''").strip('"""')

        # Remove the triple backticks at the end of the JSON string
        if json_string.endswith('```'):
            json_string = json_string[:-3]

        # Try to parse the JSON string
        try:
            json_data = json.loads(json_string)
            st.success("JSON string parsed successfully:")
            
            # Convert the JSON data to a table
            table = json2table.convert(json_data)

            # Display the table in Streamlit
            st.write(table, unsafe_allow_html=True)

        except json.JSONDecodeError as e:
            # If the JSON string is not valid, display an error message
            st.error(f"Invalid JSON string: {e}")
            st.write("JSON string that failed to parse:")
            st.code(json_string)  # Display the JSON string in a code block for better visibility

    else:
        # If the 'raw_content' key does not exist, display the document as is
        st.json(data[selected_index])

    # Add a horizontal line to separate documents
    st.markdown("---")

# Business Thingy
def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def parse_financial_data(text):
    prompt = f"""
    Extract the following financial information from the given text:
    - Total revenue (in dollars)
    - Revenue growth (percentage)
    - Operating profit (in dollars)
    - Operating margin (percentage)
    - Net income (in dollars)
    - Earnings per share (EPS) (in dollars)
    - Operating cash flow (in dollars)
    - Revenue breakdown: Provide a detailed breakdown of revenue by all available categories, segments, or product lines. Include ALL subcategories mentioned in the report.

    Text: {text}

    Provide the output as a JSON object with appropriate keys and values. 
    If you can't find a specific piece of information, use null for its value.

    IMPORTANT: 
    1. For all monetary values (revenue, profit, income, cash flow):
       - If the report states the values are in millions or billions, convert them to the full number.
       - For example, if it says "$70.01 billion", the value should be 70010000000.
       - If it says "$80.46 million", the value should be 80460000.
    2. Ensure all monetary values are in whole numbers (no decimals).
    3. Percentages should be in decimal form (e.g., 15% should be 15, not 0.15).
    4. For the revenue breakdown:
       - Include ALL categories and subcategories mentioned in the report.
       - This might include terms like "Family of Apps", "Reality Labs", "Advertising", "Other revenue", etc.
       - Provide this as a nested dictionary if there are subcategories.
       - Use the same scale as the total revenue.
    5. The EPS should be in dollars and cents (e.g., 4.88 for $4.88 per share).
    6. If operating profit is not explicitly stated, it may be referred to as "Income from operations" or similar terms.

    Be extremely thorough in extracting all revenue categories and subcategories. Do not omit any breakdown information provided in the report.

    Reply with the JSON object only. Don't give any explanations or comments. Just provide the structured JSON data.
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": "You are a highly skilled financial analyst AI that extracts and structures financial data accurately and comprehensively."},
                {"role": "user", "content": prompt}
            ]
        )

        content = response.choices[0].message.content

        # Remove markdown formatting if present
        content = content.replace("```json", "").replace("```", "").strip()

        # Parse the JSON
        data = json.loads(content)

        # Ensure all required keys are present
        required_keys = [
            'total_revenue', 'revenue_growth', 'operating_profit', 'operating_margin',
            'net_income', 'earnings_per_share', 'operating_cash_flow', 'revenue_breakdown'
        ]
        for key in required_keys:
            if key not in data:
                data[key] = None

        return data

    except json.JSONDecodeError as e:
        st.error(f"Failed to parse the AI response. JSON error: {str(e)}")
        st.text("Raw AI response:")
        st.code(content)
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")

    return None

def format_large_number(num):
    if num is None:
        return "N/A"
    if abs(num) >= 1e9:
        return f"${num/1e9:.2f}B"
    elif abs(num) >= 1e6:
        return f"${num/1e6:.2f}M"
    elif abs(num) >= 1e3:
        return f"${num/1e3:.2f}K"
    else:
        return f"${num:,.2f}"

def business_metrics_dashboard():
    st.title("Strategic Financial Intelligence Hub")

    uploaded_file = st.file_uploader("Upload your quarterly financial report (PDF)", type=["pdf"])
    
    if uploaded_file is not None:
        with st.spinner(" Processing the PDF..."):
            text = extract_text_from_pdf(uploaded_file)
            financial_data = parse_financial_data(text)
        
        if financial_data:
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Total Revenue",
                    value=format_large_number(financial_data['total_revenue']),
                    delta=f"{financial_data['revenue_growth']}% YoY" if financial_data['revenue_growth'] is not None else None
                )
    
            with col2:
                st.metric(
                    label="Operating Profit",
                    value=format_large_number(financial_data['operating_profit']),
                    delta=f"Margin: {financial_data['operating_margin']}%" if financial_data['operating_margin'] is not None else None
                )
    
            with col3:
                st.metric(
                    label="Net Income",
                    value=format_large_number(financial_data['net_income']),
                    delta=f"EPS: ${financial_data['earnings_per_share']:.2f}" if financial_data['earnings_per_share'] is not None else "No EPS available."
                )
    
            with col4: st.metric(
                    label="Operating Cash Flow",
                    value=format_large_number(financial_data['operating_cash_flow']),
                    delta=f"{financial_data['operating_cash_flow']/financial_data['total_revenue']*100:.1f}% of Rev" if financial_data['operating_cash_flow'] is not None and financial_data['total_revenue'] is not None else None
                )

            # Create pie chart for revenue breakdown
            st.subheader("Revenue Breakdown by Segments")
            if financial_data['revenue_breakdown']:
                flattened_breakdown = flatten_dict(financial_data['revenue_breakdown'])
                fig = px.pie(
                    values=list(flattened_breakdown.values()),
                    names=list(flattened_breakdown.keys()),
                    title='Revenue by Segment'
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No revenue breakdown data available.")

            st.subheader("Earnings Per Share (EPS)")
            if financial_data['earnings_per_share'] is not None:
                fig_eps = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = financial_data['earnings_per_share'],
                    number = {'prefix': "$"},
                    title = {"text": "Earnings Per Share"}
                ))
                st.plotly_chart(fig_eps, use_container_width=True)
            else:
                st.info("No EPS available.")

            st.subheader("Key Financial Metrics")
            if financial_data['total_revenue'] is not None and financial_data['operating_profit'] is not None and financial_data['net_income'] is not None:
                fig_metrics = go.Figure(data=[
                    go.Bar(name='Total Revenue', x=['Total'], y=[financial_data['total_revenue']]),
                    go.Bar(name='Operating Profit', x=['Total'], y=[financial_data['operating_profit']]),
                    go.Bar(name='Net Income', x=['Total'], y=[financial_data['net_income']])
                ])
                fig_metrics.update_layout(barmode='group', title='Key Financial Metrics')
                st.plotly_chart(fig_metrics, use_container_width=True)
            else:
                st.info("No key financial metrics available.")

        else:
            st.error("Failed to extract financial data from the uploaded report.")

    else:
        st.info("Please upload a quarterly financial report (PDF) to view the dashboard.")

def create_dataframe(data):
    yearly_data = data.get('yearly_data', {})
    df = pd.DataFrame(yearly_data)
    df['Date'] = df.index
    # Ensure column names are consistent
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    return df

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
        if st.button("Store PDF to MongoDB"):
            st.session_state.page = "Store PDF to MongoDB"
        if st.button("Audit Data on MongoDB"):
            st.session_state.page = "Audit Data on MongoDB"
        if st.button("View MongoDB Documents"):
            st.session_state.page = "View MongoDB Documents"
        if st.button("Strategic Financial Intelligence Hub"):
            st.session_state.page = "Strategic Financial Intelligence Hub"

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

    elif st.session_state.page == "Store PDF to MongoDB":
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

    elif st.session_state.page == "View MongoDB Documents":
        if selected == "Info":
            with st.expander("What is this page?"):
                st.markdown('''
This page displays the documents stored in MongoDB as tables. 
Each document is presented in a tabular format for easy viewing, 
with an option to see the raw JSON data. 
Use this page to quickly review and verify the structured data 
extracted from your PDF documents.
''')
        db_image_page()
    
    elif st.session_state.page == "Strategic Financial Intelligence Hub":
        if selected == "Info":
            with st.expander("What is this page?"):
                st.markdown('''
    This page provides a comprehensive Strategic Financial Intelligence Hub. Here's what you can do:

    1. **Upload Financial Reports**: You can upload PDF files containing financial reports.
    2. **View Key Metrics**: The dashboard displays four key financial metrics:
       - Total Revenue with year-over-year growth
       - Operating Profit with Operating Margin
       - Net Income with Earnings Per Share (EPS)
       - Key Business Segment Revenue with growth percentage
    3. **Interactive Graphs**: The page generates interactive graphs based on the uploaded data, including:
       - Revenue Trends
       - Profit Margins
       - Segment Performance
       - Earnings Per Share (EPS) Trend
    4. **AI-Powered Analysis**: The dashboard uses OpenAI's API to extract and structure financial data from the uploaded PDFs.

    This dashboard helps you quickly visualize and understand key financial metrics and trends from your company's reports.
    ''')
        business_metrics_dashboard()

if __name__ == "__main__":
    main()
