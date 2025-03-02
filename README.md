# EnterRAG

**Talk to your Enterprise Documents 📄, Receive Insights: AI. Unlocked.**

*Powered by LangChain*

## Overview

EnterRAG is a comprehensive enterprise document analysis and financial intelligence platform built with Streamlit. It combines the power of vector databases (ChromaDB), large language models (OpenAI), and NoSQL databases (MongoDB) to provide intelligent document management, chatbot functionality, and financial analysis capabilities.

## Features

### AI Chatbot
- Upload, manage, and chat with PDF documents using AI
- Semantic search through document collections
- Real-time streaming responses from GPT-4
- RAG (Retrieval Augmented Generation) implementation for context-aware answers

### Document Management
- Create, modify, and delete collections of PDF documents
- Automated text extraction, chunking, and embedding
- Document storage and retrieval using ChromaDB

### MongoDB Integration
- Convert PDFs to structured data in MongoDB
- View, edit, and audit MongoDB documents
- AI-assisted document editing

### Strategic Financial Intelligence Hub
- Upload financial reports for automated analysis
- Extract key financial metrics (revenue, profit, EPS, etc.)
- Interactive visualizations of financial data
- Breakdown of business segments and performance

## Tech Stack

- **Streamlit**: Web application framework
- **OpenAI**: GPT models and embeddings
- **ChromaDB**: Vector database for document embeddings
- **MongoDB**: NoSQL database for structured data
- **Plotly**: Interactive data visualization
- **PyPDF2**: PDF text extraction
- **Pandas**: Data manipulation and analysis

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/enterrag.git
   cd enterrag
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   - Create a `.env` file with your API keys and MongoDB connection string:
   ```
   OPENAI_API_KEY=your_openai_api_key
   MONGODB_URI=your_mongodb_connection_string
   ```

4. Run the application:
   ```
   streamlit run enterag.py
   ```

## Usage Guide

### Getting Started

1. Navigate to the application in your web browser (default: http://localhost:8501)
2. Use the sidebar to navigate between different features
3. Start by creating a document collection in the "Manage Collections" page

### Document Collections

1. Go to "Manage Collections" from the sidebar
2. Select "Add Collection" and provide a collection name
3. Upload PDF files to add to your collection
4. Use "Modify Collection" to add or remove files from existing collections

### AI Chatbot

1. Go to "AI Chatbot" from the sidebar
2. Select a collection from the dropdown menu
3. Type your questions in the chat input
4. The AI will respond with relevant information from your documents

### Financial Analysis

1. Go to "Strategic Financial Intelligence Hub" from the sidebar
2. Upload a financial report PDF (quarterly or annual report)
3. View the automatically generated financial metrics and visualizations
4. Analyze revenue breakdowns, profit margins, and other key metrics

### MongoDB Integration

1. Use "Store PDF to MongoDB" to extract and store structured data
2. View the stored documents in "View MongoDB Documents"
3. Edit and audit data using "Audit Data on MongoDB"

## Demo Collections

The application provides demo collections from Google, Meta, and Netflix. To use these:

1. Download the demo collection from the provided links in the "AI Chatbot" page
2. Unzip the collection to access the PDF files
3. Upload the files in the "Manage Collections" page
4. Return to the "AI Chatbot" page to start chatting with the documents

## Requirements

- Python 3.8+
- OpenAI API key
- MongoDB Atlas account (or local MongoDB instance)
- Internet connection for API calls

## Dependencies

- streamlit
- plotly
- openai
- chromadb
- pandas
- numpy
- pymongo
- PyPDF2
- streamlit_option_menu
- sklearn
