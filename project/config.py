import os
from datetime import datetime, timedelta

# BigQuery Configuration
PROJECT_ID = "chatbot-project-464108"  # Replace with your actual project ID
DATASET_ID = "customer_service_data"  # Replace with your actual dataset ID
CONVERSATIONS_TABLE = "conversations"  # Your table with raw data

# FAISS Configuration
FAISS_INDEX_PATH = "local_faiss_index.faiss"
METADATA_PATH = "local_faiss_metadata.json"

# Default date range for BigQuery if no specific dates are provided
# Queries data from the last 2 years, adjust as needed based on your data volume and retention.
DEFAULT_BQ_START_DATE = (datetime.now() - timedelta(days=2 * 365)).strftime('%Y-%m-%d')
DEFAULT_BQ_END_DATE = datetime.now().strftime('%Y-%m-%d')

# GROQ LLM Configuration (loaded via .env for API key)
GROQ_MODEL_NAME = "llama3-8b-8192"
LLM_TEMPERATURE = 0.0

# GraphRAG (Simulated) Configuration
BRAND_DOCUMENTS_DIR = "data/brand_documents"