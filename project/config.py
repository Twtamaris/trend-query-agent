import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables from .env file at the very beginning
load_dotenv()

# BigQuery Configuration
PROJECT_ID = "usps-geo-dashboard"  # Replace with your actual project ID
DATASET_ID = "heatmap_data"  # Replace with your actual dataset ID
CONVERSATIONS_TABLE = "all_tags"  # Your table with raw data

# FAISS Index and Metadata Paths
# Ensure these directories exist or are created by your script
FAISS_INDEX_DIR = "faiss_index_data"
FAISS_INDEX_PATH = os.path.join(FAISS_INDEX_DIR, "conversations.faiss")
METADATA_PATH = os.path.join(FAISS_INDEX_DIR, "conversations_metadata.json")

# Default date range for BigQuery if no specific dates are provided
# Queries data from the last 2 years, adjust as needed based on your data volume and retention.
DEFAULT_BQ_START_DATE = (datetime.now() - timedelta(days=8)).strftime('%Y-%m-%d')
DEFAULT_BQ_END_DATE = datetime.now().strftime('%Y-%m-%d')

# GraphRAG (Knowledge Graph) Configuration
BRAND_DOCUMENTS_DIR = "data/documents" # Directory for your brand knowledge documents
GRAPHRAG_OUTPUT_DIR = "graphrag_output" # Directory for GraphRAG related files/logs (if any)

# --- Neo4j Configuration ---
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Check and warn if Neo4j credentials are missing
if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
    print("WARNING: Neo4j environment variables not fully set. Knowledge Graph functionality may be impaired.")
    print("Please set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD in your .env file.")

# --- API Keys ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY not found in environment variables. Groq LLM functionality may be impaired.")
    print("Please set the GROQ_API_KEY environment variable in your .env file.")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY not found in environment variables. GraphRAG LLM functionality may be impaired.")
    print("Please set the OPENAI_API_KEY environment variable in your .env file.")

# --- LLM Model Names ---
GROQ_LLM_MODEL_NAME = "llama3-8b-8192" # Or "mixtral-8x7b-32768" for Groq
OPENAI_LLM_MODEL_NAME = "gpt-4o" # Or "gpt-4-turbo" for OpenAI
LLM_TEMPERATURE = 0.0 # Can be adjusted per model if needed, or keep generic