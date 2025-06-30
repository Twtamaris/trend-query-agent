import os
from google.cloud import bigquery
import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, PodSpec, ServerlessSpec # Import Pinecone and Index classes, and spec types
from dotenv import load_dotenv
from tqdm.auto import tqdm # For progress bar during upsert
from datetime import datetime # Import datetime for timestamp handling

# Load environment variables from .env file
load_dotenv()

# --- Initialize BigQuery client ---
client = bigquery.Client()

# --- Initialize Pinecone ---
try:
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    # The 'environment' parameter for Pinecone client is often deprecated or not needed
    # for newer API versions. The region is now specified within the index spec.

    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY not set in .env file.")

    # Modern Pinecone client initialization typically only needs the API key.
    # The environment/region details are handled by the index spec.
    pc = Pinecone(api_key=pinecone_api_key)
    print("Pinecone initialized.")

except ValueError as e:
    print(f"Error initializing Pinecone: {e}")
    print("Please ensure PINECONE_API_KEY is set in your .env file.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during Pinecone initialization: {e}")
    exit()

# --- Load a pre-trained embedding model ---
print("Loading Sentence Transformer model 'all-MiniLM-L6-v2'...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")

# --- Pinecone Index Configuration ---
INDEX_NAME = "customer-conversations-index" # Must match the index name you created in Pinecone
EMBEDDING_DIMENSION = 384 # Dimension for 'all-MiniLM-L6-v2'
METRIC_TYPE = "cosine" # Must match the metric type of your Pinecone index

# Define the index specification using ServerlessSpec or PodSpec
# For most users, ServerlessSpec is recommended for its ease of use and scalability.
# Choose a cloud and region supported by Pinecone serverless.
# Common choices: "aws", "us-west-2" | "gcp", "us-central1" | "azure", "eastus"
# IMPORTANT: Verify available regions in your Pinecone console.
INDEX_SPEC = ServerlessSpec(cloud="aws", region="us-west-2") 
# If you were using a pod-based index (e.g., free starter tier before serverless was default), it might look like this:
# INDEX_SPEC = PodSpec(environment="gcp-starter") 
# However, the error indicates a preference for 'serverless' or 'pod' keys directly.

# Ensure the index exists and get a handle to it
try:
    if INDEX_NAME not in pc.list_indexes():
        print(f"Creating Pinecone index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric=METRIC_TYPE,
            spec=INDEX_SPEC # Use the defined ServerlessSpec or PodSpec
        )
        print(f"Index '{INDEX_NAME}' created. Waiting for it to be ready...")
        # Optional: Add a small delay for index creation to complete
        import time
        time.sleep(10) # Give it more time to initialize, especially for new indexes
    else:
        print(f"Index '{INDEX_NAME}' already exists.")
    
    index = pc.Index(INDEX_NAME)
    print(f"Connected to Pinecone index: '{INDEX_NAME}'")
    # Check index description to verify configuration
    print(index.describe_index_stats())

except Exception as e:
    print(f"Error connecting to or creating Pinecone index: {e}")
    print("Please ensure your Pinecone API key is valid and the specified region/cloud is correct and available for your account type.")
    exit()


def generate_and_store_embeddings_in_pinecone(project_id, dataset_id, source_table_id):
    """
    Fetches conversation texts from a BigQuery table, generates embeddings,
    and stores them in the configured Pinecone index.

    Args:
        project_id (str): Your Google Cloud Project ID.
        dataset_id (str): The BigQuery dataset ID (e.g., 'customer_service_data').
        source_table_id (str): The ID of the table containing conversation text (e.g., 'conversations').
    """
    source_table_ref = f"`{project_id}.{dataset_id}.{source_table_id}`"

    # 1. Fetch conversation data from BigQuery
    print(f"Fetching data from {source_table_ref} for embedding generation...")
    # Select all relevant columns, especially conversation_id for Pinecone ID
    query = f"""
    SELECT
        conversation_id,
        conversation_text,
        sentiment,
        issue_category,
        timestamp,
        zip_code
    FROM {source_table_ref}
    ORDER BY conversation_id
    """
    query_job = client.query(query)
    rows = list(query_job.result())

    if not rows:
        print(f"No data found in {source_table_ref}. Exiting.")
        return

    df = pd.DataFrame([dict(row) for row in rows])
    print(f"Fetched {len(df)} rows from {source_table_id}.")

    if 'conversation_text' not in df.columns:
        print("Error: 'conversation_text' column not found in the source table. Please check your schema.")
        return
    
    # Fill NaN/None values in conversation_text with empty string before encoding
    df['conversation_text'] = df['conversation_text'].fillna('')
    texts_to_encode = df['conversation_text'].tolist()

    # 2. Generate Embeddings
    print(f"Generating embeddings for {len(texts_to_encode)} conversations...")
    embeddings = model.encode(texts_to_encode, show_progress_bar=True, normalize_embeddings=True)
    print("Embeddings generated.")

    # 3. Prepare data for Pinecone upsert
    # Pinecone expects data as (id, vector, metadata) tuples
    # 'id' should be a unique string (conversation_id)
    # 'vector' is the list of floats (embedding)
    # 'metadata' is a dictionary of other data you want to filter/retrieve (sentiment, zip_code, etc.)
    
    upsert_data = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Preparing data for Pinecone"):
        conv_id = str(row['conversation_id']) # Ensure ID is a string
        embedding_vector = embeddings[i].tolist() # Convert numpy array to list
        
        # Prepare metadata dictionary
        metadata = {
            "conversation_text": row['conversation_text'], # Store original text for retrieval
            "sentiment": row['sentiment'],
            "issue_category": row['issue_category'],
            "zip_code": row['zip_code'],
            "timestamp": row['timestamp'].isoformat() if isinstance(row['timestamp'], datetime) else str(row['timestamp'])
            # Convert timestamp to ISO format string for consistency in metadata
        }
        # Filter out None values from metadata
        metadata = {k: v for k, v in metadata.items() if v is not None}
        
        upsert_data.append((conv_id, embedding_vector, metadata))
    
    # 4. Upsert embeddings to Pinecone
    print(f"Upserting {len(upsert_data)} vectors to Pinecone index '{INDEX_NAME}'...")
    # It's good practice to upsert in batches
    BATCH_SIZE = 100 # Adjust based on your data size and Pinecone limits
    for i in tqdm(range(0, len(upsert_data), BATCH_SIZE), desc="Upserting batches"):
        batch = upsert_data[i:i + BATCH_SIZE]
        try:
            index.upsert(vectors=batch)
        except Exception as e:
            print(f"Error upserting batch {i} to {i + BATCH_SIZE}: {e}")
            # You might want more sophisticated error handling or retry logic here.

    print(f"Successfully upserted data to Pinecone index: '{INDEX_NAME}'.")
    print(f"Current index stats: {index.describe_index_stats()}")


# --- Run this once to generate embeddings and store them in Pinecone ---
if __name__ == "__main__":
    # Replace with your actual project, dataset, and table IDs
    project_id = 'chatbot-project-464108' # Your actual Google Cloud Project ID
    dataset_id = 'customer_service_data'
    source_table_id = 'conversations' # Your original table name in BigQuery

    # This will fetch data, generate embeddings, and store them in Pinecone
    generate_and_store_embeddings_in_pinecone(project_id, dataset_id, source_table_id)

    print("\n--- Step 4 (Embeddings) Completed ---")
    print(f"Embeddings for your conversations are now stored in the Pinecone index '{INDEX_NAME}'.")
    print("\n--- Next Major Step (Step 5) ---")
    print("Next, you will modify your chatbot's `get_conversation_data` function or create a new one.")
    print("This function will now:")
    print("1. Take the user's query.")
    print("2. Generate an embedding for the user's query.")
    print("3. Query the Pinecone index with this embedding to find similar conversation IDs.")
    print("4. Use these conversation IDs to fetch the full conversation data (text, sentiment, etc.) from BigQuery.")
    print("This allows you to filter first by metadata in BigQuery, and then use semantic search on the resulting subset (if applicable) or query Pinecone directly for similarity based on the user's query.")