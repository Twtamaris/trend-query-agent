

# faiss_utils.py

import os
import json
import numpy as np
import faiss
from datetime import datetime, date # Ensure datetime and date are imported
from bq_utils import get_conversation_data
from models import EMBEDDING_MODEL
from config import FAISS_INDEX_PATH, METADATA_PATH, FAISS_INDEX_DIR

# --- Custom JSON Encoder for robustness ---
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat() # Convert datetime/date objects to ISO 8601 strings
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item() # Convert numpy integers/floats to standard Python int/float
        if isinstance(obj, np.ndarray):
            return obj.tolist() # Convert numpy arrays to Python lists
        # Handle decimal.Decimal if it somehow appears (e.g., from BIGNUMERIC in BQ)
        if isinstance(obj, float) and obj != obj: # Check for NaN (Not a Number) floats
            return None # JSON doesn't support NaN, serialize as null
        if isinstance(obj, (float, int)) and (obj == float('inf') or obj == float('-inf')): # Check for inf
            return None # JSON doesn't support inf, serialize as null

        return super().default(obj)

def load_or_build_faiss_index():
    if not os.path.exists(FAISS_INDEX_DIR):
        os.makedirs(FAISS_INDEX_DIR)
        print(f"Created FAISS index directory: '{FAISS_INDEX_DIR}'.")

    faiss_index = None
    all_conversations_indexed = [] # This will hold data read from metadata.json

    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
        try:
            print(f"Attempting to load FAISS index from {FAISS_INDEX_PATH} and metadata from {METADATA_PATH}...")
            faiss_index = faiss.read_index(FAISS_INDEX_PATH)
            with open(METADATA_PATH, 'r') as f:
                all_conversations_indexed = json.load(f) # No custom decoder needed for loading simple types
            print("FAISS index and metadata loaded successfully.")
        except Exception as e:
            print(f"Error loading existing FAISS index or metadata: {e}. Rebuilding...")
            faiss_index = None
            all_conversations_indexed = [] # Reset to rebuild

    print("Fetching all conversations from BigQuery for indexing check...")
    bq_all_conversations = get_conversation_data()

    # Create a map for quick lookup of existing conversations from BQ by ID
    bq_conversations_map = {conv['conversation_id']: conv for conv in bq_all_conversations if conv.get('conversation_id')}

    new_conversations_to_embed = []
    updated_conversations_for_metadata = [] # This list will be the source for the new metadata.json
    
    # Process existing indexed conversations (from file)
    for conv_data in all_conversations_indexed:
        conv_id = conv_data.get('conversation_id')
        if conv_id and conv_id in bq_conversations_map:
            # If conversation exists in both, use the BQ version for updated data
            # but preserve the embedding from the loaded file if it exists
            bq_version = bq_conversations_map[conv_id]
            if 'embedding' in conv_data and conv_data['embedding'] is not None:
                bq_version['embedding'] = conv_data['embedding']
            updated_conversations_for_metadata.append(bq_version)
            del bq_conversations_map[conv_id] # Mark as processed
        # If conversation was in metadata but not in current BQ pull, we might omit it
        # or include it if you want to retain old data. For now, we omit.

    # Add remaining new conversations from BQ (those not yet processed)
    for conv_id in bq_conversations_map:
        conv_data = bq_conversations_map[conv_id]
        if conv_data.get('conversation_text'): # Only process if content exists
            # Embed new conversations
            if 'embedding' not in conv_data or conv_data['embedding'] is None:
                print(f"Embedding new conversation: {conv_id}")
                try:
                    # Ensure the text is a string before embedding
                    text_to_embed = str(conv_data['conversation_text'])
                    embedding = EMBEDDING_MODEL.encode([text_to_embed], normalize_embeddings=True)[0].tolist()
                    conv_data['embedding'] = embedding
                except Exception as e:
                    print(f"Error embedding conversation {conv_id}: {e}")
                    conv_data['embedding'] = None # Mark as None if embedding fails
            
            # Only add to list if embedding was successful
            if conv_data['embedding'] is not None:
                new_conversations_to_embed.append(conv_data)
                updated_conversations_for_metadata.append(conv_data)

    # Filter out any conversations that might have failed to embed or lack text
    final_conversations_for_indexing = [
        conv for conv in updated_conversations_for_metadata
        if conv.get('embedding') is not None and conv.get('conversation_text') is not None
    ]

    if not final_conversations_for_indexing:
        print("No valid conversations with embeddings to build a FAISS index. Returning empty index.")
        return None, []

    embeddings_for_faiss = np.array([
        np.array(conv['embedding']).astype('float32') for conv in final_conversations_for_indexing
    ])

    # Rebuild FAISS index if necessary (new conversations, or index not loaded)
    if faiss_index is None or faiss_index.ntotal != embeddings_for_faiss.shape[0]:
        print(f"Rebuilding FAISS index from scratch. Old size: {faiss_index.ntotal if faiss_index else 0}, New size: {embeddings_for_faiss.shape[0]}.")
        d = embeddings_for_faiss.shape[1]
        faiss_index = faiss.IndexFlatL2(d)
        faiss_index.add(embeddings_for_faiss)
    else:
        print("FAISS index size matches data. No full rebuild needed (assuming no updates to existing entries).")

    # Save updated index and metadata
    try:
        faiss.write_index(faiss_index, FAISS_INDEX_PATH)
        
        # Use the custom encoder when dumping JSON
        with open(METADATA_PATH, 'w') as f:
            json.dump(updated_conversations_for_metadata, f, indent=2, cls=CustomJSONEncoder)
        print(f"FAISS index and metadata saved to {FAISS_INDEX_PATH} and {METADATA_PATH}.")
    except Exception as e:
        print(f"Error saving FAISS index or metadata: {e}")
        # Clean up potentially corrupted files
        if os.path.exists(FAISS_INDEX_PATH):
            os.remove(FAISS_INDEX_PATH)
        if os.path.exists(METADATA_PATH):
            os.remove(METADATA_PATH)

    return faiss_index, final_conversations_for_indexing