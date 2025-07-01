import os
import json
import numpy as np
import faiss
from datetime import datetime
from bq_utils import get_conversation_data
from models import EMBEDDING_MODEL
from config import FAISS_INDEX_PATH, METADATA_PATH

def load_or_build_faiss_index():
    """
    Loads FAISS index and its corresponding metadata from disk.
    If not found, or if new data exists in BigQuery, it builds/updates the index
    with embeddings for all conversations and saves it.
    Returns the FAISS index and the list of conversation data used to build it.
    """
    faiss_index = None
    all_conversations_indexed = [] # This list will hold the actual conversation data, in the order of the FAISS index

    # Try to load existing index and metadata
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
        try:
            print("Attempting to load existing FAISS index and metadata...")
            faiss_index = faiss.read_index(FAISS_INDEX_PATH)
            with open(METADATA_PATH, 'r') as f:
                all_conversations_indexed = json.load(f)
                # Convert embeddings back to numpy array on load
                for conv in all_conversations_indexed:
                    if 'embedding' in conv and isinstance(conv['embedding'], list):
                        conv['embedding'] = np.array(conv['embedding']).astype('float32')
                    # Optionally convert timestamp string back to datetime if needed for other ops
                    if 'timestamp' in conv and isinstance(conv['timestamp'], str):
                        try:
                            conv['timestamp'] = datetime.fromisoformat(conv['timestamp'])
                        except ValueError:
                            pass # Keep as string if parsing fails

            print(f"Loaded FAISS index with {faiss_index.ntotal} vectors.")
            print(f"Loaded metadata for {len(all_conversations_indexed)} conversations.")

        except Exception as e:
            print(f"Error loading FAISS index or metadata: {e}. Rebuilding index.")
            faiss_index = None
            all_conversations_indexed = []

    # If no index loaded or an error occurred, proceed to build/update
    print("Fetching all conversations from BigQuery for indexing check...")
    # Fetch all conversations to ensure we catch any new ones
    bq_all_conversations = get_conversation_data()

    current_indexed_ids = {conv['conversation_id'] for conv in all_conversations_indexed if 'conversation_id' in conv}

    new_conversations_to_embed = []
    
    # Identify new conversations or conversat   ions with no text/ID
    for conv_data in bq_all_conversations:
        conv_id = conv_data.get('conversation_id')
        conv_text = conv_data.get('conversation_text')

        if conv_id not in current_indexed_ids and conv_text:
            new_conversations_to_embed.append(conv_data)
        elif conv_id in current_indexed_ids:
            # If it's already indexed, ensure it has its embedding for the final_embeddings_array
            # For simplicity, we assume if it's in all_conversations_indexed, its embedding is valid.
            pass # We will reconstruct `all_conversations_indexed` in next step
        else:
            print(f"Warning: Conversation {conv_id if conv_id else 'N/A'} has no text or ID. Skipping for indexing.")

    if new_conversations_to_embed:
        print(f"Identified {len(new_conversations_to_embed)} new conversations for embedding.")
        new_texts = [conv['conversation_text'] for conv in new_conversations_to_embed]
        print("Generating embeddings for new conversations...")
        new_embeddings = EMBEDDING_MODEL.encode(new_texts, normalize_embeddings=True)

        for i, conv_data in enumerate(new_conversations_to_embed):
            conv_data['embedding'] = new_embeddings[i].tolist() # Store as list for JSON serialization
            all_conversations_indexed.append(conv_data)

        print(f"Added {len(new_conversations_to_embed)} new embeddings to the data list.")
    else:
        print("No new conversations found in BigQuery to add to the index.")

    # Reconstruct the full list of data to be indexed, ensuring it contains all necessary fields
    # This implicitly removes any old conversations not present in bq_all_conversations anymore
    bq_conversations_map = {conv['conversation_id']: conv for conv in bq_all_conversations}

    final_conversations_for_indexing = []
    # Merge existing indexed data with new BigQuery data
    # Iterate over `all_conversations_indexed` first to preserve order/existing embeddings
    for conv_data in all_conversations_indexed:
        conv_id = conv_data.get('conversation_id')
        if conv_id in bq_conversations_map:
            # Prefer the latest text and metadata from BQ, but keep the existing embedding if available
            merged_conv_data = bq_conversations_map[conv_id]
            # Ensure the embedding from our local cache is used
            merged_conv_data['embedding'] = conv_data['embedding']
            final_conversations_for_indexing.append(merged_conv_data)
            del bq_conversations_map[conv_id] # Remove from map so we only process truly new ones

    # Add any remaining (truly new) conversations from BigQuery that weren't in all_conversations_indexed initially
    for conv_id in bq_conversations_map:
        conv_data = bq_conversations_map[conv_id]
        if 'embedding' not in conv_data and conv_data.get('conversation_text'): # If it's a new one not yet embedded
            print(f"Warning: Conversation {conv_id} from BQ was not pre-embedded. Embedding now.")
            conv_data['embedding'] = EMBEDDING_MODEL.encode([conv_data['conversation_text']], normalize_embeddings=True)[0].tolist()
        if 'embedding' in conv_data: # Only add if it now has an embedding
            final_conversations_for_indexing.append(conv_data)

    # Filter out any conversations that ended up without an embedding
    final_conversations_for_indexing = [conv for conv in final_conversations_for_indexing if conv.get('embedding') is not None]

    if not final_conversations_for_indexing:
        print("No valid conversations with embeddings to build a FAISS index. Returning empty index.")
        return None, []

    # Prepare embeddings for FAISS
    embeddings_for_faiss = np.array([
        np.array(conv['embedding']).astype('float32') for conv in final_conversations_for_indexing
    ])

    if faiss_index is None or faiss_index.ntotal != embeddings_for_faiss.shape[0]:
        print("Rebuilding FAISS index from scratch due to changes or initial build.")
        d = embeddings_for_faiss.shape[1]
        faiss_index = faiss.IndexFlatL2(d)
        faiss_index.add(embeddings_for_faiss)
    else:
        print("FAISS index size matches data. No full rebuild needed (assuming no updates to existing entries).")

    # Save updated index and metadata
    try:
        faiss.write_index(faiss_index, FAISS_INDEX_PATH)
        serializable_conversations = []
        for conv in final_conversations_for_indexing:
            serializable_conv = conv.copy()
            if 'embedding' in serializable_conv and isinstance(serializable_conv['embedding'], np.ndarray):
                serializable_conv['embedding'] = serializable_conv['embedding'].tolist()
            if 'timestamp' in serializable_conv and isinstance(serializable_conv['timestamp'], datetime):
                serializable_conv['timestamp'] = serializable_conv['timestamp'].isoformat()
            serializable_conversations.append(serializable_conv)

        with open(METADATA_PATH, 'w') as f:
            json.dump(serializable_conversations, f)
        print(f"FAISS index and metadata saved to {FAISS_INDEX_PATH} and {METADATA_PATH}.")
    except Exception as e:
        print(f"Error saving FAISS index or metadata: {e}")

    return faiss_index, final_conversations_for_indexing