import os
from google.cloud import bigquery
import pandas as pd
import numpy as np # For handling numpy arrays
from datetime import datetime, timedelta
import json # For parsing LLM output

# LangChain components for GROQ and prompting
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine # For calculating cosine similarity
from dotenv import load_dotenv

# Load environment variables (ensure GROQ_API_KEY is set)
load_dotenv()

# --- Initialize BigQuery client ---
client = bigquery.Client()

# --- Initialize GROQ LLM ---
try:
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable not set.")
    # Using 'llama3-8b-8192' as it's a good balance for speed and summarization
    llm = ChatGroq(temperature=0.0, groq_api_key=groq_api_key, model_name="llama3-8b-8192")
except ValueError as e:
    print(f"Error initializing ChatGroq: {e}")
    print("Please set the GROQ_API_KEY environment variable.")
    exit()

# --- Load embedding model (same one used for generating embeddings in Step 4) ---
print("Loading Sentence Transformer model 'all-MiniLM-L6-v2'...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.\n")

# --- BigQuery Data Retrieval Function (MODIFIED for embeddings) ---
def get_conversation_data_with_embeddings(zip_code=None, start_date=None, end_date=None, sentiment=None, issue_category=None):
    """
    Retrieves conversation data including embeddings from BigQuery based on filters.
    Dates should be in 'YYYY-MM-DD' format.
    Queries the 'conversations_with_embeddings' table.
    """
    project_id = "chatbot-project-464108"
    table_name = "conversations_with_embeddings" # Using the table created in Step 4

    query = f"""
    SELECT
        conversation_id,
        conversation_text,
        sentiment,
        issue_category,
        timestamp,
        zip_code,
        embedding
    FROM
        `{project_id}.customer_service_data.{table_name}`
    WHERE
        1=1
    """

    if zip_code:
        query += f" AND zip_code = '{zip_code}'"
    if start_date:
        query += f" AND DATE(timestamp) >= DATE('{start_date}')"
    if end_date:
        query += f" AND DATE(timestamp) <= DATE('{end_date}')"
    if sentiment:
        query += f" AND LOWER(sentiment) = LOWER('{sentiment}')"
    if issue_category:
        query += f" AND LOWER(issue_category) = LOWER('{issue_category}')"

    print(f"Executing BigQuery query for filtering:\n{query}")
    query_job = client.query(query)
    results = query_job.result()
    
    # Convert BigQuery Row objects to dictionaries and ensure embedding is a list of floats
    processed_results = []
    for row in results:
        row_dict = dict(row)
        # BigQuery returns ARRAY<FLOAT64> as a tuple of floats, convert to list
        if 'embedding' in row_dict and isinstance(row_dict['embedding'], (tuple, list)):
            row_dict['embedding'] = list(row_dict['embedding'])
        processed_results.append(row_dict)
        
        
    print("HEHEHEHEHEHEHEHEHEHEEHEHEHEHEEHEHEHEHEHEHEH")
    print("HEHEHEHEHEHEHEHEHEHEEHEHEHEHEEHEHEHEHEHEHEH")
    
    print("HEHEHEHEHEHEHEHEHEHEEHEHEHEHEEHEHEHEHEHEHEH")
    
    print("HEHEHEHEHEHEHEHEHEHEEHEHEHEHEEHEHEHEHEHEHEH")
    
    print(processed_results)
    
    
    return processed_results

# --- LLM-Powered Summarization Function (from previous steps) ---
def summarize_conversations_with_llm(conversations):
    """
    Summarizes conversations, identifies key issues, and overall sentiment using an LLM.
    """
    if not conversations:
        return {
            "summary": "No conversations found for the given criteria.",
            "issues": [],
            "overall_sentiment": "N/A"
        }

    all_texts = [conv['conversation_text'] for conv in conversations if conv['conversation_text']]
    if not all_texts:
        return {
            "summary": "No conversation text available.",
            "issues": [],
            "overall_sentiment": "N/A"
        }

    # Limit the number of conversations to send to the LLM to manage context window and cost
    conversations_for_llm = "\n---\n".join(all_texts[:5]) 

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert customer service assistant. Your task is to analyze customer conversations.
                Summarize the main points, identify distinct issues mentioned, and determine the overall sentiment.
                Provide the output in a JSON format with the following keys:
                - 'summary': A concise summary of all conversations.
                - 'issues': A list of unique, distinct issues mentioned by customers.
                - 'overall_sentiment': The overall sentiment (Positive, Negative, Neutral, or Mixed) across all conversations.
                YOUR ENTIRE RESPONSE MUST BE ONLY THE JSON OBJECT, NOTHING ELSE. DO NOT INCLUDE ANY INTRODUCTORY OR CONCLUDING REMARKS, OR MARKDOWN BACKTICKS (```json).
                Example JSON output:
                {{
                    "summary": "Customer service issues included internet slowness and billing disputes.",
                    "issues": ["Internet speed", "Billing error"],
                    "overall_sentiment": "Negative"
                }}
                """
            ),
            ("human", "Analyze the following customer service conversations:\n\n{conversations}"),
        ]
    )

    llm_chain = prompt_template | llm | StrOutputParser()

    try:
        llm_response = llm_chain.invoke({"conversations": conversations_for_llm})
        # print(f"\nLLM Raw Response:\n{llm_response}") # For debugging LLM output

        # Attempt to parse the JSON response
        # This regex looks for a block starting with '{' and ending with '}'
        json_match = re.search(r'\{(.*?)\}', llm_response, re.DOTALL)

        if json_match:
            json_string = "{" + json_match.group(1) + "}"
        else:
            json_string = llm_response.strip()

        parsed_response = json.loads(json_string)

        return {
            "summary": parsed_response.get("summary", "No summary provided by LLM."),
            "issues": parsed_response.get("issues", []),
            "overall_sentiment": parsed_response.get("overall_sentiment", "N/A")
        }

    except Exception as e:
        print(f"Error during LLM call or parsing: {e}")
        # Fallback to basic summary if LLM fails
        fallback_summary_text = f"LLM summarization failed. Found {len(conversations)} conversations.\n"
        fallback_issues = {}
        fallback_sentiments = {'positive': 0, 'negative': 0, 'neutral': 0}
        for conv in conversations:
            if conv.get('issue_category'):
                fallback_issues[conv['issue_category']] = fallback_issues.get(conv['issue_category'], 0) + 1
            if conv.get('sentiment'):
                fallback_sentiments[conv['sentiment'].lower()] = fallback_sentiments.get(conv['sentiment'].lower(), 0) + 1
        
        return {
            "summary": fallback_summary_text,
            "issues": list(fallback_issues.keys()),
            "overall_sentiment": next(iter(fallback_sentiments.keys())) if fallback_sentiments else "N/A"
        }


def answer_user_query_hybrid(user_query, zip_code=None, days_back=None, sentiment=None, issue_category=None):
    """
    Answers a user query using a hybrid approach: BigQuery filtering + vector search.
    """
    print(f"\n--- Processing User Query: '{user_query}' ---")

    # 1. Extract Structured Filters (basic parsing, extend for more complexity)
    start_date_str = None
    end_date_str = None
    if days_back:
        today = datetime.now()
        start_date = today - timedelta(days=days_back)
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = today.strftime('%Y-%m-%d')
    
    # NOTE: For real-world use, you would use an NLU model (or another LLM call)
    # to extract `zip_code`, `days_back`, `sentiment`, `issue_category`
    # directly from the `user_query` if they are not provided as separate arguments.
    # For this example, we assume these are either provided or manually extracted.


    # 2. BigQuery Filtering (get relevant conversations with their embeddings)
    print("Step 2: Filtering conversations using BigQuery (and retrieving embeddings)...")
    bq_filtered_conversations = get_conversation_data_with_embeddings(
        zip_code=zip_code,
        start_date=start_date_str,
        end_date=end_date_str,
        sentiment=sentiment,
        issue_category=issue_category
    )

    if not bq_filtered_conversations:
        return "No relevant conversations found based on your specified structured criteria."

    # Separate conversation texts and their embeddings for similarity search
    # Ensure all conversations have an embedding and conversation_text
    conversations_with_embeddings = [
        conv for conv in bq_filtered_conversations 
        if conv.get('embedding') is not None and conv.get('conversation_text') is not None
    ]

    if not conversations_with_embeddings:
        return "No conversations with valid embeddings found after filtering."
        
    filtered_texts = [conv['conversation_text'] for conv in conversations_with_embeddings]
    filtered_embeddings = np.array([conv['embedding'] for conv in conversations_with_embeddings])
    
    print(f" → Found {len(filtered_embeddings)} conversations after BigQuery filtering.")


    # 3. Generate Query Embedding
    print("Step 3: Generating embedding for the user query...")
    query_embedding = embedding_model.encode([user_query], normalize_embeddings=True)[0]
    print(" → Query embedding generated.")

    # 4. Vector Similarity Search on Filtered Embeddings
    print("Step 4: Performing vector similarity search on filtered results...")
    # Calculate cosine similarity (1 - cosine distance)
    # Handle cases where embeddings might have different dimensions or be problematic
    similarities = []
    for conv_emb in filtered_embeddings:
        try:
            similarities.append(1 - cosine(query_embedding, conv_emb))
        except ValueError as e:
            # Handle cases where embedding might be malformed or empty
            similarities.append(-1.0) # Assign a low similarity score
            # print(f"Warning: Could not calculate similarity for an embedding: {e}")

    # Get the top N most similar conversations (e.g., top 5 or 10)
    top_n = 5
    # Use np.argsort to get indices that would sort the similarities in ascending order
    # Then slice [::-1] to reverse for descending order (highest similarity first)
    # Then take [:top_n] for the top N.
    top_indices = np.argsort(similarities)[::-1][:top_n]

    # Reconstruct the top conversations from the original filtered list
    # Ensure indices are valid
    valid_top_indices = [idx for idx in top_indices if 0 <= idx < len(conversations_with_embeddings)]
    top_similar_conversations_data = [conversations_with_embeddings[idx] for idx in valid_top_indices]

    # Filter out conversations with similarity -1.0 if any were problematic
    top_similar_conversations_data = [
        conv for conv, sim_score in zip(top_similar_conversations_data, [similarities[idx] for idx in valid_top_indices])
        if sim_score != -1.0
    ]

    if not top_similar_conversations_data:
        return "No semantically similar conversations found within the filtered set."

    print(f" → Found {len(top_similar_conversations_data)} top similar conversations.")

    # 5. LLM for Synthesis
    print("Step 5: Synthesizing answer using LLM...")
    # Pass the actual data of the top similar conversations to your LLM summarization function
    llm_summary_output = summarize_conversations_with_llm(top_similar_conversations_data)

    final_answer = f"Based on your query and relevant conversations:\n\n"
    final_answer += f"**Summary of Top Conversations:** {llm_summary_output['summary']}\n"
    final_answer += f"**Key Issues Identified:** {', '.join(llm_summary_output['issues'])}\n"
    final_answer += f"**Overall Sentiment (of top results):** {llm_summary_output['overall_sentiment']}\n\n"
    
    # Optionally add details of the top conversations for transparency
    # final_answer += "--- Details of Most Similar Conversations ---\n"
    # for i, conv in enumerate(top_similar_conversations_data):
    #     final_answer += f"Conversation {i+1} (ID: {conv.get('conversation_id', 'N/A')}, Score: {similarities[valid_top_indices[i]]:.2f}):\n"
    #     final_answer += f"  Text: {conv.get('conversation_text', '')[:200]}...\n"
    #     if conv.get('sentiment'): final_answer += f"  Sentiment: {conv['sentiment']}\n"
    #     if conv.get('issue_category'): final_answer += f"  Issue: {conv['issue_category']}\n"
    #     final_answer += "---\n"


    return final_answer

# --- Main execution flow for Step 5 ---
if __name__ == "__main__":
    import re # Ensure re is imported for summarize_conversations_with_llm

    # Example 1: User asks about general issues in a specific zip code recently
    user_query_1 = "What are the common problems customers faced recently in zip code 90210?"
    zip_code_1 = '90210' # Example zip from your data
    days_back_1 = 10   # Example recent days
    print(answer_user_query_hybrid(user_query_1, zip_code=zip_code_1, days_back=days_back_1))

    # Example 2: User asks about negative sentiment related to service outages
    user_query_2 = "Tell me about negative experiences related to service outages in my area (90210)."
    zip_code_2 = '90210'
    sentiment_2 = 'Negative'
    issue_category_2 = 'Service Outage' # Assuming this is one of your categories
    days_back_2 = 14 # Look back further
    print(answer_user_query_hybrid(user_query_2, zip_code=zip_code_2, sentiment=sentiment_2, issue_category=issue_category_2, days_back=days_back_2))

    # Example 3: A very specific query without structured filters, relying heavily on semantic search
    user_query_3 = "Customers complaining about their internet constantly dropping."
    # If no zip_code or days_back are provided, the BigQuery filter will return all conversations
    # which can be slow if the table is very large.
    print(answer_user_query_hybrid(user_query_3, days_back=7)) # still limiting by date for performance

    print("\n--- Step 5 (Hybrid Search) Completed ---")
    print("This setup demonstrates how to combine BigQuery's filtering with semantic search using embeddings stored in BigQuery.")