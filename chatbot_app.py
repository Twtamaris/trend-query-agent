import os
from google.cloud import bigquery
import pandas as pd
import numpy as np # For handling numpy arrays
from datetime import datetime, timedelta
import json # For parsing LLM output
import re # For regex parsing in LLM output

# LangChain components for GROQ and prompting
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine # For calculating cosine similarity
from dotenv import load_dotenv

# --- GraphRAG Specific Imports (Conceptual) ---
# In a real GraphRAG project, you'd likely interact with its API for
# graph building and querying. These are placeholders for illustration.
# You might need to import specific components from graphrag depending on
# how you integrate (e.g., if you're using its query interface directly).
# For full integration, you might have GraphRAG manage its own LLM and embeddings.
# For simplicity, we'll assume GraphRAG has *pre-built* a graph, and we're
# simulating retrieval for the LLM.
# from graphrag.config import GraphRAGConfig
# from graphrag.core import GraphRAG

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
    project_id = "chatbot-project-464108" # Replace with your actual project ID
    dataset_id = "customer_service_data" # Replace with your actual dataset ID
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
        `{project_id}.{dataset_id}.{table_name}`
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
            
    print(f"Retrieved {len(processed_results)} rows from BigQuery after filtering.")
    return processed_results

# --- LLM-Powered Summarization Function ---
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
    similarities = []
    for conv_emb in filtered_embeddings:
        try:
            similarities.append(1 - cosine(query_embedding, conv_emb))
        except ValueError as e:
            similarities.append(-1.0) # Assign a low similarity score
            # print(f"Warning: Could not calculate similarity for an embedding: {e}") # Debug if needed

    top_n = 5 # Get the top N most similar conversations
    top_indices = np.argsort(similarities)[::-1][:top_n]

    valid_top_indices = [idx for idx in top_indices if 0 <= idx < len(conversations_with_embeddings)]
    top_similar_conversations_data = [conversations_with_embeddings[idx] for idx in valid_top_indices]

    top_similar_conversations_data = [
        conv for conv, sim_score in zip(top_similar_conversations_data, [similarities[idx] for idx in valid_top_indices])
        if sim_score != -1.0 # Filter out problematic similarities
    ]

    if not top_similar_conversations_data:
        return "No semantically similar conversations found within the filtered set."

    print(f" → Found {len(top_similar_conversations_data)} top similar conversations.")

    # 5. LLM for Synthesis
    print("Step 5: Synthesizing answer using LLM...")
    llm_summary_output = summarize_conversations_with_llm(top_similar_conversations_data)

    final_answer = f"Based on your query and relevant conversations:\n\n"
    final_answer += f"**Summary of Top Conversations:** {llm_summary_output['summary']}\n"
    final_answer += f"**Key Issues Identified:** {', '.join(llm_summary_output['issues'])}\n"
    final_answer += f"**Overall Sentiment (of top results):** {llm_summary_output['overall_sentiment']}\n\n"
    
    return final_answer

# --- GraphRAG Integration: Step 6 ---

# Conceptual setup for GraphRAG document folder:
# You would place your brand-specific documents (e.g., 'brand_overview.txt', 'product_strategy.md')
# into this directory.
BRAND_DOCUMENTS_DIR = "data/brand_documents"

if not os.path.exists(BRAND_DOCUMENTS_DIR):
    os.makedirs(BRAND_DOCUMENTS_DIR)
    print(f"Created directory: '{BRAND_DOCUMENTS_DIR}'. Please place your brand documents here for GraphRAG.")
    # You might want to create some dummy files for initial testing
    with open(os.path.join(BRAND_DOCUMENTS_DIR, "brand_overview.txt"), "w") as f:
        f.write("InnovateTech is a leading AI solutions provider. Our core mission is to democratize AI. Our flagship product, 'AI-Assistant Pro', focuses on natural language understanding and automated customer support. A significant trend is the push for explainable AI. We aim to integrate more deeply with enterprise CRM systems.")
    with open(os.path.join(BRAND_DOCUMENTS_DIR, "product_roadmap.txt"), "w") as f:
        f.write("The roadmap for AI-Assistant Pro includes enhancements for real-time sentiment analysis and predictive issue detection. We are exploring partnerships with major cloud providers. The market shows a strong trend towards customized AI agents for specific industries. Our competitor, 'Global AI Solutions', recently launched a similar product, but ours offers superior scalability.")
    print("Created dummy brand documents for demonstration purposes.")


# --- GraphRAG Querying Function (Simulated) ---
# In a full GraphRAG integration, this function would:
# 1. Use GraphRAG's configuration to load the pre-built graph.
# 2. Use GraphRAG's internal query interface to retrieve relevant nodes/edges/subgraphs
#    based on the user's query (often involving embeddings and graph algorithms).
# 3. Format the retrieved graph information into a concise text context for the LLM.

def query_knowledge_graph_agent(user_query: str) -> str:
    """
    Simulates querying a GraphRAG-built knowledge graph for nuanced brand information,
    trends, product features, and competitive analysis.
    This function acts as the "Topic Agent" or "Trend Agent."
    """
    print(f"\n--- Querying Knowledge Graph Agent for: '{user_query}' ---")

    # --- SIMULATED GRAPH RETRIEVAL ---
    # In a real GraphRAG setup, this would be the result of a complex process:
    # 1. GraphRAG pipeline running on documents to create entities, relationships.
    # 2. Query embedding: user_query -> embedding.
    # 3. Graph traversal/search: Use the query embedding to find relevant nodes/edges in the graph.
    # 4. Context assembly: Extract relevant facts/text snippets from those graph components.
    
    # For this demonstration, we'll use a very basic keyword-driven simulation
    # to provide 'context' that an actual GraphRAG retrieval would yield.
    retrieved_graph_context = ""
    query_lower = user_query.lower()

    if "trend" in query_lower or "trends" in query_lower or "emerging patterns" in query_lower or "future" in query_lower:
        retrieved_graph_context = (
            "A significant trend observed is the push for explainable AI. "
            "Another strong trend is towards customized AI agents for specific industries. "
            "Future developments for AI-Assistant Pro include enhancements for real-time sentiment analysis and predictive issue detection. "
            "We are exploring partnerships with major cloud providers."
        )
    elif "brand" in query_lower or "innovatetech" in query_lower or "mission" in query_lower:
        retrieved_graph_context = (
            "InnovateTech is a leading AI solutions provider. "
            "Our core mission is to democratize AI. "
            "Our flagship product is 'AI-Assistant Pro'."
        )
    elif "product" in query_lower or "features" in query_lower or "ai-assistant pro" in query_lower:
        retrieved_graph_context = (
            "'AI-Assistant Pro' focuses on natural language understanding and automated customer support. "
            "Planned enhancements include real-time sentiment analysis and predictive issue detection. "
            "It aims to integrate more deeply with enterprise CRM systems."
        )
    elif "competitor" in query_lower or "compare" in query_lower or "global ai solutions" in query_lower:
        retrieved_graph_context = (
            "Our competitor is 'Global AI Solutions'. They recently launched a similar product, "
            "but AI-Assistant Pro offers superior scalability."
        )
    else:
        # Default, general info if no specific keyword matches.
        retrieved_graph_context = (
            "The brand is InnovateTech, a leading AI solutions provider whose mission is to democratize AI. "
            "Its flagship product is AI-Assistant Pro, focusing on natural language understanding and automated customer support. "
            "Key trends include explainable AI and customized AI agents. "
            "Future plans involve real-time sentiment analysis and predictive issue detection. "
            "Global AI Solutions is a competitor, but InnovateTech's product offers superior scalability."
        )

    if not retrieved_graph_context:
        return "I could not find relevant information in the knowledge graph for your query."

    print(f" → Simulated GraphRAG Context Provided to LLM:\n{retrieved_graph_context[:300]}...")

    # --- LLM for Nuanced Answer Generation ---
    # The LLM takes the user's query and the context retrieved from the graph.
    # It synthesizes the information to answer nuanced questions.
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an intelligent 'Trend Agent' and 'Topic Agent' for the brand InnovateTech.
                Your task is to provide concise, insightful, and nuanced answers about the brand,
                its products, market trends, and competitive landscape, based *only* on the provided context.
                Do not mention 'knowledge graph' or 'documents'. Synthesize the information clearly.
                If the context does not contain enough information to answer, state that.
                """
            ),
            ("human", "User Query: {user_query}\n\nContext from Knowledge Graph:\n{context}"),
        ]
    )

    llm_chain = prompt_template | llm | StrOutputParser()

    try:
        final_answer = llm_chain.invoke({"user_query": user_query, "context": retrieved_graph_context})
        print(f" → LLM's Nuanced Answer from Knowledge Graph:\n{final_answer}")
        return final_answer
    except Exception as e:
        print(f"Error during LLM call for knowledge graph query: {e}")
        return "Sorry, I could not synthesize an answer from the knowledge graph at this time."

# --- Main Chatbot Orchestration (Intent Classification) ---
# This function routes the user's query to the appropriate agent (BigQuery or GraphRAG).
# For a production system, this intent classification would be more robust,
# perhaps using a dedicated LLM call or a sophisticated NLU model.

def chatbot_main_agent(user_query: str, **kwargs) -> str:
    """
    The main entry point for the chatbot. It determines the user's intent
    and routes the query to either the Customer Conversation Agent (BigQuery+Vector)
    or the Knowledge Graph (Brand/Trend) Agent.
    """
    query_lower = user_query.lower()

    # Keywords for routing to the Knowledge Graph Agent
    graph_keywords = ["trend", "trends", "emerging", "future", "roadmap", "brand", "innovatetech",
                      "product", "features", "ai-assistant pro", "competitor", "compare", "global ai solutions",
                      "mission", "strategy", "vision"]

    # Keywords for routing to the Customer Conversation Agent
    bigquery_keywords = ["summarize", "conversations", "issues", "sentiment", "problems",
                         "zip code", "area", "geography", "customer service", "complaints"]

    # Simple heuristic-based routing
    if any(keyword in query_lower for keyword in graph_keywords) and \
       not any(keyword in query_lower for keyword in bigquery_keywords): # Prioritize graph if relevant and not strong BQ
        print("\n--- Routing query to Knowledge Graph (Brand/Trend) Agent ---")
        return query_knowledge_graph_agent(user_query)
    elif any(keyword in query_lower for keyword in bigquery_keywords):
        print("\n--- Routing query to Customer Conversation (BigQuery + Vector) Agent ---")
        # Pass relevant kwargs if available
        return answer_user_query_hybrid(
            user_query,
            zip_code=kwargs.get('zip_code'),
            days_back=kwargs.get('days_back'),
            sentiment=kwargs.get('sentiment'),
            issue_category=kwargs.get('issue_category')
        )
    else:
        # If no clear intent, you might default to one or ask for clarification.
        # For now, we'll try the hybrid search as it's more general for "data."
        print("\n--- No clear intent detected, defaulting to Customer Conversation Agent ---")
        return answer_user_query_hybrid(
            user_query,
            zip_code=kwargs.get('zip_code'),
            days_back=kwargs.get('days_back'),
            sentiment=kwargs.get('sentiment'),
            issue_category=kwargs.get('issue_category')
        )


# --- Main execution flow for Step 6 ---
if __name__ == "__main__":
    print("--- Starting Chatbot Main Agent Demonstrations ---")

    # --- Test GraphRAG Agent Queries ---
    print("\n--- Testing Knowledge Graph (Brand/Trend) Agent ---")
    
    # Query 1: Top trends
    query_graph_1 = "What are the top emerging trends for InnovateTech in the AI space?"
    print(chatbot_main_agent(query_graph_1))

    # Query 2: Product features
    query_graph_2 = "Can you describe the key features of AI-Assistant Pro and its future roadmap?"
    print(chatbot_main_agent(query_graph_2))

    # Query 3: Competitors
    query_graph_3 = "How does InnovateTech's product compare to Global AI Solutions?"
    print(chatbot_main_agent(query_graph_3))

    # Query 4: Brand mission
    query_graph_4 = "What is InnovateTech's core mission and vision?"
    print(chatbot_main_agent(query_graph_4))

    # Query 5: Nuanced query for trends that might not have explicit keywords
    query_graph_5 = "Are there any new directions in AI solutions that InnovateTech is considering?"
    print(chatbot_main_agent(query_graph_5))

    # --- Test BigQuery + Vector Search Agent Queries ---
    # These will only work if your BigQuery table 'conversations_with_embeddings'
    # is populated with data (as per your Step 4 and 5 setup).
    print("\n--- Testing Customer Conversation (BigQuery + Vector) Agent (Requires BigQuery Data) ---")

    # Example: User asks about general issues in a specific zip code recently
    # IMPORTANT: Replace with a zip code that exists in your BigQuery data for testing.
    # Also, ensure your BigQuery table has conversations within the last 10 days for this zip.
    user_query_bigquery_1 = "What are the common problems customers faced recently in zip code 92122?"
    print(chatbot_main_agent(user_query_bigquery_1, zip_code='92122', days_back=10))

    # Example: User asks about negative sentiment related to billing issues
    # IMPORTANT: Replace with a zip code, sentiment, and issue_category that exist in your BigQuery data.
    user_query_bigquery_2 = "Tell me about negative experiences related to internet connectivity issues in my area (92122)."
    print(chatbot_main_agent(user_query_bigquery_2, zip_code='92122', sentiment='Negative', issue_category='internet_connectivity', days_back=30))

    # Example: A general query that will default to BigQuery agent, limiting by date for performance
    user_query_bigquery_3 = "Summarize conversations about account management."
    print(chatbot_main_agent(user_query_bigquery_3, days_back=7, issue_category='account management'))


    print("\n--- Step 6 (GraphRAG Integration) Conceptual Code Demonstration Completed ---")
    print("\nNext Steps: Real GraphRAG setup, more robust intent classification, and potentially a user interface.")