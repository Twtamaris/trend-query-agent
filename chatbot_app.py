import os
from google.cloud import bigquery
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import re # For regex parsing in LLM output

# LangChain components for GROQ and prompting
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
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
    # Use the project ID from the BigQuery client directly if it's initialized with one,
    # or ensure it's provided here (e.g., from an environment variable or config).
    # For now, let's keep it hardcoded as in your original example, but good practice
    # is to retrieve it dynamically or from a config file.
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
    
    processed_results = []
    for row in results:
        row_dict = dict(row)
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
        print(f"Error during LLM call or parsing summarization: {e}")
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
    print(f"\n--- Processing User Query: '{user_query}' with Customer Conversation Agent ---")

    start_date_str = None
    end_date_str = None
    if days_back:
        today = datetime.now()
        start_date = today - timedelta(days=days_back)
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = today.strftime('%Y-%m-%d')
    
    print("Step 1: Filtering conversations using BigQuery (and retrieving embeddings)...")
    bq_filtered_conversations = get_conversation_data_with_embeddings(
        zip_code=zip_code,
        start_date=start_date_str,
        end_date=end_date_str,
        sentiment=sentiment,
        issue_category=issue_category
    )

    if not bq_filtered_conversations:
        return "No relevant conversations found based on your specified structured criteria."

    conversations_with_embeddings = [
        conv for conv in bq_filtered_conversations 
        if conv.get('embedding') is not None and conv.get('conversation_text') is not None
    ]

    if not conversations_with_embeddings:
        return "No conversations with valid embeddings found after filtering."
        
    filtered_embeddings = np.array([conv['embedding'] for conv in conversations_with_embeddings])
    
    print(f" → Found {len(filtered_embeddings)} conversations after BigQuery filtering.")


    print("Step 2: Generating embedding for the user query...")
    query_embedding = embedding_model.encode([user_query], normalize_embeddings=True)[0]
    print(" → Query embedding generated.")

    print("Step 3: Performing vector similarity search on filtered results...")
    similarities = []
    for conv_emb in filtered_embeddings:
        try:
            similarities.append(1 - cosine(query_embedding, conv_emb))
        except ValueError as e:
            similarities.append(-1.0) # Assign a low similarity score

    top_n = 5
    top_indices = np.argsort(similarities)[::-1][:top_n]

    valid_top_indices = [idx for idx in top_indices if 0 <= idx < len(conversations_with_embeddings)]
    top_similar_conversations_data = [conversations_with_embeddings[idx] for idx in valid_top_indices]

    top_similar_conversations_data = [
        conv for conv, sim_score in zip(top_similar_conversations_data, [similarities[idx] for idx in valid_top_indices])
        if sim_score != -1.0
    ]

    if not top_similar_conversations_data:
        return "No semantically similar conversations found within the filtered set."

    print(f" → Found {len(top_similar_conversations_data)} top similar conversations.")

    print("Step 4: Synthesizing answer using LLM...")
    llm_summary_output = summarize_conversations_with_llm(top_similar_conversations_data)

    final_answer = f"Based on your query and relevant conversations:\n\n"
    final_answer += f"**Summary of Top Conversations:** {llm_summary_output['summary']}\n"
    final_answer += f"**Key Issues Identified:** {', '.join(llm_summary_output['issues'])}\n"
    final_answer += f"**Overall Sentiment (of top results):** {llm_summary_output['overall_sentiment']}\n\n"
    
    return final_answer

# --- GraphRAG Integration: Step 6 ---

BRAND_DOCUMENTS_DIR = "data/brand_documents"

if not os.path.exists(BRAND_DOCUMENTS_DIR):
    os.makedirs(BRAND_DOCUMENTS_DIR)
    print(f"Created directory: '{BRAND_DOCUMENTS_DIR}'. Please place your brand documents here for GraphRAG.")
    # Create dummy files for initial testing
    with open(os.path.join(BRAND_DOCUMENTS_DIR, "brand_overview.txt"), "w") as f:
        f.write("InnovateTech is a leading AI solutions provider. Our core mission is to democratize AI. Our flagship product, 'AI-Assistant Pro', focuses on natural language understanding and automated customer support. A significant trend is the push for explainable AI. We aim to integrate more deeply with enterprise CRM systems.")
    with open(os.path.join(BRAND_DOCUMENTS_DIR, "product_roadmap.txt"), "w") as f:
        f.write("The roadmap for AI-Assistant Pro includes enhancements for real-time sentiment analysis and predictive issue detection. We are exploring partnerships with major cloud providers. The market shows a strong trend towards customized AI agents for specific industries. Our competitor, 'Global AI Solutions', recently launched a similar product, but ours offers superior scalability.")
    print("Created dummy brand documents for demonstration purposes.")


def query_knowledge_graph_agent(user_query: str) -> str:
    """
    Simulates querying a GraphRAG-built knowledge graph for nuanced brand information,
    trends, product features, and competitive analysis.
    This function acts as the "Topic Agent" or "Trend Agent."
    """
    print(f"\n--- Processing User Query: '{user_query}' with Knowledge Graph Agent ---")

    # --- SIMULATED GRAPH RETRIEVAL ---
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

# --- Intent Classification LLM Setup ---
# This LLM will analyze the user's query and decide the intent and extract parameters.
intent_classification_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful assistant for routing user queries.
            Your task is to analyze the user's query and determine their intent.
            You should output a JSON object with two main keys:
            - 'intent': This should be one of 'customer_conversation', 'brand_knowledge', or 'unclear'.
            - 'parameters': This should be a JSON object containing any relevant extracted parameters.

            For 'customer_conversation' intent, extract the following parameters if present:
            - 'zip_code': (string, e.g., "90210")
            - 'days_back': (integer, e.g., 7 for 'last 7 days')
            - 'sentiment': (string, e.g., "positive", "negative", "neutral")
            - 'issue_category': (string, e.g., "billing", "internet speed", "service outage")

            For 'brand_knowledge' intent, no specific parameters are required, but you can extract general topics if they are clear.

            Example Outputs:
            1. User: "Summarize conversations for the last 7 days in zip code 12345 with negative sentiment."
               {{
                 "intent": "customer_conversation",
                 "parameters": {{
                   "zip_code": "12345",
                   "days_back": 7,
                   "sentiment": "negative"
                 }}
               }}
            2. User: "What are the top trends for InnovateTech?"
               {{
                 "intent": "brand_knowledge",
                 "parameters": {{}}
               }}
            3. User: "Tell me about their new AI features."
               {{
                 "intent": "brand_knowledge",
                 "parameters": {{}}
               }}
            4. User: "How are customers feeling about internet speed issues in 90210?"
               {{
                 "intent": "customer_conversation",
                 "parameters": {{
                   "zip_code": "90210",
                   "issue_category": "internet speed"
                 }}
               }}
            5. User: "Hello, how are you?"
               {{
                 "intent": "unclear",
                 "parameters": {{}}
               }}
            
            YOUR ENTIRE RESPONSE MUST BE ONLY THE JSON OBJECT, NOTHING ELSE. DO NOT INCLUDE ANY INTRODUCTORY OR CONCLUDING REMARKS, OR MARKDOWN BACKTICKS (```json).
            """
        ),
        ("human", "User query: {user_query}"),
    ]
)

intent_classification_chain = intent_classification_prompt | llm | StrOutputParser()


def chatbot_main_agent(user_query: str) -> str:
    """
    The main entry point for the chatbot. It uses an LLM to determine the user's intent
    and routes the query to either the Customer Conversation Agent (BigQuery+Vector)
    or the Knowledge Graph (Brand/Trend) Agent.
    """
    print(f"\n--- Chatbot Main Agent Processing Query: '{user_query}' ---")

    try:
        # Step 1: Use LLM for Intent Classification and Parameter Extraction
        print("Step 1: Classifying intent and extracting parameters using LLM...")
        llm_routing_response = intent_classification_chain.invoke({"user_query": user_query})
        
        print("-------------------------------")
        print("-------------------------------")
        print("-------------------------------")
        print("-------------------------------")
        print("-------------------------------")
        print(llm_routing_response)
        print("-------------------------------")
        print("-------------------------------")
        print("-------------------------------")
        print("-------------------------------")
        print("-------------------------------")
        
        
        # # Robustly parse the JSON output from the LLM
        # json_match = re.search(r'\{(.*?)\}', llm_routing_response, re.DOTALL)
        # if json_match:
        #     json_string = "{" + json_match.group(1) + "}"
        # else:
        #     print(f"Warning: LLM did not return valid JSON for intent classification. Raw: {llm_routing_response}")
        #     # Fallback for parsing issues: assume unclear intent
        #     parsed_routing = {"intent": "unclear", "parameters": {}}
        
        # Remove this(hehe)(not now)
        json_string = llm_routing_response

        try:
            parsed_routing = json.loads(json_string)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from LLM intent: {e}. Raw: {json_string}")
            # Fallback for JSON parsing error
            parsed_routing = {"intent": "unclear", "parameters": {}}


        intent = parsed_routing.get("intent", "unclear")
        params = parsed_routing.get("parameters", {})

        print(f" → Detected Intent: '{intent}', Parameters: {params}")

        # Step 2: Route based on detected intent
        if intent == "customer_conversation":
            print("--- Routing to Customer Conversation Agent ---")
            return answer_user_query_hybrid(user_query, **params)
        elif intent == "brand_knowledge":
            print("--- Routing to Knowledge Graph (Brand/Trend) Agent ---")
            return query_knowledge_graph_agent(user_query)
        else:
            print("--- Intent unclear, providing a general response ---")
            return "I'm sorry, I couldn't understand your request clearly. Could you please rephrase it or ask about customer conversations (e.g., summary, issues, sentiment, zip code) or about InnovateTech's brand, products, or trends?"

    except Exception as e:
        print(f"An error occurred during intent classification: {e}")
        return "I apologize, but I encountered an internal error. Please try again later."


# --- Main execution flow ---
if __name__ == "__main__":
    print("--- Starting Chatbot Main Agent Demonstrations ---")

    # --- Test GraphRAG Agent Queries (routed by LLM) ---
    print("\n--- Testing Knowledge Graph (Brand/Trend) Agent ---")
    
    query_graph_1 = "What are the top emerging trends for InnovateTech in the AI space?"
    print(chatbot_main_agent(query_graph_1))

    query_graph_2 = "Can you describe the key features of AI-Assistant Pro and its future roadmap?"
    print(chatbot_main_agent(query_graph_2))

    query_graph_3 = "How does InnovateTech's product compare to Global AI Solutions?"
    print(chatbot_main_agent(query_graph_3))

    query_graph_4 = "What is InnovateTech's core mission and vision?"
    print(chatbot_main_agent(query_graph_4))

    query_graph_5 = "Are there any new directions in AI solutions that InnovateTech is considering?"
    print(chatbot_main_agent(query_graph_5))

    # --- Test BigQuery + Vector Search Agent Queries (routed by LLM) ---
    print("\n--- Testing Customer Conversation (BigQuery + Vector) Agent ---")

    # IMPORTANT: Replace with a zip code that exists in your BigQuery data for testing.
    # Ensure your BigQuery table has conversations within the specified days for these zip codes.
    user_query_bigquery_1 = "Summarize conversations for the last 10 days in zip code 92122."
    print(chatbot_main_agent(user_query_bigquery_1))

    user_query_bigquery_2 = "How are customers feeling about internet connectivity issues in 92122 over the past 2 weeks?"
    print(chatbot_main_agent(user_query_bigquery_2)) # LLM should extract zip, issue, days_back

    user_query_bigquery_3 = "Show me negative sentiment issues from last month."
    # This query doesn't specify zip code, so it will search broadly, then filter by sentiment.
    print(chatbot_main_agent(user_query_bigquery_3)) # LLM should extract sentiment, and interpret "last month" for days_back

    # --- Test Unclear / Default Queries ---
    print("\n--- Testing Unclear / Default Handling ---")
    query_unclear_1 = "Hello chatbot, how are you today?"
    print(chatbot_main_agent(query_unclear_1))

    query_unclear_2 = "Tell me a joke."
    print(chatbot_main_agent(query_unclear_2))

    print("\n--- Chatbot Main Agent Demonstration Completed ---")