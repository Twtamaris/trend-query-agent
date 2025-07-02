import os
import json
import numpy as np
import faiss
from datetime import datetime, timedelta
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from models import GROQ_LLM, OPENAI_LLM, EMBEDDING_MODEL # Import both LLMs
from bq_utils import get_conversation_data
from config import BRAND_DOCUMENTS_DIR
from graphrag_utils import query_graphrag # Import the GraphRAG utility

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

    # Ensure we only process conversations that have text
    all_texts = [conv['conversation_text'] for conv in conversations if conv.get('conversation_text')]
    if not all_texts:
        return {
            "summary": "No conversation text available for summarization.",
            "issues": [],
            "overall_sentiment": "N/A"
        }

    # Limit the number of conversations sent to the LLM for summarization context
    # This helps manage token limits and focuses the summary.
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

    llm_chain = prompt_template | GROQ_LLM | StrOutputParser()

    llm_response = "" # Initialize outside try block
    try:
        llm_response = llm_chain.invoke({"conversations": conversations_for_llm})

        # --- ROBUST JSON PARSING ---
        # 1. Strip leading/trailing whitespace
        json_string_candidate = llm_response.strip()

        # 2. Remove markdown code block fences if present (e.g., ```json ... ```)
        if json_string_candidate.startswith("```json") and json_string_candidate.endswith("```"):
            json_string_candidate = json_string_candidate[7:-3].strip()

        parsed_response = {} # Initialize parsed_response

        # Attempt 1: Parse directly from detected braces
        first_brace = json_string_candidate.find('{')
        last_brace = json_string_candidate.rfind('}')

        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_string_to_parse = json_string_candidate[first_brace : last_brace + 1]
            try:
                parsed_response = json.loads(json_string_to_parse)
            except json.JSONDecodeError:
                # Fallback if direct parse fails, try to "fix" it
                pass # Continue to Attempt 2 if this fails

        # Attempt 2: If previous parsing failed or was incomplete, try forcing a closing brace
        if not parsed_response and json_string_candidate.startswith("{"):
            print("Attempting to fix incomplete JSON by appending '}'.")
            try:
                parsed_response = json.loads(json_string_candidate + '}')
            except json.JSONDecodeError as e:
                # Still failed, raise a more specific error
                raise ValueError(f"Failed to parse JSON even after attempting to fix. Error: {e}")

        if not parsed_response:
             raise ValueError("Could not find a valid JSON object in LLM response after all attempts.")
        # --- END ROBUST JSON PARSING ---

        return {
            "summary": parsed_response.get("summary", "No summary provided by LLM."),
            "issues": parsed_response.get("issues", []),
            "overall_sentiment": parsed_response.get("overall_sentiment", "N/A")
        }

    except Exception as e:
        print(f"Error during LLM call or parsing summarization: {e}. Raw LLM response: \n{llm_response}")
        fallback_summary_text = f"LLM summarization failed. Found {len(conversations)} conversations.\n"
        fallback_issues = {}
        fallback_sentiments = {'positive': 0, 'negative': 0, 'neutral': 0}
        for conv in conversations:
            # These keys ('issue_category', 'sentiment') are expected due to mapping in bq_utils.py
            if conv.get('issue_category'):
                fallback_issues[conv['issue_category']] = fallback_issues.get(conv['issue_category'], 0) + 1
            if conv.get('sentiment'):
                # Ensure sentiment is lowercased for consistent counting
                fallback_sentiments[conv['sentiment'].lower()] = fallback_sentiments.get(conv['sentiment'].lower(), 0) + 1

        # Determine overall sentiment based on counts
        identified_issues = list(fallback_issues.keys()) if fallback_issues else ["No specific issues identified"]
        overall_sentiment_fallback = "Mixed"
        if fallback_sentiments['positive'] > fallback_sentiments['negative'] and fallback_sentiments['positive'] > fallback_sentiments['neutral']:
            overall_sentiment_fallback = "Positive"
        elif fallback_sentiments['negative'] > fallback_sentiments['positive'] and fallback_sentiments['negative'] > fallback_sentiments['neutral']:
            overall_sentiment_fallback = "Negative"
        elif fallback_sentiments['neutral'] > 0 and fallback_sentiments['positive'] == 0 and fallback_sentiments['negative'] == 0:
            overall_sentiment_fallback = "Neutral"
        elif fallback_sentiments['positive'] == 0 and fallback_sentiments['negative'] == 0 and fallback_sentiments['neutral'] == 0:
            overall_sentiment_fallback = "N/A" # No sentiments found

        return {
            "summary": fallback_summary_text,
            "issues": identified_issues,
            "overall_sentiment": overall_sentiment_fallback
        }

# --- Modified answer_user_query_hybrid with BigQuery filtering and local FAISS search ---
def answer_user_query_hybrid(user_query, faiss_index, all_conversations_indexed, zip_code=None, days_back=None, sentiment=None, issue_category=None):
    """
    Answers a user query using a hybrid approach: BigQuery structured filtering
    followed by FAISS vector search on the *filtered subset* using pre-computed embeddings.
    """
    print(f"\n--- Processing User Query: '{user_query}' with Customer Conversation Agent ---")

    start_date_str = None
    end_date_str = None
    if days_back:
        today = datetime.now()
        start_date = today - timedelta(days=days_back)
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = today.strftime('%Y-%m-%d')

    print("Step 1: Filtering conversations using BigQuery (retrieving raw texts for subset)...")
    # get_conversation_data will now return all fields, mapped to expected keys
    bq_filtered_conversations = get_conversation_data(
        zip_code=zip_code,
        start_date=start_date_str,
        end_date=end_date_str,
        sentiment=sentiment,
        issue_category=issue_category
    )

    if not bq_filtered_conversations:
        return "No relevant conversations found based on your specified structured criteria."

    # Map conversation_id to its full data including embedding from our pre-indexed data
    # This map is crucial for quickly retrieving the full conversation object (including embedding)
    # after filtering by BigQuery.
    indexed_conversations_map = {conv['conversation_id']: conv for conv in all_conversations_indexed if 'conversation_id' in conv}

    conversations_for_faiss_subset = []
    for conv_data_bq in bq_filtered_conversations:
        conv_id = conv_data_bq.get('conversation_id')
        if conv_id in indexed_conversations_map:
            # We found the conversation and its embedding in our pre-indexed data
            # Ensure the embedding is a numpy array for FAISS
            conv_with_embedding = indexed_conversations_map[conv_id]
            if isinstance(conv_with_embedding.get('embedding'), list):
                conv_with_embedding['embedding'] = np.array(conv_with_embedding['embedding']).astype('float32')
            conversations_for_faiss_subset.append(conv_with_embedding)
        else:
            print(f"Warning: Conversation ID {conv_id} from BigQuery filter not found in local FAISS index. Skipping.")
            # This can happen if BigQuery data is newer than the last index build.
            # A re-run of load_or_build_faiss_index would fix it.

    if not conversations_for_faiss_subset:
        return "No conversations with available embeddings found within the BigQuery filtered set."

    # Prepare embeddings for a *temporary* FAISS index (for the filtered subset)
    # This creates a new FAISS index only for the subset of conversations that passed BigQuery filters.
    filtered_embeddings_array = np.array([conv['embedding'] for conv in conversations_for_faiss_subset if conv.get('embedding') is not None]).astype('float32')

    print(f" → Total {len(filtered_embeddings_array)} conversations with embeddings prepared for FAISS subset search.")

    if filtered_embeddings_array.shape[0] == 0:
        return "No conversations with valid embeddings to perform vector search on after filtering."

    print("Step 2: Building temporary FAISS index for filtered subset...")
    d = filtered_embeddings_array.shape[1]
    temp_index = faiss.IndexFlatL2(d) # Using L2 distance (Euclidean) for similarity search
    temp_index.add(filtered_embeddings_array)
    print(f" → Temporary FAISS index built with {temp_index.ntotal} vectors.")

    print("Step 3: Generating embedding for the user query...")
    # query_embedding = EMBEDDING_MODEL.encode([user_query], normalize_embeddings=True)[0].astype('float32')
    query_embedding = np.array(EMBEDDING_MODEL.embed_query(user_query)).astype('float32')
    print(" → Query embedding generated.")

    print("Step 4: Performing FAISS vector similarity search on filtered results...")
    k = min(5, temp_index.ntotal) # Number of top similar conversations to retrieve, max 5 or total available
    if k == 0:
        return "Not enough conversations to perform a meaningful similarity search after filtering."

    D, I = temp_index.search(query_embedding.reshape(1, -1), k) # D are distances, I are indices

    # Map FAISS indices from the temporary index back to the conversations_for_faiss_subset
    top_indices = [int(i) for i in I[0] if i != -1 and i < len(conversations_for_faiss_subset)]

    if not top_indices:
        return "No semantically similar conversations found within the filtered set using FAISS."

    final_top_conversations = [conversations_for_faiss_subset[idx] for idx in top_indices]

    print(f" → Found {len(final_top_conversations)} top similar conversations using FAISS.")

    print("Step 5: Synthesizing answer using LLM...")
    llm_summary_output = summarize_conversations_with_llm(final_top_conversations)

    final_answer = f"Based on your query and relevant conversations:\n\n"
    final_answer += f"**Summary of Top Conversations:** {llm_summary_output['summary']}\n"
    final_answer += f"**Key Issues Identified:** {', '.join(llm_summary_output['issues'])}\n"
    final_answer += f"**Overall Sentiment (of top results):** {llm_summary_output['overall_sentiment']}\n\n"

    return final_answer


# --- GraphRAG Integration (Uses OPENAI_LLM) ---
def query_knowledge_graph_agent(user_query: str) -> str:
    """
    Queries the GraphRAG-built knowledge graph for nuanced brand information,
    trends, product features, and competitive analysis.
    This function acts as the "Topic Agent" or "Trend Agent."
    """
    print(f"\n--- Processing User Query: '{user_query}' with Knowledge Graph Agent ---")

    # Call the actual GraphRAG utility function, passing the OPENAI_LLM
    retrieved_graph_context = query_graphrag(user_query, openai_llm=OPENAI_LLM) # Pass the specific LLM

    if not retrieved_graph_context:
        return "I could not find relevant information in the knowledge graph for your query."

    print(f" → GraphRAG Context Provided to LLM:\n{retrieved_graph_context[:500]}...")

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

    llm_chain = prompt_template | OPENAI_LLM | StrOutputParser() # Use OPENAI_LLM here

    try:
        final_answer = llm_chain.invoke({"user_query": user_query, "context": retrieved_graph_context})
        print(f" → LLM's Nuanced Answer from Knowledge Graph:\n{final_answer}")
        return final_answer
    except Exception as e:
        print(f"Error during LLM call for knowledge graph query: {e}")
        return "Sorry, I could not synthesize an answer from the knowledge graph at this time."


# --- Intent Classification LLM Setup (Uses GROQ_LLM) ---
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

intent_classification_chain = intent_classification_prompt | OPENAI_LLM | StrOutputParser() # Use GROQ_LLM here


# --- Main Chatbot Agent (No Change in logic, only updated imports/initialization) ---
def chatbot_main_agent(user_query: str, faiss_index: faiss.Index, all_conversations_indexed: list) -> str:
    """
    The main entry point for the chatbot. It uses an LLM to determine the user's intent
    and routes the query to either the Customer Conversation Agent (BigQuery+Vector)
    or the Knowledge Graph (Brand/Trend) Agent.
    """
    print(f"\n--- Chatbot Main Agent Processing Query: '{user_query}' ---")

    try:
        print("Step 1: Classifying intent and extracting parameters using LLM...")
        llm_routing_response = intent_classification_chain.invoke({"user_query": user_query})

        print("-------------------------------")
        print("Raw LLM Routing Response:")
        print(llm_routing_response)
        print("-------------------------------")

        json_string = llm_routing_response

        parsed_routing = json.loads(json_string)
        print("This is parsed_routing", parsed_routing)


        intent = parsed_routing.get("intent", "unclear")
        params = parsed_routing.get("parameters", {})

        print(f" → Detected Intent: '{intent}', Parameters: {params}")

        if intent == "customer_conversation":
            print("--- Routing to Customer Conversation Agent ---")
            return answer_user_query_hybrid(user_query, faiss_index, all_conversations_indexed, **params)
        elif intent == "brand_knowledge":
            print("--- Routing to Knowledge Graph (Brand/Trend) Agent ---")
            # The query_knowledge_graph_agent function itself handles which LLM to use internally
            return query_knowledge_graph_agent(user_query)
        else:
            print("--- Intent Unclear ---")
            return "I'm sorry, I couldn't understand your intent. Please rephrase your query, or ask about customer conversations (e.g., 'summarize negative feedback in zip 12345') or brand knowledge (e.g., 'tell me about InnovateTech')."

    except Exception as e:
        print(f"An unexpected error occurred in the main chatbot agent: {e}")
        return "I apologize, but I encountered an error while processing your request."