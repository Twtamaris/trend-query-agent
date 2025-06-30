# import os
# from typing import List, Dict
# import json
# import re
# from datetime import datetime, timedelta

# # BigQuery client
# from google.cloud import bigquery

# # LangChain components for LLM interactions (for main chatbot and summarization)
# from langchain_groq import ChatGroq # Using ChatGroq as primary LLM for LangChain
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from langchain_core.output_parsers import StrOutputParser # For robust LLM output parsing

# # Sentence Transformers for embeddings in the BigQuery/Vector Search path
# from sentence_transformers import SentenceTransformer
# from scipy.spatial.distance import cosine # For manual cosine similarity with SentenceTransformer embeddings

# from dotenv import load_dotenv

# # --- LlamaIndex Specific Imports for Knowledge Graph ---
# from llama_index.core import SimpleDirectoryReader, StorageContext, load_index_from_storage
# from llama_index.core.indices.knowledge_graph import KnowledgeGraphIndex # Modern KG Index
# from llama_index.core import Settings # For global LlamaIndex settings
# from llama_index.llms.groq import Groq as LlamaIndexGroqLLM # Groq LLM for LlamaIndex
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding # For local embeddings in LlamaIndex

# # Load environment variables (ensure GROQ_API_KEY is set)
# load_dotenv()

# # --- Initialize BigQuery client ---
# try:
#     bq_client = bigquery.Client()
# except Exception as e:
#     print(f"Error initializing BigQuery client: {e}")
#     print("Please ensure you are authenticated to GCP and have the necessary permissions.")
#     print("For local development, set GOOGLE_APPLICATION_CREDENTIALS to your service account key file path.")
#     exit()

# # --- Initialize GROQ LLM for LangChain (for direct RAG calls and general use) ---
# try:
#     groq_api_key = os.environ.get("GROQ_API_KEY")
#     if not groq_api_key:
#         raise ValueError("GROQ_API_KEY environment variable not set.")
    
#     # LangChain's ChatGroq instance
#     llm = ChatGroq(temperature=0.0, groq_api_key=groq_api_key, model_name="llama3-8b-8192")
# except ValueError as e:
#     print(f"Error initializing ChatGroq for LangChain: {e}")
#     print("Please set the GROQ_API_KEY environment variable.")
#     exit()
# except Exception as e:
#     print(f"An unexpected error occurred during ChatGroq (LangChain) initialization: {e}")
#     exit()

# # --- Load Sentence Transformer model (for BigQuery/Vector Search embeddings) ---
# print("Loading Sentence Transformer model 'all-MiniLM-L6-v2' for BigQuery vector search...")
# try:
#     embedding_model_sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
#     print("Sentence Transformer model loaded.\n")
# except Exception as e:
#     print(f"Error loading Sentence Transformer model: {e}")
#     print("Please ensure 'sentence_transformers' is installed correctly.")
#     exit()


# # --- BigQuery Data Retrieval Function ---
# def get_conversation_data_with_embeddings(zip_code=None, start_date=None, end_date=None, sentiment=None, issue_category=None):
#     """
#     Retrieves conversation data including embeddings from BigQuery based on filters.
#     Dates should be in 'YYYY-MM-DD' format.
#     Queries the 'conversations_with_embeddings' table.
#     """
#     project_id = "chatbot-project-464108"
#     dataset_id = "customer_service_data"
#     table_name = "conversations_with_embeddings"

#     query = f"""
#     SELECT
#         conversation_id,
#         conversation_text,
#         sentiment,
#         issue_category,
#         timestamp,
#         zip_code,
#         embedding
#     FROM
#         `{project_id}.{dataset_id}.{table_name}`
#     WHERE
#         1=1
#     """

#     if zip_code:
#         query += f" AND zip_code = '{zip_code}'"
#     if start_date:
#         query += f" AND DATE(timestamp) >= DATE('{start_date}')"
#     if end_date:
#         query += f" AND DATE(timestamp) <= DATE('{end_date}')"
#     if sentiment:
#         query += f" AND LOWER(sentiment) = LOWER('{sentiment}')"
#     if issue_category:
#         query += f" AND LOWER(issue_category) = LOWER('{issue_category}')"

#     print(f"Executing BigQuery query for filtering:\n{query}")
#     try:
#         query_job = bq_client.query(query)
#         results = query_job.result()
        
#         processed_results = []
#         for row in results:
#             row_dict = dict(row)
#             # BigQuery returns ARRAY<FLOAT64> as a tuple, convert to list for consistency
#             if 'embedding' in row_dict and isinstance(row_dict['embedding'], (tuple, list)):
#                 row_dict['embedding'] = list(row_dict['embedding'])
#             processed_results.append(row_dict)
            
#         print(f"Retrieved {len(processed_results)} rows from BigQuery after filtering.")
#         return processed_results
#     except Exception as e:
#         print(f"Error retrieving data from BigQuery: {e}")
#         return []

# # --- LLM-Powered Summarization Function ---
# def summarize_conversations_with_llm(conversations: List[Dict]) -> Dict:
#     """
#     Summarizes conversations, identifies key issues, and overall sentiment using an LLM.
#     """
#     if not conversations:
#         return {
#             "summary": "No conversations found for the given criteria.",
#             "issues": [],
#             "overall_sentiment": "N/A"
#         }

#     all_texts = [conv['conversation_text'] for conv in conversations if conv['conversation_text']]
#     if not all_texts:
#         return {
#             "summary": "No conversation text available.",
#             "issues": [],
#             "overall_sentiment": "N/A"
#         }

#     conversations_for_llm = "\n---\n".join(all_texts[:5]) # Limit to first 5 for prompt size

#     prompt_template = PromptTemplate(
#         template="""You are an expert customer service assistant. Your task is to analyze customer conversations.
#                 Summarize the main points, identify distinct issues mentioned, and determine the overall sentiment.
#                 Provide the output in a JSON format with the following keys:
#                 - 'summary': A concise summary of all conversations.
#                 - 'issues': A list of unique, distinct issues mentioned by customers.
#                 - 'overall_sentiment': The overall sentiment (Positive, Negative, Neutral, or Mixed) across all conversations.
#                 YOUR ENTIRE RESPONSE MUST BE ONLY THE JSON OBJECT, NOTHING ELSE. DO NOT INCLUDE ANY INTRODUCTORY OR CONCLUDING REMARKS, OR MARKDOWN BACKTICKS (```json).
#                 Example JSON output:
#                 {{
#                     "summary": "Customer service issues included internet slowness and billing disputes.",
#                     "issues": ["Internet speed", "Billing error"],
#                     "overall_sentiment": "Negative"
#                 }}
                
#                 Analyze the following customer service conversations:
                
#                 {conversations}""",
#         input_variables=["conversations"]
#     )

#     llm_chain = LLMChain(prompt=prompt_template, llm=llm) # Uses LangChain's ChatGroq instance

#     try:
#         llm_response = llm_chain.run(conversations=conversations_for_llm)
        
#         # Robustly parse the JSON output from the LLM
#         json_match = re.search(r'\{(.*?)\}', llm_response, re.DOTALL)
#         if json_match:
#             json_string = "{" + json_match.group(1) + "}"
#         else:
#             # Fallback if regex doesn't find a clean JSON block
#             print(f"Warning: LLM did not return a clearly parsable JSON block for summarization. Raw: {llm_response}")
#             json_string = llm_response.strip() # Try to parse the whole response as JSON

#         parsed_response = json.loads(json_string)

#         return {
#             "summary": parsed_response.get("summary", "No summary provided by LLM."),
#             "issues": parsed_response.get("issues", []),
#             "overall_sentiment": parsed_response.get("overall_sentiment", "N/A")
#         }

#     except Exception as e:
#         print(f"Error during LLM call or parsing summarization: {e}")
#         # Fallback to simple aggregation if LLM parsing fails
#         fallback_summary_text = f"LLM summarization failed. Found {len(conversations)} conversations.\n"
#         fallback_issues = {}
#         fallback_sentiments = {'positive': 0, 'negative': 0, 'neutral': 0}
#         for conv in conversations:
#             if conv.get('issue_category'):
#                 fallback_issues[conv['issue_category']] = fallback_issues.get(conv['issue_category'], 0) + 1
#             if conv.get('sentiment'):
#                 fallback_sentiments[conv['sentiment'].lower()] = fallback_sentiments.get(conv['sentiment'].lower(), 0) + 1
        
#         # Determine overall sentiment based on counts
#         overall_sentiment_fb = "N/A"
#         if fallback_sentiments:
#             max_sentiment = max(fallback_sentiments, key=fallback_sentiments.get)
#             if fallback_sentiments[max_sentiment] > 0: # Only if there's any sentiment data
#                 overall_sentiment_fb = max_sentiment.capitalize()

#         return {
#             "summary": fallback_summary_text + " (Using fallback summarization)",
#             "issues": list(fallback_issues.keys()),
#             "overall_sentiment": overall_sentiment_fb
#         }


# def answer_user_query_hybrid(user_query: str, zip_code=None, days_back=None, sentiment=None, issue_category=None) -> str:
#     """
#     Answers a user query using a hybrid approach: BigQuery filtering + vector search.
#     """
#     print(f"\n--- Processing User Query: '{user_query}' with Customer Conversation Agent ---")

#     start_date_str = None
#     end_date_str = None
#     if days_back:
#         today = datetime.now()
#         start_date = today - timedelta(days=days_back)
#         start_date_str = start_date.strftime('%Y-%m-%d')
#         end_date_str = today.strftime('%Y-%m-%d')
    
#     print("Step 1: Filtering conversations using BigQuery (and retrieving embeddings)...")
#     bq_filtered_conversations = get_conversation_data_with_embeddings(
#         zip_code=zip_code,
#         start_date=start_date_str,
#         end_date=end_date_str,
#         sentiment=sentiment,
#         issue_category=issue_category
#     )

#     if not bq_filtered_conversations:
#         return "No relevant conversations found based on your specified structured criteria."

#     conversations_with_embeddings = [
#         conv for conv in bq_filtered_conversations 
#         if conv.get('embedding') is not None and conv.get('conversation_text') is not None
#     ]

#     if not conversations_with_embeddings:
#         return "No conversations with valid embeddings found after filtering."
        
#     filtered_embeddings = np.array([conv['embedding'] for conv in conversations_with_embeddings])
    
#     print(f" → Found {len(filtered_embeddings)} conversations after BigQuery filtering.")


#     print("Step 2: Generating embedding for the user query...")
#     # Use SentenceTransformer to encode the query
#     query_embedding = embedding_model_sentence_transformer.encode([user_query], normalize_embeddings=True)[0]
#     print(" → Query embedding generated.")

#     print("Step 3: Performing vector similarity search on filtered results...")
#     similarities = []
#     for conv_emb in filtered_embeddings:
#         try:
#             # Use dot product for cosine similarity with normalized vectors (SentenceTransformer does this by default)
#             similarity_score = np.dot(query_embedding, np.array(conv_emb))
#             similarities.append(similarity_score)
#         except ValueError as e:
#             print(f"Warning: Could not calculate similarity for an embedding due to {e}. Skipping.")
#             similarities.append(-1.0) # Mark as invalid similarity

#     top_n = 5
#     # Filter out -1.0 similarities before sorting
#     valid_similarities_indices = [i for i, sim in enumerate(similarities) if sim != -1.0]
#     if not valid_similarities_indices:
#         return "No valid embeddings to perform semantic search."

#     # Create a temporary list of (similarity, index) for valid similarities
#     temp_sort_list = [(similarities[i], i) for i in valid_similarities_indices]
#     temp_sort_list.sort(key=lambda x: x[0], reverse=True)
    
#     # Get the original indices of the top_n most similar conversations
#     top_indices = [idx for sim_score, idx in temp_sort_list[:top_n]]

#     top_similar_conversations_data = [conversations_with_embeddings[idx] for idx in top_indices]

#     if not top_similar_conversations_data:
#         return "No semantically similar conversations found within the filtered set."

#     print(f" → Found {len(top_similar_conversations_data)} top similar conversations.")

#     print("Step 4: Synthesizing answer using LLM...")
#     llm_summary_output = summarize_conversations_with_llm(top_similar_conversations_data)

#     final_answer = f"Based on your query and relevant conversations:\n\n"
#     final_answer += f"**Summary of Top Conversations:** {llm_summary_output['summary']}\n"
#     final_answer += f"**Key Issues Identified:** {', '.join(llm_summary_output['issues'])}\n"
#     final_answer += f"**Overall Sentiment (of top results):** {llm_summary_output['overall_sentiment']}\n\n"
    
#     return final_answer

# # --- LlamaIndex for Knowledge Graph (replacing Microsoft GraphRAGClient logic) ---

# BRAND_DOCUMENTS_DIR = "data/brand_documents"
# # Directory where LlamaIndex will persist its Knowledge Graph
# LLAMA_INDEX_KG_PERSIST_DIR = "llama_index_kg_index"

# if not os.path.exists(BRAND_DOCUMENTS_DIR):
#     os.makedirs(BRAND_DOCUMENTS_DIR)
#     print(f"Created directory: '{BRAND_DOCUMENTS_DIR}'. Please place your brand documents here for LlamaIndex KG.")
#     # Create dummy files for initial testing
#     with open(os.path.join(BRAND_DOCUMENTS_DIR, "brand_overview.txt"), "w", encoding="utf-8") as f:
#         f.write("InnovateTech is a leading AI solutions provider. Our core mission is to democratize AI. Our flagship product, 'AI-Assistant Pro', focuses on natural language understanding and automated customer support. A significant trend is the push for explainable AI. We aim to integrate more deeply with enterprise CRM systems.")
#     with open(os.path.join(BRAND_DOCUMENTS_DIR, "product_roadmap.txt"), "w", encoding="utf-8") as f:
#         f.write("The roadmap for AI-Assistant Pro includes enhancements for real-time sentiment analysis and predictive issue detection. We are exploring partnerships with major cloud providers. The market shows a strong trend towards customized AI agents for specific industries. Our competitor, 'Global AI Solutions', recently launched a similar product, but ours offers superior scalability.")
#     with open(os.path.join(BRAND_DOCUMENTS_DIR, "ethical_ai_policy.txt"), "w", encoding="utf-8") as f:
#         f.write("InnovateTech is committed to ethical AI development, ensuring fairness, transparency, and accountability in all our solutions. We adhere to strict data privacy standards and promote responsible AI practices across the industry.")
#     print("Created dummy brand documents for demonstration purposes.")

# if not os.path.exists(LLAMA_INDEX_KG_PERSIST_DIR):
#     os.makedirs(LLAMA_INDEX_KG_PERSIST_DIR)
#     print(f"Created LlamaIndex KG persistence directory: '{LLAMA_INDEX_KG_PERSIST_DIR}'.")


# # --- Global LlamaIndex components to be initialized once ---
# llama_index_kg_index = None
# llama_index_kg_query_engine = None

# def initialize_llama_index_components():
#     """
#     Initializes LlamaIndex's global settings (LLM, Embeddings) and loads the KG index.
#     This should be called ONCE at application startup.
#     """
#     global llama_index_kg_index, llama_index_kg_query_engine

#     if llama_index_kg_index and llama_index_kg_query_engine:
#         print("LlamaIndex KG components already initialized.")
#         return

#     print("Initializing LlamaIndex KG components...")
#     try:
#         # Set global LlamaIndex LLM to Groq
#         Settings.llm = LlamaIndexGroqLLM(
#             api_key=os.environ.get("GROQ_API_KEY"),
#             model_name="llama3-8b-8192",
#             temperature=0.0
#         )
#         print("LlamaIndex LLM (Groq) configured.")

#         # Set global LlamaIndex Embedding Model
#         Settings.embed_model = HuggingFaceEmbedding(
#             model_name="sentence-transformers/all-MiniLM-L6-v2"
#         )
#         print("LlamaIndex Embedding Model (HuggingFaceEmbedding) configured.")

#         # Load the persisted Knowledge Graph Index
#         if os.path.exists(LLAMA_INDEX_KG_PERSIST_DIR) and os.listdir(LLAMA_INDEX_KG_PERSIST_DIR):
#             print(f"Loading Knowledge Graph Index from '{LLAMA_INDEX_KG_PERSIST_DIR}'...")
#             storage_context = StorageContext.from_defaults(persist_dir=LLAMA_INDEX_KG_PERSIST_DIR)
#             llama_index_kg_index = load_index_from_storage(storage_context)
#             llama_index_kg_query_engine = llama_index_kg_index.as_query_engine(
#                 # You can specify response_mode like "tree_summarize", "compact", "refine", etc.
#                 # knowledge_graph_query_engine options can be tuned here
#                 # Example: service_context=llama_index_service_context (if you created one)
#             )
#             print("Knowledge Graph Index loaded and query engine created.")
#         else:
#             print(f"No existing Knowledge Graph Index found at '{LLAMA_INDEX_KG_PERSIST_DIR}'. Please run build_graph_index first.")
#             llama_index_kg_index = None
#             llama_index_kg_query_engine = None

#     except Exception as e:
#         print(f"Error initializing LlamaIndex KG components: {e}")
#         print("Ensure GROQ_API_KEY is set and LlamaIndex/HuggingFace libraries are correctly installed.")
#         llama_index_kg_index = None
#         llama_index_kg_query_engine = None


# def build_graph_index_llama_index(data_dir: str, persist_dir: str):
#     """
#     Ingests brand documents and builds a LlamaIndex Knowledge Graph, persisting it to disk.
#     This function should be run once or whenever documents change.
#     """
#     print(f"\n--- Building LlamaIndex Knowledge Graph from '{data_dir}' ---")
#     print("This may take some time depending on document size and LLM calls for KG extraction.")

#     try:
#         # Load documents from the specified directory
#         documents = SimpleDirectoryReader(data_dir).load_data()
#         if not documents:
#             print(f"No documents found in '{data_dir}'. Skipping KG index building.")
#             return

#         print(f"Loaded {len(documents)} documents. Building Knowledge Graph Index...")

#         # Create StorageContext to persist the index
#         storage_context = StorageContext.from_defaults(persist_dir=persist_dir)

#         # Build the Knowledge Graph Index
#         # Settings.llm and Settings.embed_model must be configured globally before this call
#         index = KnowledgeGraphIndex.from_documents(
#             documents,
#             storage_context=storage_context,
#             # service_context=llama_index_service_context, # If you have a custom service context
#             show_progress=True,
#             # Other KG-specific parameters can be added here, e.g.,
#             # kg_triple_extract_llm=Settings.llm, # Use Groq for triple extraction
#             # max_triplets_per_chunk=3,
#         )
        
#         # Persist the index
#         index.storage_context.persist(persist_dir)
#         print(f"Knowledge Graph Index built and saved to '{persist_dir}'.")
#     except Exception as e:
#         print(f"Error building LlamaIndex Knowledge Graph: {e}")
#         print("Ensure GROQ_API_KEY is set and LlamaIndex/HuggingFace libraries are correctly installed.")

# def query_knowledge_graph_agent(user_query: str) -> str:
#     """
#     Queries the LlamaIndex-built Knowledge Graph for nuanced brand information.
#     This function acts as the "Topic Agent" or "Trend Agent."
#     """
#     print(f"\n--- Processing User Query: '{user_query}' with Knowledge Graph Agent (LlamaIndex) ---")

#     global llama_index_kg_index, llama_index_kg_query_engine
#     if not llama_index_kg_index or not llama_index_kg_query_engine:
#         print("LlamaIndex KG components not initialized. Attempting to re-initialize...")
#         initialize_llama_index_components()
#         if not llama_index_kg_index or not llama_index_kg_query_engine:
#             return "LlamaIndex Knowledge Graph is not set up correctly. Cannot answer knowledge graph queries."

#     try:
#         print(f" → Asking LlamaIndex KG to answer: '{user_query}'")
#         response = llama_index_kg_query_engine.query(user_query)
        
#         final_answer = response.response
#         print(f" → Knowledge Graph's Nuanced Answer:\n{final_answer}")
#         return final_answer

#     except Exception as e:
#         print(f"Error querying LlamaIndex Knowledge Graph: {e}")
#         return "Sorry, I encountered an error while retrieving information from the LlamaIndex knowledge graph."


# # --- Intent Classification LLM Setup (uses LangChain's ChatGroq) ---
# intent_classification_prompt_template = PromptTemplate(
#     template="""You are a helpful assistant for routing user queries.
#             Your task is to analyze the user's query and determine their intent.
#             You should output a JSON object with two main keys:
#             - 'intent': This should be one of 'customer_conversation', 'brand_knowledge', or 'unclear'.
#             - 'parameters': This should be a JSON object containing any relevant extracted parameters.

#             For 'customer_conversation' intent, extract the following parameters if present:
#             - 'zip_code': (string, e.g., "90210")
#             - 'days_back': (integer, e.g., 7 for 'last 7 days'. Interpret 'last week' as 7, 'last month' as 30, 'last quarter' as 90, 'last year' as 365.)
#             - 'sentiment': (string, e.g., "positive", "negative", "neutral")
#             - 'issue_category': (string, e.g., "billing", "internet speed", "service outage", "account management")

#             For 'brand_knowledge' intent, no specific parameters are required, but you can extract general topics if they are clear.

#             Example Outputs:
#             1. User: "Summarize conversations for the last 7 days in zip code 12345 with negative sentiment."
#                 {{
#                    "intent": "customer_conversation",
#                    "parameters": {{
#                       "zip_code": "12345",
#                       "days_back": 7,
#                       "sentiment": "negative"
#                    }}
#                 }}
#             2. User: "What are the top trends for InnovateTech?"
#                 {{
#                    "intent": "brand_knowledge",
#                    "parameters": {{}}
#                 }}
#             3. User: "Tell me about their new AI features."
#                 {{
#                    "intent": "brand_knowledge",
#                    "parameters": {{}}
#                 }}
#             4. User: "How are customers feeling about internet speed issues in 90210?"
#                 {{
#                    "intent": "customer_conversation",
#                    "parameters": {{
#                       "zip_code": "90210",
#                       "issue_category": "internet speed"
#                    }}
#                 }}
#             5. User: "Hello, how are you?"
#                 {{
#                    "intent": "unclear",
#                    "parameters": {{}}
#                 }}
#             6. User: "Show me negative sentiment issues from last month."
#                 {{
#                    "intent": "customer_conversation",
#                    "parameters": {{
#                       "days_back": 30,
#                       "sentiment": "negative"
#                    }}
#                 }}
            
#             YOUR ENTIRE RESPONSE MUST BE ONLY THE JSON OBJECT, NOTHING ELSE. DO NOT INCLUDE ANY INTRODUCTORY OR CONCLUDING REMARKS, OR MARKDOWN BACKTICKS (```json).
            
#             User query: {user_query}""",
#     input_variables=["user_query"]
# )

# intent_classification_chain = LLMChain(prompt=intent_classification_prompt_template, llm=llm) # Uses LangChain's ChatGroq instance

# def chatbot_main_agent(user_query: str) -> str:
#     """
#     The main entry point for the chatbot. It uses an LLM to determine the user's intent
#     and routes the query to either the Customer Conversation Agent (BigQuery+Vector)
#     or the Knowledge Graph (Brand/Trend) Agent.
#     """
#     print(f"\n--- Chatbot Main Agent Processing Query: '{user_query}' ---")

#     try:
#         # Step 1: Use LLM for Intent Classification and Parameter Extraction
#         print("Step 1: Classifying intent and extracting parameters using LLM...")
#         llm_routing_response = intent_classification_chain.run(user_query=user_query)
        
#         json_string = ""
#         json_match = re.search(r'\{(.*?)\}', llm_routing_response, re.DOTALL)
#         if json_match:
#             json_string = "{" + json_match.group(1) + "}"
#         else:
#             print(f"Warning: LLM did not return valid JSON for intent classification. Raw: {llm_routing_response}")
#             # Fallback: try to parse the entire response if no curly braces were found
#             json_string = llm_routing_response.strip()

#         parsed_routing = {"intent": "unclear", "parameters": {}} # Default fallback
#         try:
#             parsed_routing = json.loads(json_string)
#         except json.JSONDecodeError as e:
#             print(f"Error decoding JSON from LLM intent: {e}. Raw: {json_string}")
#             # parsed_routing remains the default fallback

#         intent = parsed_routing.get("intent", "unclear")
#         params = parsed_routing.get("parameters", {})

#         print(f" → Detected Intent: '{intent}', Parameters: {params}")

#         # Step 2: Route based on detected intent
#         if intent == "customer_conversation":
#             print("--- Routing to Customer Conversation Agent ---")
#             return answer_user_query_hybrid(user_query, **params)
#         elif intent == "brand_knowledge":
#             print("--- Routing to Knowledge Graph (Brand/Trend) Agent ---")
#             return query_knowledge_graph_agent(user_query)
#         else:
#             print("--- Intent unclear, providing a general response ---")
#             return "I'm sorry, I couldn't understand your request clearly. Could you please rephrase it or ask about customer conversations (e.g., summary, issues, sentiment, zip code) or about InnovateTech's brand, products, or trends?"

#     except Exception as e:
#         print(f"An error occurred during intent classification: {e}")
#         return "I apologize, but I encountered an internal error. Please try again later."


# # --- Main execution flow ---
# if __name__ == "__main__":
#     # Ensure LlamaIndex global settings are set before building or loading KG
#     # These settings are used when the KG is built and queried.
#     print("\n--- Initializing LlamaIndex global settings (LLM and Embeddings) ---")
    
#     # Initialize LlamaIndex LLM and Embedding models globally
#     # This is required for KnowledgeGraphIndex.from_documents and .as_query_engine
#     Settings.llm = LlamaIndexGroqLLM(
#         api_key=os.environ.get("GROQ_API_KEY"),
#         model_name="llama3-8b-8192",
#         temperature=0.0
#     )
#     Settings.embed_model = HuggingFaceEmbedding(
#         model_name="sentence-transformers/all-MiniLM-L6-v2"
#     )
#     print("LlamaIndex global LLM and Embedding models configured.")

#     # First, run the LlamaIndex KG pipeline to build and persist the graph.
#     # This is typically a one-time setup or run when documents change.
#     print("--- Running initial LlamaIndex KG pipeline (if not already built) ---")
#     # You might want to add a check here to only run if KG_PERSIST_DIR is empty
#     # or if a force_rebuild flag is set.
#     build_graph_index_llama_index(BRAND_DOCUMENTS_DIR, LLAMA_INDEX_KG_PERSIST_DIR)
    
#     # Initialize LlamaIndex query components after the pipeline has potentially run
#     initialize_llama_index_components()

#     print("\n--- Starting Chatbot Main Agent Demonstrations ---")

#     # --- Test Knowledge Graph Agent Queries (routed by LLM) ---
#     print("\n--- Testing Knowledge Graph (Brand/Trend) Agent (via LlamaIndex) ---")
    
#     query_graph_1 = "What are the top emerging trends for InnovateTech in the AI space?"
#     print(chatbot_main_agent(query_graph_1))

#     query_graph_2 = "Can you describe the key features of AI-Assistant Pro and its future roadmap?"
#     print(chatbot_main_agent(query_graph_2))

#     query_graph_3 = "How does InnovateTech's product compare to Global AI Solutions?"
#     print(chatbot_main_agent(query_graph_3))

#     query_graph_4 = "What is InnovateTech's core mission and vision?"
#     print(chatbot_main_agent(query_graph_4))

#     query_graph_5 = "Are there any new directions in AI solutions that InnovateTech is considering?"
#     print(chatbot_main_agent(query_graph_5))

#     # --- Test BigQuery + Vector Search Agent Queries (routed by LLM) ---
#     print("\n--- Testing Customer Conversation (BigQuery + Vector) Agent ---")

#     # NOTE: These queries will only work if you have a BigQuery table
#     # 'chatbot-project-464108.customer_service_data.conversations_with_embeddings'
#     # with relevant data. If not, they will return "No relevant conversations found..."
#     user_query_bigquery_1 = "Summarize conversations for the last 10 days in zip code 92122."
#     print(chatbot_main_agent(user_query_bigquery_1))

#     user_query_bigquery_2 = "How are customers feeling about internet connectivity issues in 92122 over the past 2 weeks?"
#     print(chatbot_main_agent(user_query_bigquery_2))

#     user_query_bigquery_3 = "Show me negative sentiment issues from last month."
#     print(chatbot_main_agent(user_query_bigquery_3))

#     # --- Test Unclear / Default Queries ---
#     print("\n--- Testing Unclear / Default Handling ---")
#     query_unclear_1 = "Hello chatbot, how are you today?"
#     print(chatbot_main_agent(query_unclear_1))

#     query_unclear_2 = "Tell me a joke."
#     print(chatbot_main_agent(query_unclear_2))

#     print("\n--- Chatbot Main Agent Demonstration Completed ---")


import os
from google.cloud import bigquery
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import re

# LangChain components for GROQ and prompting
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from dotenv import load_dotenv

# --- GraphRAG Specific Imports ---
# These are actual imports you'd use for GraphRAG
# from graphrag.index.lsa import run_lsa_graph_pipeline # For building the graph
# from graphrag.query.llm import LLM # GraphRAG's internal LLM abstraction
# from graphrag.query.structured_retriever import StructuredRetriever # For querying
# from graphrag.query.rag_qa.query_qa import QueryQA # For RAG-based answering
# from graphrag.config import GraphRAGConfig # To configure GraphRAG
# from graphrag.llm import load_llm_from_config # Utility to load LLM in GraphRAG format
# from graphrag.embed import load_embedder_from_config # Utility to load embedder in GraphRAG format
# from graphrag import GraphRAGClient


# Load environment variables (ensure GROQ_API_KEY is set)
load_dotenv()

# --- Initialize BigQuery client ---
client = bigquery.Client()

# --- Initialize GROQ LLM (for direct RAG calls and general use) ---
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

# --- BigQuery Data Retrieval Function ---
def get_conversation_data_with_embeddings(zip_code=None, start_date=None, end_date=None, sentiment=None, issue_category=None):
    """
    Retrieves conversation data including embeddings from BigQuery based on filters.
    Dates should be in 'YYYY-MM-DD' format.
    Queries the 'conversations_with_embeddings' table.
    """
    project_id = "chatbot-project-464108"
    dataset_id = "customer_service_data"
    table_name = "conversations_with_embeddings"

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
            similarities.append(-1.0)

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

# --- GraphRAG Integration: Actual Setup and Querying ---

BRAND_DOCUMENTS_DIR = "data/brand_documents"
# Directory where GraphRAG will store its intermediate and final graph data
GRAPHRAG_OUTPUT_DIR = "graphrag_output"

if not os.path.exists(BRAND_DOCUMENTS_DIR):
    os.makedirs(BRAND_DOCUMENTS_DIR)
    print(f"Created directory: '{BRAND_DOCUMENTS_DIR}'. Please place your brand documents here for GraphRAG.")
    # Create dummy files for initial testing
    with open(os.path.join(BRAND_DOCUMENTS_DIR, "brand_overview.txt"), "w") as f:
        f.write("InnovateTech is a leading AI solutions provider. Our core mission is to democratize AI. Our flagship product, 'AI-Assistant Pro', focuses on natural language understanding and automated customer support. A significant trend is the push for explainable AI. We aim to integrate more deeply with enterprise CRM systems.")
    with open(os.path.join(BRAND_DOCUMENTS_DIR, "product_roadmap.txt"), "w") as f:
        f.write("The roadmap for AI-Assistant Pro includes enhancements for real-time sentiment analysis and predictive issue detection. We are exploring partnerships with major cloud providers. The market shows a strong trend towards customized AI agents for specific industries. Our competitor, 'Global AI Solutions', recently launched a similar product, but ours offers superior scalability.")
    with open(os.path.join(BRAND_DOCUMENTS_DIR, "ethical_ai_policy.txt"), "w") as f:
        f.write("InnovateTech is committed to ethical AI development, ensuring fairness, transparency, and accountability in all our solutions. We adhere to strict data privacy standards and promote responsible AI practices across the industry.")
    print("Created dummy brand documents for demonstration purposes.")

if not os.path.exists(GRAPHRAG_OUTPUT_DIR):
    os.makedirs(GRAPHRAG_OUTPUT_DIR)
    print(f"Created GraphRAG output directory: '{GRAPHRAG_OUTPUT_DIR}'.")


# --- Global GraphRAG components to be initialized once ---
graphrag_llm_instance = None
graphrag_embedder_instance = None
graphrag_retriever = None
graphrag_query_qa = None

def initialize_graphrag_components():
    """
    Initializes GraphRAG's LLM, Embedder, and Retriever.
    This should be called ONCE at application startup.
    """
    global graphrag_llm_instance, graphrag_embedder_instance, graphrag_retriever, graphrag_query_qa

    if graphrag_llm_instance and graphrag_embedder_instance and graphrag_retriever and graphrag_query_qa:
        print("GraphRAG components already initialized.")
        return

    print("Initializing GraphRAG components...")
    try:
        # Define the LLM configuration for GraphRAG's internal use
        # This will use your GROQ_API_KEY
        llm_config = {
            "api_key": os.environ.get("GROQ_API_KEY"),
            "model_name": "llama3-8b-8192",
            "temperature": 0.0,
            "type": "openai_chat", # GraphRAG uses 'openai_chat' type for OpenAI-compatible APIs like GROQ
            "api_base": "[https://api.groq.com/openai/v1](https://api.groq.com/openai/v1)" # Specify GROQ's API base URL
        }

        # Define the Embedder configuration for GraphRAG
        # Using the same model as your existing setup
        embedder_config = {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_dim": 384, # Dimensions for all-MiniLM-L6-v2
            "community_level": 2 # Adjust as needed for graph community detection
        }

        # Load LLM and Embedder for GraphRAG
        graphrag_llm_instance = load_llm_from_config(llm_config)
        graphrag_embedder_instance = load_embedder_from_config(embedder_config)

        # Initialize the GraphRAG Retriever and QueryQA components
        # This assumes the graph data (nodes.csv, relationships.csv, etc.) exists in GRAPHRAG_OUTPUT_DIR
        graphrag_retriever = StructuredRetriever(
            input_dir=GRAPHRAG_OUTPUT_DIR,
            llm=graphrag_llm_instance,
            embedding_model=graphrag_embedder_instance,
            response_type="multiple_facts_per_entity" # Or "nodes_and_relationships"
        )
        
        # QueryQA uses the retriever and LLM to answer questions
        graphrag_query_qa = QueryQA(
            llm=graphrag_llm_instance,
            retriever=graphrag_retriever,
            # Adjust these prompts if you want to customize the final answer generation
            # query_rag_prompt="...",
            # map_llm_prompt="...",
            # reduce_llm_prompt="..."
        )
        print("GraphRAG components initialized successfully.")
    except Exception as e:
        print(f"Error initializing GraphRAG components: {e}")
        print("Ensure you have set GROQ_API_KEY and GraphRAG libraries are correctly installed.")
        # Optionally re-raise or exit if GraphRAG is critical
        graphrag_llm_instance = None
        graphrag_embedder_instance = None
        graphrag_retriever = None
        graphrag_query_qa = None

def run_graphrag_pipeline():
    """
    Runs the GraphRAG pipeline to build the knowledge graph from source documents.
    This should be run AS A SEPARATE SCRIPT/PROCESS when documents change,
    not on every user query.
    """
    print(f"\n--- Running GraphRAG Pipeline to Build Knowledge Graph from '{BRAND_DOCUMENTS_DIR}' ---")
    print("This may take some time depending on the number and size of documents and LLM calls.")

    # Define GraphRAG configuration for the pipeline run
    config = GraphRAGConfig(
        root_dir=os.getcwd(), # Current working directory
        input_dir=BRAND_DOCUMENTS_DIR,
        output_dir=GRAPHRAG_OUTPUT_DIR,
        llm={
            "api_key": os.environ.get("GROQ_API_KEY"),
            "model_name": "llama3-8b-8192",
            "temperature": 0.0,
            "type": "openai_chat",
            "api_base": "[https://api.groq.com/openai/v1](https://api.groq.com/openai/v1)"
        },
        embeddings={
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_dim": 384,
            "community_level": 2
        },
        # You can add more detailed configuration here for community detection,
        # entity extraction, record linking, etc.
        # For simplicity, we'll use defaults for many aspects.
    )

    try:
        # Run the LSA (Local Semantic Analysis) graph pipeline
        # This will process documents, extract entities/relationships,
        # and build the graph. The results are saved to GRAPHRAG_OUTPUT_DIR.
        pipeline_result = run_lsa_graph_pipeline(config)
        
        if pipeline_result is None:
            raise ValueError("GraphRAG pipeline returned None. Check logs for errors.")

        # Check if expected output files exist (e.g., nodes.csv, relationships.csv)
        if os.path.exists(os.path.join(GRAPHRAG_OUTPUT_DIR, "nodes.csv")) and \
           os.path.exists(os.path.join(GRAPHRAG_OUTPUT_DIR, "relationships.csv")):
            print("GraphRAG pipeline completed successfully. Graph data created.")
            # After building, re-initialize components if they were already loaded,
            # so they pick up the new graph data.
            initialize_graphrag_components()
        else:
            print("GraphRAG pipeline ran, but expected output files were not found. Check GraphRAG logs.")

    except Exception as e:
        print(f"Error running GraphRAG pipeline: {e}")
        print("Please ensure your GraphRAG setup is correct and API keys are valid.")
        print("Refer to GraphRAG documentation for detailed setup instructions.")

def query_knowledge_graph_agent(user_query: str) -> str:
    """
    Queries the GraphRAG-built knowledge graph for nuanced brand information.
    This function acts as the "Topic Agent" or "Trend Agent."
    Assumes `graphrag_query_qa` is already initialized.
    """
    print(f"\n--- Processing User Query: '{user_query}' with Knowledge Graph Agent ---")

    if not graphrag_query_qa:
        print("GraphRAG components not initialized. Attempting to initialize...")
        initialize_graphrag_components()
        if not graphrag_query_qa:
            return "GraphRAG is not set up correctly. Cannot answer knowledge graph queries."

    try:
        # Use GraphRAG's QueryQA to answer the question using the built graph
        print(f" → Asking GraphRAG to answer: '{user_query}'")
        qa_response = graphrag_query_qa.query(user_query)
        
        final_answer = qa_response.response
        print(f" → GraphRAG's Nuanced Answer:\n{final_answer}")
        return final_answer

    except Exception as e:
        print(f"Error querying GraphRAG: {e}")
        return "Sorry, I encountered an error while retrieving information from the knowledge graph."


# --- Intent Classification LLM Setup ---
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
            - 'days_back': (integer, e.g., 7 for 'last 7 days'. Interpret 'last week' as 7, 'last month' as 30, 'last quarter' as 90, 'last year' as 365.)
            - 'sentiment': (string, e.g., "positive", "negative", "neutral")
            - 'issue_category': (string, e.g., "billing", "internet speed", "service outage", "account management")

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
            6. User: "Show me negative sentiment issues from last month."
               {{
                 "intent": "customer_conversation",
                 "parameters": {{
                   "days_back": 30,
                   "sentiment": "negative"
                 }}
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
        
        json_match = re.search(r'\{(.*?)\}', llm_routing_response, re.DOTALL)
        if json_match:
            json_string = "{" + json_match.group(1) + "}"
        else:
            print(f"Warning: LLM did not return valid JSON for intent classification. Raw: {llm_routing_response}")
            parsed_routing = {"intent": "unclear", "parameters": {}} # Fallback

        try:
            parsed_routing = json.loads(json_string)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from LLM intent: {e}. Raw: {json_string}")
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
    # First, run the GraphRAG pipeline to build the graph.
    # This is typically a one-time setup or run when documents change.
    print("--- Running initial GraphRAG pipeline (if not already built) ---")
    # You might want to add a check here to only run if graph_output_dir is empty
    # or if a force_rebuild flag is set.
    run_graphrag_pipeline()
    
    # Initialize GraphRAG query components after the pipeline has potentially run
    initialize_graphrag_components()

    print("\n--- Starting Chatbot Main Agent Demonstrations ---")

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

    user_query_bigquery_1 = "Summarize conversations for the last 10 days in zip code 92122."
    print(chatbot_main_agent(user_query_bigquery_1))

    user_query_bigquery_2 = "How are customers feeling about internet connectivity issues in 92122 over the past 2 weeks?"
    print(chatbot_main_agent(user_query_bigquery_2))

    user_query_bigquery_3 = "Show me negative sentiment issues from last month."
    print(chatbot_main_agent(user_query_bigquery_3))

    # --- Test Unclear / Default Queries ---
    print("\n--- Testing Unclear / Default Handling ---")
    query_unclear_1 = "Hello chatbot, how are you today?"
    print(chatbot_main_agent(query_unclear_1))

    query_unclear_2 = "Tell me a joke."
    print(chatbot_main_agent(query_unclear_2))

    print("\n--- Chatbot Main Agent Demonstration Completed ---")