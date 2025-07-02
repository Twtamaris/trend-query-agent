import os
import json
import faiss
from faiss_utils import load_or_build_faiss_index
from llm_agents import chatbot_main_agent
from config import BRAND_DOCUMENTS_DIR
from graphrag_utils import build_knowledge_graph_and_vector_index # Import the new build function
from models import OPENAI_LLM # Import the specific LLM for GraphRAG

def run_chatbot():
    """
    Main function to run the chatbot application.
    Loads/builds the FAISS index, builds the GraphRAG knowledge graph,
    and then enters a loop for user queries.
    """
    print("Initializing Chatbot System...")

    # # --- Ensure brand documents directory exists and create dummy files ---
    # if not os.path.exists(BRAND_DOCUMENTS_DIR):
    #     os.makedirs(BRAND_DOCUMENTS_DIR)
    #     print(f"Created directory: '{BRAND_DOCUMENTS_DIR}'. Please place your brand documents here for GraphRAG.")

    # dummy_brand_overview_path = os.path.join(BRAND_DOCUMENTS_DIR, "brand_overview.txt")
    # dummy_product_roadmap_path = os.path.join(BRAND_DOCUMENTS_DIR, "product_roadmap.txt")

    # if not os.path.exists(dummy_brand_overview_path):
    #     with open(dummy_brand_overview_path, "w") as f:
    #         f.write("InnovateTech is a leading AI solutions provider. Our core mission is to democratize AI. Our flagship product, 'AI-Assistant Pro', focuses on natural language understanding and automated customer support. A significant trend is the push for explainable AI. We aim to integrate more deeply with enterprise CRM systems.")
    #     print(f"Created dummy file: {dummy_brand_overview_path}")
    # if not os.path.exists(dummy_product_roadmap_path):
    #     with open(dummy_product_roadmap_path, "w") as f:
    #         f.write("The roadmap for AI-Assistant Pro includes enhancements for real-time sentiment analysis and predictive issue detection. We are exploring partnerships with major cloud providers. The market shows a strong trend towards customized AI agents for specific industries. Our competitor, 'Global AI Solutions', recently launched a similar product, but ours offers superior scalability.")
    #     print(f"Created dummy file: {dummy_product_roadmap_path}")
    # print("Dummy brand documents ensured for demonstration purposes.")
    # # --- End of dummy document creation ---

    # --- Initialize GraphRAG Knowledge Graph ---
    print("\n--- Initializing GraphRAG Knowledge Graph ---")
    # Pass the OPENAI_LLM to the build function
    build_knowledge_graph_and_vector_index(openai_llm=OPENAI_LLM)
    print("--- GraphRAG Knowledge Graph Initialization Complete ---")


    # --- Initialize FAISS Index for Customer Conversations ---
    print("\n--- Initializing FAISS Index for Customer Conversations ---")
    faiss_index, all_conversations_indexed = load_or_build_faiss_index()

    if faiss_index is None:
        print("FAISS index could not be loaded or built. Chatbot will not be able to process customer conversation queries.")
        print("Please ensure your BigQuery table has data and check BigQuery permissions.")
    print("--- FAISS Index Initialization Complete ---")


    print("\nChatbot Ready! Type 'exit' to quit.")

    while True:
        user_input = input("\nEnter your query: ")
        if user_input.lower() == 'exit':
            print("Exiting Chatbot. Goodbye!")
            break

        # Pass the loaded FAISS index and indexed conversation data to the main agent
        # The chatbot_main_agent internally decides which LLM to use based on intent
        response = chatbot_main_agent(user_input, faiss_index, all_conversations_indexed)
        print(f"\nChatbot Response:\n{response}")

if __name__ == "__main__":
    run_chatbot()