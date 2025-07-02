import os
from typing import List
from dotenv import load_dotenv

# LangChain Imports for GraphRAG
from langchain_community.document_loaders import PyPDFLoader, TextLoader # Make sure TextLoader is imported
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import ChatOpenAI # Import for type hinting


# Import embedding model and LLM from your models.py
from models import EMBEDDING_MODEL, OPENAI_LLM

# Import Neo4j config from your config.py
from config import BRAND_DOCUMENTS_DIR, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

# --- Global Neo4j and RAG components ---
_graph = None
_cypher_qa_chain = None
_neo4j_vector_store = None
_initialized = False # Flag to ensure single initialization
_llm_for_graph = None # To hold the OpenAI LLM instance

def _initialize_graphrag_components(openai_llm: ChatOpenAI):
    """
    Initializes Neo4jGraph, LLMGraphTransformer, and QA chains.
    Takes the OpenAI LLM instance as an argument.
    """
    global _graph, _cypher_qa_chain, _neo4j_vector_store, _initialized, _llm_for_graph

    if _initialized and _llm_for_graph is openai_llm: # Check if already initialized with the same LLM
        return

    _llm_for_graph = openai_llm # Store the provided LLM instance

    if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
        print("Neo4j connection details are missing. GraphRAG functionality will be disabled.")
        _initialized = True
        return

    try:
        print("Initializing Neo4j Graph connection...")
        _graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD
        )
        _graph.refresh_schema()
        print("Neo4j connection successful. Schema refreshed.")

        # --- Setup Graph Query Chain ---
        print("Setting up Graph Cypher QA Chain...")
        _cypher_qa_chain = GraphCypherQAChain.from_llm(
            cypher_llm=_llm_for_graph, # Use the passed OpenAI LLM
            qa_llm=_llm_for_graph,     # Use the passed OpenAI LLM
            graph=_graph,
            verbose=True,
            validate_cypher=True,
            allow_dangerous_requests=True
        )
        print("Graph Cypher QA Chain initialized.")

        # --- Setup Neo4j Vector Store for Chunk Retrieval ---
        print("Attempting to initialize Neo4j Vector Store for document chunks...")
        try:
            _neo4j_vector_store = Neo4jVector(
                embedding=EMBEDDING_MODEL, # Still using SentenceTransformer for embeddings
                url=NEO4J_URI,
                username=NEO4J_USERNAME,
                password=NEO4J_PASSWORD,
                index_name="document_chunks_embeddings",
                node_label="Chunk",
                text_node_property="text",
                embedding_node_property="embedding",
            )
            print("Neo4j Vector Store for chunks initialized.")
        except Exception as e:
            print(f"Could not initialize Neo4jVector store. Ensure index 'document_chunks_embeddings' exists and 'Chunk' nodes have 'text' and 'embedding' properties. Error: {e}")
            _neo4j_vector_store = None

        _initialized = True
        print("GraphRAG components initialized.")

    except Exception as e:
        print(f"Error initializing GraphRAG components: {e}")
        print("GraphRAG functionality will be unavailable.")
        _initialized = True
        _graph = None
        _cypher_qa_chain = None
        _neo4j_vector_store = None
        _llm_for_graph = None # Reset LLM if initialization fails

def build_knowledge_graph_and_vector_index(openai_llm: ChatOpenAI):
    """
    Loads documents, extracts knowledge graph, stores in Neo4j,
    and creates a vector index for document chunks.
    This function should be called once at startup if the graph needs building/updating.
    Takes the OpenAI LLM instance as an argument.
    """
    _initialize_graphrag_components(openai_llm) # Pass the LLM here

    if not _graph or not _llm_for_graph: # Check if graph and LLM are properly initialized
        print("Cannot build knowledge graph: Neo4j connection or OpenAI LLM not established.")
        return

    print(f"\n--- Building Knowledge Graph from '{BRAND_DOCUMENTS_DIR}' ---")
    documents = []

    # Iterate through files and load them
    for filename in os.listdir(BRAND_DOCUMENTS_DIR):
        filepath = os.path.join(BRAND_DOCUMENTS_DIR, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())
        elif filename.endswith(".txt"):
            # This is where the fix for UnicodeDecodeError goes
            loader = TextLoader(filepath, encoding="utf-8") # <--- THIS IS THE ONLY CHANGE NEEDED HERE
            documents.extend(loader.load())

    if not documents:
        print(f"No documents found in '{BRAND_DOCUMENTS_DIR}'. Skipping knowledge graph build.")
        return

    print(f"Loaded {len(documents)} raw documents.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)
    print(f"Split documents into {len(splits)} chunks.")

    print("Extracting knowledge graph using LLMGraphTransformer...")
    llm_transformer = LLMGraphTransformer(llm=_llm_for_graph) # Use the passed OpenAI LLM
    graph_documents = llm_transformer.convert_to_graph_documents(splits)
    print(f"Extracted {len(graph_documents)} graph documents.")

    print("Adding graph documents to Neo4j...")
    _graph.add_graph_documents(graph_documents)
    print("Knowledge graph construction complete.")

    print("Creating/updating Neo4j vector index for document chunks...")
    try:
        # Corrected Cypher syntax for dropping index if it exists
        # _graph.query("DROP INDEX FOR (c:Chunk) ON (c.embedding) IF EXISTS")
        _graph.query("DROP INDEX document_chunks_embeddings IF EXISTS")
        print("Dropped existing 'Chunk' embedding index (if any).")
    except Exception as e:
        print(f"Could not drop existing index (may not exist or permission issue): {e}")

    global _neo4j_vector_store
    _neo4j_vector_store = Neo4jVector.from_documents(
        splits,
        EMBEDDING_MODEL,
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        index_name="document_chunks_embeddings",
        node_label="Chunk",
        text_node_property="text",
        embedding_node_property="embedding",
    )
    print("Neo4j vector store for chunks built/updated.")
    _graph.refresh_schema()
    print("Knowledge Graph and Vector Index build process finished.")


def query_graphrag(user_query: str, openai_llm: ChatOpenAI) -> str:
    """
    Queries the GraphRAG system (Neo4j knowledge graph and vector store)
    and returns relevant context.
    Takes the OpenAI LLM instance as an argument.
    """
    _initialize_graphrag_components(openai_llm) # Ensure components are initialized with the correct LLM

    if not _graph or not _cypher_qa_chain or not _neo4j_vector_store:
        return "GraphRAG system is not initialized or configured correctly."

    print(f"\n--- Querying GraphRAG for: '{user_query}' ---")

    combined_context = []

    print("Attempting structured query on Knowledge Graph...")
    try:
        # Note: The _cypher_qa_chain was initialized with _llm_for_graph (OpenAI LLM)
        graph_result = _cypher_qa_chain.invoke({"query": user_query})
        graph_answer = graph_result.get('result') or graph_result.get('answer')
        if graph_answer:
            combined_context.append(f"Structured Graph Information: {graph_answer}")
            print(" → Got structured graph context.")
        else:
            print(" → No structured graph context found.")

    except Exception as e:
        print(f"Error during structured graph query: {e}")

    print("Attempting vector search on document chunks...")
    try:
        vector_docs = _neo4j_vector_store.similarity_search(user_query, k=3)
        if vector_docs:
            combined_context.append("Relevant Document Chunks:")
            for doc in vector_docs:
                combined_context.append(doc.page_content)
            print(" → Got vector search context.")
        else:
            print(" → No relevant document chunks found via vector search.")
    except Exception as e:
        print(f"Error during vector search on chunks: {e}")

    if not combined_context:
        print("No context retrieved from GraphRAG.")
        return ""

    return "\n\n".join(combined_context)

# The global initialization call should be removed as it now needs an LLM.
# It will be called explicitly from main.py's build_knowledge_graph_and_vector_index.
# _initialize_graphrag_components() # Remove or comment out this line