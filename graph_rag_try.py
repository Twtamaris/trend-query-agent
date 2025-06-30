import os
from typing import List, Dict

from graphrag import GraphRAGClient
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Groq
from langchain.chains import RetrievalQA

# Environment variables for API keys
os.environ['GRAPHRAG_API_KEY'] = 'YOUR_GRAPHRAG_API_KEY'
os.environ['GROQ_API_KEY'] = 'YOUR_GROQ_API_KEY'

# Initialize GraphRAG client
def init_graphrag() -> GraphRAGClient:
    api_key = os.getenv('GRAPHRAG_API_KEY')
    return GraphRAGClient(api_key=api_key)

# Retrieve unstructured docs from GraphRAG by topic or trend
def fetch_documents(gr_client: GraphRAGClient, topic: str, limit: int = 5) -> List[Dict[str, str]]:
    # Use GraphRAG's search API to get relevant documents
    docs = gr_client.search(topic=topic, top_k=limit)
    return docs  # Expect list of {'id': str, 'title': str, 'content': str}

# Build vector store over GraphRAG documents
def build_vector_store(docs: List[Dict[str, str]], persist_path: str = 'faiss_index') -> FAISS:
    texts = [doc['content'] for doc in docs]
    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_texts(texts=texts, embedding=embeddings)
    vectordb.save_local(persist_path)
    return vectordb

# Load existing vector store
def load_vector_store(persist_path: str = 'faiss_index') -> FAISS:
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(persist_path, embeddings)

# Create a QA chain using GROQ LLM and FAISS vector store
def create_qa_chain(vectordb: FAISS) -> RetrievalQA:
    llm = Groq(api_key=os.getenv('GROQ_API_KEY'), model="groq-llm")
    retriever = vectordb.as_retriever()
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Main Graphrag-only agent function
def graphrag_agent(query: str, topic: str, rebuild_index: bool = False) -> str:
    # 1. Initialize GraphRAG and fetch docs
    gr_client = init_graphrag()
    docs = fetch_documents(gr_client, topic=topic)

    # 2. Build or load vector store
    persist_path = f"faiss_index_{topic.replace(' ', '_')}"
    if rebuild_index or not os.path.exists(persist_path):
        vectordb = build_vector_store(docs, persist_path)
    else:
        vectordb = load_vector_store(persist_path)

    # 3. Run RetrievalQA with GROQ
    qa_chain = create_qa_chain(vectordb)
    answer = qa_chain.run(query)

    return answer

# Example usage
def main():
    topic = ""
    user_query = "What are the top trends about this brand in the latest documents?"
    # Set rebuild_index=True if documents have changed
    response = graphrag_agent(user_query, topic, rebuild_index=True)
    print("GROQ Answer:\n", response)

if __name__ == "__main__":
    main()
