import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI # New import for OpenAI
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from typing import List

# Load environment variables (ensure GROQ_API_KEY and OPENAI_API_KEY are set)
load_dotenv()

# Import model names and temperature from config
from config import (
    GROQ_LLM_MODEL_NAME,
    OPENAI_LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    GROQ_API_KEY,
    OPENAI_API_KEY
)

class SentenceTransformerEmbeddingsWrapper(Embeddings):
    """
    A wrapper to make SentenceTransformer models compatible with LangChain's Embeddings interface.
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        print(f"Loaded Sentence Transformer model: {model_name}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a query text."""
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()

class LLMModels:
    _groq_llm = None
    _openai_llm = None
    _embedding_model = None

    @classmethod
    def get_groq_llm(cls):
        if cls._groq_llm is None:
            try:
                if not GROQ_API_KEY:
                    raise ValueError("GROQ_API_KEY environment variable not set.")
                cls._groq_llm = ChatGroq(
                    temperature=LLM_TEMPERATURE,
                    groq_api_key=GROQ_API_KEY,
                    model_name=GROQ_LLM_MODEL_NAME
                )
                print(f"Initialized ChatGroq LLM with model: {GROQ_LLM_MODEL_NAME}")
            except ValueError as e:
                print(f"Error initializing ChatGroq: {e}")
                print("Please set the GROQ_API_KEY environment variable in your .env file.")
                raise
        return cls._groq_llm

    @classmethod
    def get_openai_llm(cls):
        if cls._openai_llm is None:
            try:
                if not OPENAI_API_KEY:
                    raise ValueError("OPENAI_API_KEY environment variable not set.")
                cls._openai_llm = ChatOpenAI(
                    temperature=LLM_TEMPERATURE, # You might want a different temperature for OpenAI
                    openai_api_key=OPENAI_API_KEY,
                    model=OPENAI_LLM_MODEL_NAME # Use 'model' for OpenAI
                )
                print(f"Initialized ChatOpenAI LLM with model: {OPENAI_LLM_MODEL_NAME}")
            except ValueError as e:
                print(f"Error initializing ChatOpenAI: {e}")
                print("Please set the OPENAI_API_KEY environment variable in your .env file.")
                raise
        return cls._openai_llm

    @classmethod
    def get_embedding_model(cls):
        if cls._embedding_model is None:
            cls._embedding_model = SentenceTransformerEmbeddingsWrapper('all-MiniLM-L6-v2')
            print("Embedding model loaded.")
        return cls._embedding_model

# Initialize models globally on import to avoid re-initialization
GROQ_LLM = LLMModels.get_groq_llm()
OPENAI_LLM = LLMModels.get_openai_llm() # New global variable for OpenAI LLM
EMBEDDING_MODEL = LLMModels.get_embedding_model()