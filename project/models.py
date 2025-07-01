import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from config import GROQ_MODEL_NAME, LLM_TEMPERATURE

# Load environment variables (ensure GROQ_API_KEY is set)
load_dotenv()

class LLMModels:
    _llm = None
    _embedding_model = None

    @classmethod
    def get_llm(cls):
        if cls._llm is None:
            try:
                groq_api_key = os.environ.get("GROQ_API_KEY")
                if not groq_api_key:
                    raise ValueError("GROQ_API_KEY environment variable not set.")
                cls._llm = ChatGroq(temperature=LLM_TEMPERATURE, groq_api_key=groq_api_key, model_name=GROQ_MODEL_NAME)
                print(f"Initialized ChatGroq LLM with model: {GROQ_MODEL_NAME}")
            except ValueError as e:
                print(f"Error initializing ChatGroq: {e}")
                print("Please set the GROQ_API_KEY environment variable.")
                exit()
        return cls._llm

    @classmethod
    def get_embedding_model(cls):
        if cls._embedding_model is None:
            print("Loading Sentence Transformer model 'all-MiniLM-L6-v2'...")
            cls._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Embedding model loaded.")
        return cls._embedding_model

# Initialize models globally on import to avoid re-initialization
LLM = LLMModels.get_llm()
EMBEDDING_MODEL = LLMModels.get_embedding_model()