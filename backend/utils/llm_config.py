from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

GLOBAL_LLM = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature = 0.7)
GLOBAL_EMBEDDINGS_MODEL = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
GLOBAL_EXTRACTION_LLM = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature = 0.0)

MONGO_URI = os.environ.get("MONGO_URI")
DB_NAME = "rag_db"
DOCS_COLLECTION_NAME = "documents_collection"
FACTS_COLLECTION_NAME = "long_term_facts_collection"
DOCS_VECTOR_INDEX_NAME = "vector_index_documents"
FACTS_VECTOR_INDEX_NAME = "vector_index_facts"
global_personal_memory_agent_instance = None 

def get_global_llm():
    return GLOBAL_LLM
def get_global_embeddings_model():
    return GLOBAL_EMBEDDINGS_MODEL
def get_global_extraction_llm():
    return GLOBAL_EXTRACTION_LLM
def get_mongo_connection_details():
    return MONGO_URI, DB_NAME, DOCS_COLLECTION_NAME, FACTS_COLLECTION_NAME, DOCS_VECTOR_INDEX_NAME, FACTS_VECTOR_INDEX_NAME

def get_global_personal_memory_agent():
    global global_personal_memory_agent_instance
    if global_personal_memory_agent_instance is None:
        from agents.personal_memory_agent import PersonalMemoryAgent 
        global_personal_memory_agent_instance = PersonalMemoryAgent()
    return global_personal_memory_agent_instance