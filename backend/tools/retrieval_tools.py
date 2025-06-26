from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.tools import Tool 
from langchain.docstore.document import Document
from typing import List

from utils.llm_config import get_global_embeddings_model, get_mongo_connection_details, get_global_personal_memory_agent
from pymongo import MongoClient

MONGO_URI, DB_NAME, DOCS_COLLECTION_NAME, FACTS_COLLECTION_NAME, DOCS_VECTOR_INDEX_NAME, FACTS_VECTOR_INDEX_NAME = get_mongo_connection_details()

try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    
    embeddings_model = get_global_embeddings_model()

    # Vectorstore cho tài liệu gốc
    global_vectorstore_docs_retriever = MongoDBAtlasVectorSearch(
        collection=db[DOCS_COLLECTION_NAME],
        embedding=embeddings_model,
        index_name=DOCS_VECTOR_INDEX_NAME
    )
    print("RetrievalTools: Vectorstore tài liệu gốc đã sẵn sàng.")

except Exception as e:
    print(f"RetrievalTools: Lỗi khi khởi tạo MongoDB Atlas cho tài liệu gốc: {e}")
    raise RuntimeError("Không thể khởi tạo RetrievalTools do lỗi MongoDB.")


def get_global_retriever_main_docs():
    return global_vectorstore_docs_retriever.as_retriever(search_kwargs={"k": 15})

def get_global_retriever_long_term():
    personal_memory_agent_instance = get_global_personal_memory_agent()
    return personal_memory_agent_instance.long_term_vectorstore.as_retriever(search_kwargs={"k": 8})


document_retriever_tool = Tool(
    name="document_retriever",
    func=get_global_retriever_main_docs().ainvoke,
    description="Hữu ích khi cần truy xuất thông tin từ tài liệu gốc. Input là câu hỏi."
)

personal_facts_retriever_tool = Tool(
    name="personal_facts_retriever",
    func=get_global_retriever_long_term().ainvoke,
    description="Hữu ích khi cần truy xuất thông tin cá nhân hoặc các facts đã học về người dùng. Input là câu hỏi."
)