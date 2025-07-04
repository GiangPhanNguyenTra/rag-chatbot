import time
from typing import List
from langchain.docstore.document import Document 
from langchain.schema import HumanMessage, AIMessage
from langchain.tools import Tool

from utils.llm_config import get_global_extraction_llm, get_mongo_connection_details, get_global_embeddings_model
from utils.prompts import FACT_EXTRACTION_PROMPT

from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
MONGO_URI, DB_NAME, DOCS_COLLECTION_NAME, FACTS_COLLECTION_NAME, DOCS_VECTOR_INDEX_NAME, FACTS_VECTOR_INDEX_NAME = get_mongo_connection_details()

try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    long_term_collection_for_facts = db[FACTS_COLLECTION_NAME]
    print(f"FactExtractionTool: Đã kết nối MongoDB Atlas cho collection '{FACTS_COLLECTION_NAME}'.")

    long_term_vectorstore_writer = MongoDBAtlasVectorSearch(
        collection=long_term_collection_for_facts,
        embedding=get_global_embeddings_model(),
        index_name=FACTS_VECTOR_INDEX_NAME 
    )
    print("FactExtractionTool: MongoDB Atlas Vector Search writer đã sẵn sàng.")
except Exception as e:
    print(f"FactExtractionTool: Lỗi khi khởi tạo MongoDB Atlas cho Fact Extraction: {e}")
    raise RuntimeError("Không thể khởi tạo FactExtractionTool do lỗi MongoDB.")

def extract_name_from_question(question: str) -> str:
    """Trích xuất tên từ câu hỏi 'Ai là ai?', 'Thông tin về ai?', v.v."""
    if "là ai" in question.lower():
        return question.lower().replace("là ai", "").replace("?", "").strip().title()
    elif question.lower().startswith("ai là"):
        return question[6:].replace("?", "").strip().title()
    elif "thông tin về" in question.lower():
        return question.lower().replace("thông tin về", "").replace("?", "").strip().title()
    return question.strip() # Trả về nguyên câu hỏi nếu không khớp


async def _internal_extract_and_store_facts(query: str, answer: str, chat_history_messages: List[AIMessage | HumanMessage]):
    """
    Trích xuất các thông tin quan trọng (facts) từ hội thoại gần nhất và lưu vào bộ nhớ dài hạn (MongoDB Atlas Vectorstore).
    """
    # 1. Format lại lịch sử chat (bao gồm cả lượt hiện tại cho LLM)
    full_conversation_for_extraction = chat_history_messages + [HumanMessage(content=query), AIMessage(content=answer)]
    formatted_full_conversation = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in full_conversation_for_extraction])

    # 2. Lấy LLM trích xuất 
    llm_extractor = get_global_extraction_llm()
    extraction_chain = FACT_EXTRACTION_PROMPT | llm_extractor 

    # 3. Gọi LLM để trích xuất facts
    try:
        print("\n--- FactExtractionTool: Bắt đầu trích xuất facts từ hội thoại ---")
        response = await extraction_chain.ainvoke({"full_conversation_formatted": formatted_full_conversation})
        facts_output = response.content.strip()
        print(f"FactExtractionTool: Kết quả trích xuất thô: {facts_output[:200]}...")

        # 4. Xử lý kết quả
        if facts_output.upper() == "NONE" or not facts_output.strip():
            print("FactExtractionTool: ❌ Không có facts nào được trích xuất.")
            return

        facts_lines = [line.strip() for line in facts_output.split("\n") if line.strip()]
        fact_docs = [
            Document(
                page_content=fact,
                metadata={
                    "source": "long-term-memory-chat",
                    "type": "fact_from_chat",
                    "timestamp": time.time()
                }
            )
            for fact in facts_lines
        ]

        long_term_vectorstore_writer.add_documents(fact_docs)
        print("FactExtractionTool: ✅ Đã lưu các facts vào bộ nhớ dài hạn:", facts_lines)

    except Exception as e:
        print(f"FactExtractionTool: Lỗi khi trích xuất hoặc lưu facts: {e}")
        import traceback
        traceback.print_exc()
