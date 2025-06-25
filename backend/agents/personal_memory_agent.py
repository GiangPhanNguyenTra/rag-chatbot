import os
import time
import hashlib
from typing import List, Dict, Any

from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage

from utils.llm_config import get_global_embeddings_model, get_global_extraction_llm, get_mongo_connection_details
from utils.prompts import FACT_EXTRACTION_PROMPT

from pymongo import MongoClient


class PersonalMemoryAgent:
    """
    Agent quản lý bộ nhớ dài hạn của người dùng, trích xuất và lưu trữ facts,
    đồng thời truy xuất chúng khi cần, tránh trùng lặp và đảm bảo truy xuất chính xác.
    """
    def __init__(self):
        print("PersonalMemoryAgent: Khởi tạo bộ nhớ dài hạn...")

        # 1. Lấy các instance LLM và Embeddings dùng chung
        self.embeddings_model = get_global_embeddings_model()
        self.extraction_llm = get_global_extraction_llm()

        # 2. Thiết lập kết nối MongoDB Atlas
        MONGO_URI, DB_NAME, _, FACTS_COLLECTION_NAME, _, FACTS_VECTOR_INDEX_NAME = get_mongo_connection_details()
        
        try:
            self.client = MongoClient(MONGO_URI)
            self.db = self.client[DB_NAME]
            self.long_term_collection = self.db[FACTS_COLLECTION_NAME]
            # Đảm bảo có index cho hash để kiểm tra trùng lặp nhanh
            self.long_term_collection.create_index("content_hash", unique=True)
            print(f"PersonalMemoryAgent: Kết nối tới MongoDB Atlas. Collection: {FACTS_COLLECTION_NAME}")
        except Exception as e:
            print(f"PersonalMemoryAgent: Lỗi kết nối MongoDB Atlas: {e}")
            raise RuntimeError("Không thể kết nối đến MongoDB Atlas.")

        # 3. Khởi tạo MongoDB Atlas Vector Store
        self.long_term_vectorstore = MongoDBAtlasVectorSearch(
            collection=self.long_term_collection,
            embedding=self.embeddings_model,
            index_name=FACTS_VECTOR_INDEX_NAME
        )
        print("PersonalMemoryAgent: Bộ nhớ dài hạn (Vector Search) đã sẵn sàng.")

    def _compute_content_hash(self, content: str) -> str:
        """Tính hash của nội dung fact để kiểm tra trùng lặp."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    async def store_conversation_facts(self, query: str, answer: str, chat_history_messages: List[AIMessage | HumanMessage]):
        """
        Trích xuất và lưu facts từ hội thoại vào bộ nhớ dài hạn, đảm bảo không trùng lặp.
        """
        # Format hội thoại bao gồm query và answer hiện tại
        full_conversation = chat_history_messages + [HumanMessage(content=query), AIMessage(content=answer)]
        formatted_conversation = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in full_conversation])

        # Chuẩn bị chain trích xuất facts
        extraction_chain = FACT_EXTRACTION_PROMPT | self.extraction_llm

        try:
            print("\n--- PersonalMemoryAgent: Trích xuất facts từ hội thoại ---")
            extracted_result = await extraction_chain.ainvoke({
                "full_conversation_formatted": formatted_conversation,
            })
            facts_str = extracted_result.content.strip()
            print(f"PersonalMemoryAgent: Kết quả trích xuất: {facts_str[:200]}...")

            # Tách các fact thành danh sách
            facts_list = [line.strip() for line in facts_str.split("\n") if line.strip()]
            new_unique_facts = []

            # Kiểm tra trùng lặp và thêm fact đảo chiều
            for fact_content in facts_list:
                fact_hash = self._compute_content_hash(fact_content)

                # Kiểm tra hash trong MongoDB
                if not self.long_term_collection.find_one({"content_hash": fact_hash}):
                    # Kiểm tra thêm similarity để đảm bảo
                    similar_facts = self.long_term_vectorstore.similarity_search_with_score(fact_content, k=1)
                    if not similar_facts or similar_facts[0][1] < 0.95:
                        new_unique_facts.append((fact_content, fact_hash))
                        print(f"Thêm fact mới: {fact_content}")

                        # Tạo và kiểm tra fact đảo chiều
                        reversed_fact = generate_reversed_fact(fact_content)
                        if reversed_fact:
                            reversed_hash = self._compute_content_hash(reversed_fact)
                            if not self.long_term_collection.find_one({"content_hash": reversed_hash}):
                                reversed_sim = self.long_term_vectorstore.similarity_search_with_score(reversed_fact, k=1)
                                if not reversed_sim or reversed_sim[0][1] < 0.95:
                                    new_unique_facts.append((reversed_fact, reversed_hash))
                                    print(f"Thêm fact đảo chiều: {reversed_fact}")
                    else:
                        print(f"Bỏ qua fact tương tự: {fact_content}")
                else:
                    print(f"Bỏ qua fact trùng lặp: {fact_content}")

            # Lưu các fact mới vào vector store
            if new_unique_facts:
                fact_docs = [
                    Document(
                        page_content=fact,
                        metadata={
                            "source": "long-term-memory-chat",
                            "type": "fact_from_chat",
                            "timestamp": time.time(),
                            "content_hash": fact_hash
                        }
                    )
                    for fact, fact_hash in new_unique_facts
                ]
                self.long_term_vectorstore.add_documents(fact_docs)
                print(f"PersonalMemoryAgent: Lưu {len(new_unique_facts)} facts mới: {[fact for fact, _ in new_unique_facts]}")
            else:
                print("PersonalMemoryAgent: Không có facts mới để lưu.")

        except Exception as e:
            print(f"PersonalMemoryAgent: Lỗi khi trích xuất/lưu facts: {e}")
            import traceback
            traceback.print_exc()

    async def retrieve_relevant_facts(self, query: str, k: int = 5) -> List[Document]:
        """
        Truy xuất các facts liên quan từ bộ nhớ dài hạn cho truy vấn.
        """
        print(f"PersonalMemoryAgent: Truy xuất facts cho truy vấn: '{query}' (k={k})")
        
        # Tạo retriever với các tham số tìm kiếm
        retriever = self.long_term_vectorstore.as_retriever(search_kwargs={"k": k})
        
        # Truy xuất facts
        facts = await retriever.ainvoke(query)
        
        print(f"PersonalMemoryAgent: Tìm thấy {len(facts)} facts liên quan: {[doc.page_content for doc in facts]}")
        return facts


def generate_reversed_fact(fact: str) -> str | None:
    """Tạo fact đảo chiều cho các mẫu quan hệ phù hợp."""
    patterns = {
        "là bạn gái của": lambda subj, obj: f"{obj.strip()} là người yêu của {subj.strip()}.",
        "là bạn trai của": lambda subj, obj: f"{obj.strip()} là người yêu của {subj.strip()}.",
        "là con của": lambda subj, obj: f"{obj.strip()} là cha/mẹ của {subj.strip()}.",
        "là bạn của": lambda subj, obj: f"{obj.strip()} là bạn của {subj.strip()}."
    }
    
    for pattern, reverse_fn in patterns.items():
        if pattern in fact:
            subj, obj = fact.split(pattern)
            return reverse_fn(subj, obj)
    
    return None


global_personal_memory_agent_instance: PersonalMemoryAgent = None


def get_global_personal_memory_agent() -> PersonalMemoryAgent:
    """Lấy instance singleton của PersonalMemoryAgent."""
    global global_personal_memory_agent_instance
    if global_personal_memory_agent_instance is None:
        global_personal_memory_agent_instance = PersonalMemoryAgent()
    return global_personal_memory_agent_instance