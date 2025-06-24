import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

VECTOR_DB_PATH = "chroma_db"

def test_retrieval():
    if not os.path.exists(VECTOR_DB_PATH):
        print(f"Lỗi: Thư mục Vector Database '{VECTOR_DB_PATH}' không tìm thấy.")
        print("Vui lòng chạy 'python ingest_data.py' trước để tạo Vector Database.")
        return

    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings_model
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # Thử k=5 để lấy nhiều hơn

    test_query = "Tổng có mấy nhóm mẫu design pattern" # Câu hỏi bạn đã thử
    
    print(f"Đang tìm kiếm tài liệu cho câu hỏi: '{test_query}'")
    retrieved_docs = retriever.invoke(test_query)

    print("\n--- Các đoạn văn bản đã được truy xuất ---")
    if retrieved_docs:
        for i, doc in enumerate(retrieved_docs):
            print(f"--- Tài liệu {i+1} ---")
            print(f"Nguồn: {doc.metadata.get('source', 'Không rõ')}")
            print(f"Trang: {doc.metadata.get('page', 'N/A')}")
            print(f"Nội dung: {doc.page_content[:500]}...")
            print("-" * 30)
    else:
        print("Không có tài liệu nào được truy xuất.")
    print("-------------------------------------------\n")

if __name__ == "__main__":
    test_retrieval()