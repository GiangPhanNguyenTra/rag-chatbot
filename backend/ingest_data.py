import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_mongodb import MongoDBAtlasVectorSearch 
from pymongo import MongoClient
from langchain_experimental.text_splitter import SemanticChunker
from utils.llm_config import get_global_embeddings_model, get_mongo_connection_details
import certifi

load_dotenv()

DATA_PATH = "data"

def clean_text(text: str) -> str:
    if "mục lục" in text.lower() or len(text.strip()) < 30:
        return ""
    return text.strip()


def ingest_documents_to_mongodb_atlas():
    print("Bắt đầu quá trình lập chỉ mục tài liệu vào MongoDB Atlas...")

    MONGO_URI, DB_NAME, DOCS_COLLECTION_NAME, _, DOCS_VECTOR_INDEX_NAME, _ = get_mongo_connection_details()

    embeddings_model = get_global_embeddings_model()

    client = MongoClient(MONGO_URI, tls=True, tlsCAFile=certifi.where())
    try:
        client.admin.command("ping")
        print("✅ Đã kết nối MongoDB Atlas thành công!")
    except Exception as e:
        print("❌ Kết nối thất bại:", e)
        exit()

    db = client[DB_NAME]
    collection = db[DOCS_COLLECTION_NAME]

    documents = []
    for file_name in os.listdir(DATA_PATH):
        file_path = os.path.join(DATA_PATH, file_name)
        current_docs = []

        if file_name.endswith(".pdf"):
            print(f"Đang tải file PDF: {file_name}")
            loader = PyPDFLoader(file_path)
            current_docs = loader.load()
        elif file_name.endswith(".txt"):
            print(f"Đang tải file TXT: {file_name}")
            loader = TextLoader(file_path, encoding='utf-8')
            current_docs = loader.load()
        elif file_name.endswith(".docx"):
            print(f'Đang tải file docx: {file_name}')
            loader = Docx2txtLoader(file_path)
            current_docs = loader.load()
        else:
            print(f'Không hỗ trợ định dạng cho file: {file_name}')
            continue
        for doc in current_docs:
            doc.page_content = clean_text(doc.page_content)
            doc.metadata["source_file"] = file_name
        documents.extend(current_docs)

    if not documents:
        print("Không tìm thấy tài liệu nào trong thư mục 'data/'. Vui lòng thêm tài liệu vào.")
        return
    
    print(f"Tổng số {len(documents)} tài liệu đã được tải.")

    # Chia thành các Chunks
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size = 1000,
    #     chunk_overlap = 200,
    #     separators=["\n\n", "\n", " ", ""]
    # )

    # chunks = [
    #     chunk for chunk in text_splitter.split_documents(documents)
    #     if len(chunk.page_content.strip()) > 50
    # ]

    text_splitter = SemanticChunker(
        embeddings_model, 
        breakpoint_threshold_type="percentile" # Đây là phương pháp xác định điểm ngắt phổ biến
    )
    all_texts = "\n\n".join([doc.page_content for doc in documents])
    chunks = text_splitter.create_documents([all_texts])
    print("Đang chia từng tài liệu để giữ lại metadata...")
    chunks = []
    for doc in documents:
        # Chạy semantic chunker trên nội dung của từng document
        doc_chunks = text_splitter.create_documents([doc.page_content])
        # Gán lại metadata từ document gốc cho các chunk mới được tạo
        for chunk in doc_chunks:
            chunk.metadata = doc.metadata.copy()
        chunks.extend(doc_chunks)

    # BẠN CÓ THỂ GIỮ LẠI BỘ LỌC NÀY
    # Lọc bỏ các chunks quá ngắn, có thể là nhiễu
    chunks = [
        chunk for chunk in chunks
        if len(chunk.page_content.strip()) > 50
    ]

    print(f"Tổng số {len(chunks)} đoạn văn bản đã được tạo.")

    # Lưu vào MongoAtlas
    vectorstore_atlas = MongoDBAtlasVectorSearch.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        collection=collection,
        index_name = DOCS_VECTOR_INDEX_NAME
    )

    print("Đã lưu documents vào MongoDB Atlas!")

if __name__ == "__main__":
    ingest_documents_to_mongodb_atlas()