import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from langchain_community.vectorstores import Chroma

load_dotenv()

DATA_PATH="data"
VECTOR_DB_PATH="chroma_db"

def ingest_documents():
    print("Bắt đầu quá trình indexing documents...")

    documents = []

    for file_name in os.listdir(DATA_PATH):
        file_path = os.path.join(DATA_PATH, file_name)
        current_docs = []

        if (file_name.endswith('.pdf')):
            print(f'Đang tải file PDF: {file_name}')
            loader = PyPDFLoader(file_path)
            current_docs = loader.load()
        elif file_name.endswith('.docx'):
            print(f'Đang tải file docx: {file_name}')
            loader = Docx2txtLoader(file_path)
            current_docs = loader.load()
        elif file_name.endswith('.txt'):
            print(f'Đang tải file txt: {file_name}')
            loader = TextLoader(file_path)
            current_docs = loader.load()
        else:
            print(f'Không hỗ trợ định dạng cho file: {file_name}')
            continue
        
        documents.extend(current_docs)
    
    if not documents:
        print(f'Không tìm thấy tài liệu nào trong thư mục \'data/\'. Vui lòng thêm tài liệu vào.')
        return

    print(f'Số tài liệu đã tải lên: {len(documents)}')

    print('Đang chia nhỏ tài liệu ... ')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f'Tổng số chunks: {len(chunks)}')

    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    print(f'Lưu embeddings vào Vector database: {VECTOR_DB_PATH}')

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=VECTOR_DB_PATH
    )

    vectorstore.persist()
    print("Vector Database đã được tạo và lưu trữ thành công!")

if __name__ == "__main__":
    ingest_documents()