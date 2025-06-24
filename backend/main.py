import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel 

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate


load_dotenv()

app = FastAPI()

# Cấu hình CORS
origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VECTOR_DB_PATH="chroma_db"

# Kiểm tra xem VectorDB đã tồn tại chưa
if not os.path.exists(VECTOR_DB_PATH):
    print(f'Error: thư mục \'{VECTOR_DB_PATH}\' không tìm thấy ')

# Khởi tạo Embedding Model 
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Load vector db
print(f'Đang tải Vector db {VECTOR_DB_PATH} ...')
vectorstore = Chroma(
    persist_directory = VECTOR_DB_PATH,
    embedding_function = embeddings_model
)
print("Vector Database đã được tải thành công!")

# Khởi tạo LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature = 0.7)

# Khởi tạo memory
memory = ConversationBufferMemory(memory_key = "chat_history", return_messages = True, output_key='answer')

CHAT_PPOMPT = PromptTemplate.from_template("""
Bạn là một trợ lý AI trả lời câu hỏi dựa trên các đoạn tài liệu liên quan.

CHỈ dựa trên tài liệu bên dưới để trả lời, KHÔNG phỏng đoán hoặc sáng tạo thông tin nếu không chắc chắn.

Nếu không tìm thấy câu trả lời, hãy nói rõ là không có thông tin.

Lịch sử hội thoại:
{chat_history}

Các đoạn tài liệu:
{context}

Câu hỏi của người dùng:
{question}

Câu trả lời chi tiết, chính xác và ngắn gọn:
""")


# Khởi tạo ConversationalRetrievalChain cho RAG và hội thoại
# Chain này sẽ tự động tìm kiếm (retrieve) các tài liệu liên quan
# và đưa chúng vào LLM cùng với lịch sử chat để tạo câu trả lời
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 8}),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": CHAT_PPOMPT},
    return_source_documents=True,
    output_key='answer'
)
print("RAG Chain và Memory đã được khởi tạo.")

# Định nghĩa Pydantic models
class ChatRequest(BaseModel):
    query: str
    # session_id: str = "default_session" 

class ChatResponse(BaseModel):
    answer: str

# Endpoint API cho Chatbot
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    user_query = request.query
    
    try:
        result = await qa_chain.ainvoke({"question": user_query})

        llm_answer = result["answer"]

        retrieved_docs = result["source_documents"]
        
        print("\n--- Các đoạn văn bản đã được truy xuất ---")
        for doc in retrieved_docs:
            print(doc.page_content[:200])


        return ChatResponse(answer=llm_answer)
    except Exception as e:
        print(f"Lỗi khi xử lý yêu cầu RAG: {e}")
        raise HTTPException(status_code=500, detail="Có lỗi xảy ra khi xử lý yêu cầu của bạn.")


@app.get("/")
async def read_root():
    return {"message": "Chatbot Backend is running!"}
