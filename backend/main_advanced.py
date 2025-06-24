import os
import time
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import MergerRetriever

load_dotenv()
app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

DATA_PATH = "data"
VECTOR_DB_PATH = "chroma_db"
LONG_TERM_MEMORY_DB_PATH = "long_term_memory_db"

if not os.path.exists(VECTOR_DB_PATH):
    print(f"Error: Thư mục tài liệu gốc {VECTOR_DB_PATH} không tìm thấy")

embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.0)

# Tải tài liệu gốc
print(f"Đang tải tài liệu từ Vector db {VECTOR_DB_PATH}")
vectorstore_docs = Chroma(
    persist_directory=VECTOR_DB_PATH,
    embedding_function=embeddings_model
)
print("Vector Database tài liệu gốc đã được tải thành công!")

# Khởi tạo bộ nhớ dài hạn
long_term_vectorstore = None
if os.path.exists(LONG_TERM_MEMORY_DB_PATH):
    print(f"Đang tải Bộ nhớ dài hạn từ '{LONG_TERM_MEMORY_DB_PATH}'...")
    long_term_vectorstore = Chroma(
        persist_directory=LONG_TERM_MEMORY_DB_PATH,
        embedding_function=embeddings_model
    )
    print("Bộ nhớ dài hạn đã được tải thành công!")
else:
    print(f"Tạo mới bộ nhớ dài hạn tại '{LONG_TERM_MEMORY_DB_PATH}'...")
    long_term_vectorstore = Chroma(
        persist_directory=LONG_TERM_MEMORY_DB_PATH,
        embedding_function=embeddings_model
    )
    print("Bộ nhớ dài hạn đã được tạo mới thành công!")

# Khởi tạo bộ nhớ ngắn hạn
chat_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')

# Prompt
CHAT_PROMPT = PromptTemplate.from_template("""
Bạn là một trợ lý thông minh.
Trả lời ngắn gọn và dễ hiểu dựa trên:
- Lịch sử trò chuyện
- Thông tin ghi nhớ lâu dài từ người dùng
- Các tài liệu liên quan

Nếu không có thông tin, trả lời "Tôi không có thông tin đó."

Lịch sử trò chuyện:
{chat_history}

Thông tin ghi nhớ:
{long_term_facts}

Tài liệu liên quan:
{document_context}

Câu hỏi: {question}
Câu trả lời:
""")



# Hàm trích xuất và lưu Long-term Memory
async def extract_and_store_facts(query: str, answer: str, chat_history_messages: List[AIMessage | HumanMessage]):
    """
    Trích xuất các thông tin quan trọng (facts) từ hội thoại gần nhất và lưu vào bộ nhớ dài hạn (vectorstore).
    """

    # 1. Format lại lịch sử chat
    formatted_chat_history = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in chat_history_messages])

    # 2. Prompt rõ ràng cho LLM biết cần làm gì
    extraction_prompt_template = PromptTemplate.from_template("""
    Bạn là một trợ lý trích xuất thông tin. 
    Nhiệm vụ của bạn là đọc đoạn hội thoại và xác định các thông tin đáng nhớ mà người dùng cung cấp.
    
    Chỉ trích xuất những thông tin như:
    - Ai là ai (quan hệ, nghề nghiệp, vị trí)
    - Thông tin cá nhân, nơi ở, sở thích, mục tiêu, sự kiện
    - Bất kỳ câu nào người dùng muốn bạn nhớ lâu dài

    Trả kết quả ở dạng MỖI DÒNG LÀ MỘT CÂU FACT.
    Không cần ghi số thứ tự, không cần chú thích.
    Nếu không tìm thấy fact nào, trả lời chính xác là "NONE".

    Lịch sử hội thoại:
    {chat_history_formatted}

    Câu hỏi: {query}
    Trả lời: {answer}

    Facts:
    """)

    # 3. Tạo chain với LLM (nên dùng temperature 0.0 để output nhất quán)
    llm_extractor = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.0)
    extraction_chain = extraction_prompt_template | llm_extractor

    # 4. Gọi LLM để trích xuất facts
    response = await extraction_chain.ainvoke({
        "chat_history_formatted": formatted_chat_history,
        "query": query,
        "answer": answer
    })

    facts_output = response.content.strip()
    print("\n📥 LLM extracted facts raw output:\n", facts_output)

    # 5. Xử lý kết quả
    if facts_output.upper() == "NONE" or not facts_output.strip():
        print("❌ Không có facts nào được trích xuất.")
        return

    facts_lines = [line.strip() for line in facts_output.split("\n") if line.strip()]
    fact_docs = [
        Document(
            page_content=fact,
            metadata={
                "source": "long-term-memory",
                "type": "fact_from_chat",
                "timestamp": time.time()
            }
        )
        for fact in facts_lines
    ]

    # 6. Lưu facts vào vectorstore dài hạn
    long_term_vectorstore.add_documents(fact_docs)
    print("✅ Đã lưu các facts vào bộ nhớ dài hạn:", facts_lines)



# RETRIEVER
class ChatRequest(BaseModel):
    query: str
    # session_id: str = "default_session" 

class ChatResponse(BaseModel):
    answer: str
    
# Cấu trúc request/response
def extract_name_from_question(question:str) -> str:
    if "là ai" in question.lower():
        return question.lower().replace("là ai", "").replace("?", "").strip().title()
    elif  question.lower().startswith("ai là"):
        return question[6:].replace("?", "").strip().title()
    elif "thông tin về" in question.lower():
        return question.lower().replace("thông tin về", "").replace("?", "").strip().title()
    return question.strip()

# Định nghĩa Retriever tùy chỉnh để kết hợp 2 vector db
retriever_main_docs = vectorstore_docs.as_retriever(search_kwargs={"k": 5})
retriever_long_term = long_term_vectorstore.as_retriever(search_kwargs={"k":5})

# Sử dụng MergerRetriever để kết hợp các retriever
combined_retriever = MergerRetriever(retrievers=[retriever_main_docs, retriever_long_term])


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint_advanced(request: ChatRequest):
    user_query = request.query
    
    try:
        current_chat_history_messages = chat_memory.load_memory_variables({})["chat_history"]
        
        retrieved_docs = await retriever_main_docs.aget_relevant_documents(user_query)
        context = "\n\n".join([d.page_content for d in retrieved_docs])
        
        keyword = extract_name_from_question(user_query)
        facts = await retriever_long_term.aget_relevant_documents(keyword)
        facts_context = "\n\n".join([f.page_content for f in facts])
        
        history_str = "\n".join([f"{m.type.capitalize()}: {m.content}" for m in current_chat_history_messages])

        print("\n--- DEBUG: Truy xuất facts liên quan keyword:", keyword)
        for fact in facts:
            print(f"💡 FACT: {fact.page_content}")

        result = await (CHAT_PROMPT | llm).ainvoke({
                "chat_history": history_str,
                "document_context": context,
                "long_term_facts": facts_context,
                "question": user_query
        })


        llm_answer = result.content

        print("\n--- Các đoạn văn bản đã được truy xuất ---")
        if retrieved_docs:
            for i, doc in enumerate(retrieved_docs):
                print(f"--- Tài liệu {i+1} ---")
                print(f"Nội dung: {doc.page_content[:200]}...")
                print("-" * 30)
        else:
            print("Không có tài liệu nào được truy xuất.")
        print("-------------------------------------------\n")

        # Trích xuất và lưu vào bộ nhớ dài hạn
        await extract_and_store_facts(user_query, llm_answer, current_chat_history_messages)
        return ChatResponse(answer=llm_answer)
    except Exception as e:
        print(f"Lỗi khi xử lý yêu cầu RAG: {e}")
        raise HTTPException(status_code=500, detail="Có lỗi xảy ra khi xử lý yêu cầu của bạn.")

@app.get("/")
async def read_root():
    return {"message": "Chatbot Backend (Advanced) is running!"}

@app.get("/longterm-facts")
async def show_long_term_facts():
    docs = long_term_vectorstore.similarity_search("", k=20)
    return {"results": [doc.page_content for doc in docs]}
