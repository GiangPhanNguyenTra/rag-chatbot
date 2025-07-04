import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from graph.builder import create_chatbot_graph
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
app = FastAPI(title="Multi-Agent Chatbot API")

origins = [
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

chatbot_graph = create_chatbot_graph()

class ChatMessage(BaseModel):
    role: str # "human" or "ai"
    content: str

class ChatRequest(BaseModel):
    query: str
    chat_history: List[ChatMessage] = []

class ChatResponse(BaseModel):
    answer: str

@app.post("/chat", response_model = ChatResponse)
async def chat_with_multi_agent(request: ChatRequest):
    langchain_history = []
    for msg in request.chat_history:
        if msg.role == "human":
            langchain_history.append(HumanMessage(content=msg.content))
        elif msg.role == "ai":
            langchain_history.append(AIMessage(content=msg.content))
        else:
            raise HTTPException(status_code=400, detail="Invalid message role")
    
    # Chuẩn bị cho input đồ thị
    initial_state = {
        "original_question": request.query,
        "chat_history": langchain_history
    }

    try:
        # ainvoke sẽ chạy toàn bộ đồ thị và trả về state cuối cùng
        final_state = await chatbot_graph.ainvoke(initial_state)
        answer = final_state.get("final_answer", "Xin lỗi, tôi không thể tìm thấy câu trả lời.")

        return ChatResponse(answer=answer)
    except Exception as e:
        print(f"Error: Lỗi khi xử lý đồ thị LangGraph: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Có lỗi xảy ra khi xử lý yêu cầu của bạn.")

@app.get("/")
async def read_root():
    return {"message": "Multi-Agent Chatbot Backend is running!"}