import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any

from agents.orchestrator import invoke_orchestrator_agent
from utils.llm_config import get_global_llm, get_global_embeddings_model, get_global_extraction_llm, get_mongo_connection_details

from pymongo import MongoClient

load_dotenv()
app = FastAPI()

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

MONGO_URI, DB_NAME, DOCS_COLLECTION_NAME, FACTS_COLLECTION_NAME,  DOCS_VECTOR_INDEX_NAME, FACTS_VECTOR_INDEX_NAME = get_mongo_connection_details()
mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client[DB_NAME]

global_llm = get_global_llm()
global_embeddings_model = get_global_embeddings_model()
global_extraction_llm = get_global_extraction_llm()

class ChatRequest(BaseModel):
    query: str
    # session_id: str = "default_session" 

class ChatResponse(BaseModel):
    answer: str

@app.post("/chat", response_model = ChatResponse)
async def chat_with_multi_agent(request: ChatRequest):
    user_query = request.query
    try:
        response = await invoke_orchestrator_agent(user_query)
        llm_answer = response["answer"]

        return ChatResponse(answer=llm_answer)
    except Exception as e:    
        print(f"Error: khi xử lý yêu cầu của Multi Agent: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Có lỗi xảy ra khi xử lý yêu cầu của bạn.")


@app.get("/")
async def read_root():
    return {"message": "Multi-Agent Chatbot Backend is running!"}