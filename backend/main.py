# import os
# from dotenv import load_dotenv
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import google.generativeai as genai

# load_dotenv()

# origins = [
#     "http://localhost:3000"
# ]

# class ChatRequest(BaseModel):
#     query: str
#     # session_id: str

# class ChatResponse(BaseModel):
#     answer: str

# def create_app():
#     _app = FastAPI()

#     _app.add_middleware(
#         CORSMiddleware,
#         allow_origins=origins,
#         allow_methods=["*"],
#         allow_headers=["*"]
#     )
    
#     genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
#     _llm_model = genai.GenerativeModel(model_name="gemini-1.5-flash")


#     @_app.post("/chat", response_model=ChatResponse)
#     async def chat_endpoint(request: ChatRequest):
#         user_query = request.query
#         try:
#             promtp_for_gemini = f"Bạn là một trợ lý chatbot thân thiện. Trả lời câu hỏi sau: {user_query}"

#             response = _llm_model.generate_content(
#                 promtp_for_gemini,
#                 generation_config = genai.types.GenerationConfig(temperature=0.7)
#             )

#             llm_answer = response.text

#             return ChatResponse(answer=llm_answer)
#         except Exception as e:
#             print(f"Lỗi khi gọi Gemini API: {e}")
#             raise HTTPException(status_code=500, detail="Có lỗi xảy ra khi xử lý yêu cầu từ LLM.")

#     @_app.get("/")
#     async def read_root():
#         return {"message": "Chatbot is running"}

# if __name__ == "__main__":
#     import uvicorn

#     app = create_app()
#     uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)

# backend/main.py
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel 

import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
llm_model = genai.GenerativeModel(model_name="gemini-1.5-flash")

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

# Định nghĩa Pydantic models
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str

# Endpoint API cho Chatbot
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    user_query = request.query
    
    try:
        prompt_for_gemini = f"Bạn là một trợ lý chatbot thân thiện. Trả lời câu hỏi sau: {user_query}"
        response = llm_model.generate_content(
            prompt_for_gemini,
            generation_config=genai.types.GenerationConfig(temperature=0.7)
        )
        llm_answer = response.text
        return ChatResponse(answer=llm_answer)
    except Exception as e:
        print(f"Lỗi khi gọi Gemini API: {e}")
        raise HTTPException(status_code=500, detail="Có lỗi xảy ra khi xử lý yêu cầu từ LLM.")

@app.get("/")
async def read_root():
    return {"message": "Chatbot Backend is running!"}
