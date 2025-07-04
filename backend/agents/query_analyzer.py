from graph.state import GraphState
from typing import Dict, List
from langchain.prompts import PromptTemplate
from utils.llm_config import get_global_llm

QUERY_EXPANSION_PROMPT = PromptTemplate.from_template(
    """Bạn là một chuyên gia về truy vấn thông tin. Nhiệm vụ của bạn là đọc câu hỏi của người dùng và tạo ra 3 phiên bản câu hỏi khác, sử dụng các từ đồng nghĩa và cách diễn đạt khác nhau để tối đa hóa khả năng tìm thấy thông tin liên quan trong một cơ sở dữ liệu vector.

    Ví dụ:
    - Câu hỏi gốc: "Sinh nhật của Phương Linh là ngày mấy?"
    - Các câu hỏi thay thế:
        Phương Linh sinh ngày nào?
        Ngày sinh của Phương Linh
        Thông tin về ngày sinh của Phương Linh

    - Câu hỏi gốc: "So sánh Singleton và Factory Pattern"
    - Các câu hỏi thay thế:
        Sự khác biệt giữa Singleton Pattern và Factory Pattern
        Singleton Pattern vs Factory Pattern
        Ưu và nhược điểm của Singleton và Factory Pattern
    
    Bây giờ, hãy thực hiện với câu hỏi sau. Trả lời chỉ với các câu hỏi được tạo, mỗi câu trên một dòng mới. KHÔNG thêm bất kỳ lời giải thích nào.

    Câu hỏi gốc: {question}
    
    Các câu hỏi thay thế:
    """
)

async def analyze_query(state: GraphState) -> dict:
    print("---NODE: Query Analyzer--- \n")

    user_question = state["original_question"]
    llm = get_global_llm()

    generation_chain = QUERY_EXPANSION_PROMPT | llm
    response = await generation_chain.ainvoke({"question": user_question})
    queries = [line.strip() for line in response.content.split("\n") if line.strip()]

    all_queries = [user_question] + queries
    print(f"-> Generated queries: {all_queries}")

    return {"expanded_queries": all_queries}