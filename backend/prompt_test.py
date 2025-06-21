import os
from dotenv import load_dotenv
import google.generativeai as genai

# 1. Load biến môi trường
load_dotenv()
print(os.environ.get("GOOGLE_API_KEY"))

# 2. Cấu hình Gemini API với API Key
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# 3. Khởi tạo mô hình Gemini
model = genai.GenerativeModel(model_name="gemini-1.5-flash")   

def  get_gemini_response(prompt_text, temperature = 0.7):
    try:
        response = model.generate_content(
            prompt_text,
            generation_config=genai.types.GenerationConfig(temperature=temperature)
        )
        return response.text
    except Exception as e:
        print("Lỗi khi gọi Gemini API: " + e)
        return "Xin lỗi, toi không thể sử lý yêu cầu lúc này"
    

# --------------- Prompt Engineering ---------------
print("--- 1. Zero shot ---")
prompt1 = "Giải thích Machine Learning là gì một cách ngắn gọn."
print(f"Prompt: {prompt1}")
print(f"Trả lời: {get_gemini_response(prompt1)}\n")

print("--- 2. Role playing & Constraints ---")
prompt2 =  """Bạn là một giáo viên tiểu học thân thiện và kiến nhẫn.
Hãy giải thích khái niệm 'Trọng lực' cho học sinh lớp 3.
Sử dụng các từ ngữ đơn giản và gần gũi với trẻ em.
Không sử dụng các thuật ngữ khoa học phức tạp
"""
print(f"Prompt: {prompt2}")
print(f"Trả lời: {get_gemini_response(prompt1, temperature=0.5)}\n")

print("--- 3. Few Shot ---")
prompt3 = """Chuyển đổi tên các quốc gia sau sang dạng viết tắt 3 chữ cái theo ví dụ.
Chỉ trả về dạng viết tắt.

Quốc gia: Việt Nam
Viết tắt: VNM

Quốc gia: Hoa Kỳ
Viết tắt: USA

Quốc gia: Anh
Viết tắt: GBR

Quốc gia: Nhật Bản
Viết tắt:
"""
print(f"Prompt: {prompt3}")
print(f"Trả lời: {get_gemini_response(prompt3, temperature=0.0)}\n")


print("--- 4. Suy nghĩ từng bước ---")
prompt4 = """Tôi có 3 quả táo. Tôi cho bạn 1 quả và ăn đi 1 quả. Sau đó tôi mua thêm 2 quả nữa.
Hỏi tôi còn bao nhiêu quả táo?
Hãy suy nghĩ từng bước một để ra kết quả cuối cùng.
"""
print(f"Prompt: {prompt4}")
print(f"Trả lời: {get_gemini_response(prompt4, temperature=0.3)}\n")

print("--- 5. Context & Constraints ---")
context5 = """
Học không giám sát (Unsupervised Learning) là một loại hình học máy sử dụng dữ liệu không có nhãn. Mục tiêu chính của nó là tìm kiếm các cấu trúc ẩn, mẫu, hoặc mối quan hệ bên trong dữ liệu. Các kỹ thuật chính bao gồm phân cụm (clustering), liên kết (association) và giảm chiều dữ liệu (dimensionality reduction). Phân cụm nhóm các điểm dữ liệu tương tự lại với nhau. Giảm chiều dữ liệu giúp giảm số lượng đặc trưng trong khi vẫn giữ được thông tin quan trọng.
"""
question5 = "Kể tên các kỹ thuật chính của học không giám sát được đề cập trong đoạn văn trên"
prompt5 = f"""Bạn là một trợ lý hỏi đáp tài liệu.
Hãy trả lời câu hỏi sau dựa trên đoạn văn được cung cấp.
Nếu câu trả lời không có trong đoạn văn hãy nói "Không tìm thấy thông tin liên quan."

Đoạn văn: 
---
{context5}
---

Câu hỏi: {question5}
"""

print(f"Prompt: {prompt5}")
print(f"Trả lời: {get_gemini_response(prompt5, temperature=0.0)}\n")

print("--- 6. Hỏi ngoài ngữ cảnh ---")
question6 = "Thủ đô nước pháp là gì?"
prompt6 = f"""Bạn là một trợ lý hỏi đáp tài liệu.
Hãy trả lời câu hỏi sau dựa trên đoạn văn được cung cấp.
Nếu câu trả lời không có trong đoạn văn hãy nói 'Không tìm thấy thông tin liên quan.'

Đoạn văn:
---
{context5}
---

Câu hỏi: {question6}
"""

print(f"Prompt: {prompt6}")
print(f"Trả lời: {get_gemini_response(prompt6, temperature=0.0)}\n")
