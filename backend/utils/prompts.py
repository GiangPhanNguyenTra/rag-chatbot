from langchain.prompts import PromptTemplate

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

FACT_EXTRACTION_PROMPT = PromptTemplate.from_template("""
Bạn là một trợ lý trích xuất thông tin chuyên nghiệp. Nhiệm vụ của bạn là đọc một đoạn hội thoại và xác định TẤT CẢ các THÔNG TIN MỚI, QUAN TRỌNG, hoặc các SỰ KIỆN CÁ NHÂN mà người dùng đã CUNG CẤP.

Các thông tin này phải là:
- Các mẩu dữ liệu độc lập, không phải là câu hỏi hoặc câu trả lời chung chung.
- Liên quan đến người dùng, người mà họ nhắc đến, hoặc các sự kiện cụ thể.
- Có thể được ghi nhớ và sử dụng để trả lời các câu hỏi trong tương lai.

TRẢ LỜI ĐÚNG ĐỊNH DẠNG: Mỗi thông tin trích xuất trên một DÒNG MỚI.
Nếu KHÔNG CÓ thông tin mới nào đáng kể để trích xuất, hãy trả lời CHÍNH XÁC là "NONE".

Ví dụ về trích xuất:
Người dùng: Tôi tên là Nam. Tôi thích chơi game.
Bot: Chào Nam! Tôi có thể giúp gì cho bạn?
Facts:
- Tên của người dùng là Nam.
- Người dùng thích chơi game.

Người dùng: Trần Đình Phương Linh là bố tôi á.
Bot: Tôi không tìm thấy thông tin về Trần Đình Phương Linh trong các tài liệu được cung cấp.
Facts:
- Trần Đình Phương Linh là bố của người dùng.

Người dùng: Tôi đang sống ở Hà Nội.
Bot: Hà Nội là thủ đô của Việt Nam.
Facts:
- Người dùng đang sống ở Hà Nội.

Người dùng: Điều này có nghĩa gì?
Bot: Điều đó có nghĩa là X.
Facts:
NONE

Lịch sử trò chuyện đầy đủ:
---
{full_conversation_formatted}
---

Thông tin quan trọng được trích xuất:
""")