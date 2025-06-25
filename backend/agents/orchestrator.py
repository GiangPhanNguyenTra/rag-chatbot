from typing import Dict, List
from langchain.schema import HumanMessage, AIMessage 
from langchain.memory import ConversationBufferMemory

from utils.llm_config import get_global_llm, get_global_personal_memory_agent 
from utils.prompts import CHAT_PROMPT
from tools.retrieval_tools import get_global_retriever_main_docs, get_global_retriever_long_term 

llm_orchestrator = get_global_llm()
personal_memory_agent = get_global_personal_memory_agent()

retriever_main_docs = get_global_retriever_main_docs()
retriever_long_term = get_global_retriever_long_term()

chat_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')

async def invoke_orchestrator_agent(user_query: str) -> Dict:
    print(f"Orchestrator nhận câu hỏi: {user_query}")
    
    current_chat_history_messages_list = chat_memory.load_memory_variables({})['chat_history']

    print("Orchestrator: Đang truy xuất tài liệu từ nguồn chính...")
    retrieved_docs_main = await retriever_main_docs.ainvoke(user_query)
    document_context = "\n\n".join([doc.page_content for doc in retrieved_docs_main])
    
    print("Orchestrator: Đang truy xuất facts từ bộ nhớ dài hạn...")
    retrieved_facts = await retriever_long_term.ainvoke(user_query)
    long_term_facts_context = "\n\n".join([fact.page_content for fact in retrieved_facts])

    print("\n--- Orchestrator: Các đoạn văn bản đã được truy xuất (từ tài liệu chính) ---")
    if retrieved_docs_main:
        for i, doc in enumerate(retrieved_docs_main):
            print(f"--- Tài liệu {i+1} ---")
            print(f"Nội dung: {doc.page_content[:200]}...")
            print("-" * 30)
    else:
        print("Không có tài liệu chính nào được truy xuất.")
    print("--------------------------------------------------\n")

    print("\n--- Orchestrator: Các facts đã được truy xuất (từ bộ nhớ dài hạn) ---")
    if retrieved_facts:
        for i, fact in enumerate(retrieved_facts):
            print(f"--- Fact {i+1} ---")
            print(f"Nội dung: {fact.page_content[:200]}...")
            print("-" * 30)
    else:
        print("Không có facts nào được truy xuất.")
    print("--------------------------------------------------\n")

    history_str = "\n".join([f"{m.type.capitalize()}: {m.content}" for m in current_chat_history_messages_list])

    final_prompt_input = {
        "chat_history": history_str,
        "document_context": document_context,
        "long_term_facts": long_term_facts_context,
        "question": user_query
    }
    
    print("Orchestrator: Đang gọi LLM để tạo câu trả lời...")
    llm_answer = "Xin lỗi, tôi gặp vấn đề khi tạo câu trả lời." 
    
    try:
        from langchain.chains import LLMChain # Thêm import này
        qa_chain_simple = LLMChain(llm=llm_orchestrator, prompt=CHAT_PROMPT)
        
        llm_response_dict = await qa_chain_simple.ainvoke(final_prompt_input)
        llm_answer = llm_response_dict['text'] 

        chat_memory.save_context({"input": user_query}, {"answer": llm_answer}) # << SỬA Ở ĐÂY

    except Exception as e:
        print(f"Orchestrator: Lỗi khi gọi LLM chính hoặc lưu memory: {e}")
        import traceback
        traceback.print_exc()
        llm_answer = "Xin lỗi, tôi gặp vấn đề khi tạo câu trả lời."

    print("Orchestrator: Đang trích xuất và lưu facts vào bộ nhớ dài hạn (cho các lượt sau)...")
    await personal_memory_agent.store_conversation_facts(user_query, llm_answer, current_chat_history_messages_list)

    return {"answer": llm_answer}