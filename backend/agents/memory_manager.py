import asyncio
from graph.state import GraphState
from tools.retrieval_tools import get_global_retriever_long_term
from utils.reranker import rerank_documents
from utils.llm_config import get_global_personal_memory_agent

async def retrieve_facts(state: GraphState) -> dict:
    print("---NODE: Long-Term Fact Retriever--- \n")

    user_question = state["original_question"]
    queries = state["expanded_queries"]
    retriever = get_global_retriever_long_term()

    tasks = [retriever.ainvoke(q) for q in queries]
    results_lists = await asyncio.gather(*tasks)

    unique_facts = {fact.page_content: fact for sublist in results_lists for fact in sublist}
    all_retrieved_facts = list(unique_facts.values())

    print(f"-> Đã truy xuất {len(all_retrieved_facts)} facts duy nhất.")

    if not all_retrieved_facts:
        print("Không có facts nào được truy xuất từ bộ nhớ dài hạn.")
        return {"retrieved_facts": ""}
    
    reranked_facts = rerank_documents(user_question, all_retrieved_facts, top_k=5)

    facts_context = "\n\n".join([fact.page_content for fact in reranked_facts])
    print(f"-> Đã rerank và tổng hợp ngữ cảnh facts.")

    return {"retrieved_facts": facts_context}

async def save_facts_to_memory(state: GraphState) -> dict:
    print("---NODE: Save Facts to Memory--- \n")

    memory_agent = get_global_personal_memory_agent()
    
    user_query = state["original_question"]
    answer = state["final_answer"]
    chat_history = state["chat_history"]

    if answer:
        await memory_agent.store_conversation_facts(user_query, answer, chat_history)
    else:
        print("Không có câu trả lời để lưu vào bộ nhớ.")

    return {}