from langgraph.graph import StateGraph, END
from .state import GraphState

from agents.query_analyzer import analyze_query
from agents.document_retriever import retrieve_documents
from agents.memory_manager import retrieve_facts, save_facts_to_memory
from agents.synthesis_agent import synthesize_answer

def create_chatbot_graph():
    workflow = StateGraph(GraphState)

    # 1. Định nghĩa các node
    workflow.add_node("analyze_query", analyze_query)
    workflow.add_node("document_retriever", retrieve_documents)
    workflow.add_node("fact_retriever", retrieve_facts)
    workflow.add_node("answer_synthesizer", synthesize_answer)
    workflow.add_node("memory_saver", save_facts_to_memory)

    # 2. Dịnh nghĩa các cạnh
    # Đây là một luồng tuần tự, dễ hiểu và debug.
    # Trong thực tế, bạn có thể chạy document_retriever và fact_retriever song song.

    workflow.set_entry_point("analyze_query")
    workflow.add_edge("analyze_query", "document_retriever")
    workflow.add_edge("document_retriever", "fact_retriever")
    workflow.add_edge("fact_retriever", "answer_synthesizer")
    workflow.add_edge("answer_synthesizer", "memory_saver")
    workflow.add_edge("memory_saver", END)

    print(" Đồ thị Chatbot đã được biên dịch thành công!")
    app = workflow.compile()
    return app