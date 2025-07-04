from typing import List, Optional, TypedDict
from langchain_core.messages import BaseMessage

class GraphState(TypedDict):
    """
    Represents the state of the graph, including nodes and edges.
    """
    original_question: str
    chat_history: List[BaseMessage]
    expanded_queries: Optional[List[str]] = None
    retrieved_docs: Optional[str] = None
    retrieved_facts: Optional[str] = None
    final_answer: Optional[str] = None