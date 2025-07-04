import asyncio
from graph.state import GraphState
from tools.retrieval_tools import get_global_retriever_main_docs
from utils.reranker import rerank_documents

async def retrieve_documents(state: GraphState) -> dict:
    print("---NODE: Document Retriever--- \n")

    user_query = state["original_question"]
    queries = state["expanded_queries"]
    retriever = get_global_retriever_main_docs()

    tasks = [retriever.ainvoke(q) for q in queries]
    results_lists = await asyncio.gather(*tasks)

    unique_docs = {doc.page_content: doc for sublist in results_lists for doc in sublist}
    all_retrieved_docs = list(unique_docs.values())

    print(f"-> Retrieved {len(all_retrieved_docs)} unique documents based on queries: {queries}")

    reranked_docs = rerank_documents(user_query, all_retrieved_docs, top_k=4)

    document_context = "\n\n".join([doc.page_content for doc in reranked_docs]) 
    print(f"-> Đã rerank và tổng hợp ngữ cảnh tài liệu.")
    
    return { "retrieved_docs": document_context}
