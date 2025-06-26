from sentence_transformers.cross_encoder import CrossEncoder
from typing import List
from langchain.docstore.document import Document

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("Mô hình Cross-Encoder đã sẵn sàng.")

def rerank_documents(query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
    if not documents:
        return []
    
    pairs = [(query, doc.page_content) for doc in documents]

    scores = cross_encoder.predict(pairs)

    doc_with_scores = list(zip(documents, scores))
    doc_with_scores.sort(key = lambda x:x[1], reverse=True)

    rerank_docs = [doc for doc, score in doc_with_scores[:top_k]]

    print(f"Reranker: Đã sắp xếp lại từ {len(documents)} docs xuống còn {len(rerank_docs)} docs tốt nhất.")
    
    return rerank_docs