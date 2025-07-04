from langchain.chains import LLMChain
from graph.state import GraphState
from utils.llm_config import get_global_llm
from utils.prompts import CHAT_PROMPT

async def synthesize_answer(state: GraphState) -> dict:
    print("---NODE: Synthesis Agent--- \n")

    print(f"[CÂU HỎI GỐC]: {state['original_question']}")
    print(f"[FACTS ĐÃ RERANK]:\n{state['retrieved_facts']}\n")
    print(f"[DOCS ĐÃ RERANK]:\n{state['retrieved_docs']}\n")

    history_str = "\n".join([f"{m.type.capitalize()}: {m.content}" for m in state["chat_history"]])

    final_prompt_input = {
        "chat_history": history_str,
        "document_context": state["retrieved_docs"],
        "long_term_facts": state["retrieved_facts"],
        "question": state["original_question"]
    }

    llm = get_global_llm()
    qa_chain = LLMChain(
        llm=llm,
        prompt=CHAT_PROMPT,
        verbose=True
    )

    llm_response = await qa_chain.ainvoke(final_prompt_input)
    llm_answer = llm_response['text'].strip()

    print(f"-> Câu trả lời được tạo: {llm_answer[:100]}...")
    
    return {"final_answer": llm_answer}