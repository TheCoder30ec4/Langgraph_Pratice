from typing import Any,Dict
from graph.state import GraphState
from ingestion import retriver


def retrieve(state: GraphState)->Dict[str,Any]:
    print("---RETRIEVE---")
    question = state["question"]
    
    documents = retriver.invoke(question)
    return {"documents": documents, "question":question}