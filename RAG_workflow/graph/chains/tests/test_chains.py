from dotenv import load_dotenv
load_dotenv() 


from graph.chains.retrieval_grader import GradeDocuments, retrival_grader
from ingestion import retriver

def test_retrival_grader_answer_yes() -> None:
    question = "agent Memory"
    docs = retriver.invoke(question)
    doc_text = docs[1].page_content
    
    res: GradeDocuments = retrival_grader.invoke(
        {"question": question, "document":doc_text}
    )
    
    assert res.binary_score == "yes"