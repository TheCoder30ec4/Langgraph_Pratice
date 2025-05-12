from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

llm = ChatGroq(model="llama3-70b-8192")

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents"""
    
    binary_score: str = Field(description="Document are relevant to the question, 'yes' or 'no' ")
    
    
structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """ You are a grader assesing relevance of a retrieved document to a user question.\n
            If the document contains keyword(s) or sematic meaning related to the quesation, grade it as relevant.\n
            Give a binary socre 'yes' or 'no' score to indicate wheather the document is relevant to the question.
"""

grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system),
        ("human","Retrived document: \n\n {document} \n\n User question: {question}")
    ]
)

retrival_grader = grader_prompt | structured_llm_grader