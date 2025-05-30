from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from rag_engine import answer_query

app = FastAPI()

class QueryRequest(BaseModel):
    question: str
    embedding_model: Optional[str] = "multi-qa-MiniLM-L6-cos-v1"
    temperature: Optional[float] = 0.4
    top_p: Optional[float] = 0.95
    similarity_threshold: Optional[float] = 0.85
    chunk_size: Optional[int] = 500
    chunk_overlap: Optional[int] = 50

@app.post("/ask")
async def ask_question(request: QueryRequest):
    answer = answer_query(
        question=request.question,
        embedding_model=request.embedding_model,
        temperature=request.temperature,
        top_p=request.top_p,
        similarity_threshold=request.similarity_threshold,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap
    )
    return {"answer": answer}
