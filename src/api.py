
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from main import initialize_rag, answer_question

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the RAG pipeline once when the API starts.
    vector_store, generator = initialize_rag()

    app.state.vector_store = vector_store
    app.state.generator = generator

    yield

    # Add cleanup here if your components require it.


app = FastAPI(
    title="RAG Chatbot API",
    description="Ask questions about your PDF documents.",
    version="1.0.0",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=2000,
        description="Question to ask about the documents",
    )


class ChatResponse(BaseModel):
    question: str
    answer: str


@app.get("/")
def health_check():
    return {"status": "API is running"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, http_request: Request):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty",
        )

    try:
        answer = answer_question(
            question,
            http_request.app.state.vector_store,
            http_request.app.state.generator,
        )

        return {
            "question": question,
            "answer": str(answer),
        }

    except Exception:
        # Log the actual exception server-side in production.
        raise HTTPException(
            status_code=500,
            detail="An error occurred while generating the answer.",
        )
