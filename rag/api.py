from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from pydantic import BaseModel
import time
from rag.rag_pipe import rag_app, logger , LLMException


app_api = FastAPI(title="Self-RAG API")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    verdict: str | None = None
    issup: str | None = None
    need_retrieval: bool
    rewrite_tries: int
    web_docs: list[object]

@app_api.get("/health")
async def health():
    return {"status": "ok"}

@app_api.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    start = time.monotonic()
    try:
        res = await rag_app.ainvoke({
            "question": req.question,
            "docs": [],
            "good_docs": [],
            "verdict": "",
            "need_retrieval": False,
            "strips": [],
            "kept_strips": [],
            "refined_context": "",
            "web_query": "",
            "web_docs": [],
            "answer": "",
            "rewrite_tries": 0,
        })
    except LLMException as e:
        logger.error(f"[/query] LLM failure: {e}")
        raise HTTPException(status_code=503, detail="AI service temporarily unavailable. Please try again.")
    except Exception as e:
        logger.error(f"[/query] unexpected failure: {e}")
        raise HTTPException(status_code=500, detail="Something went wrong processing your question.")

    print("type of res :", type(res))

    return QueryResponse(
        answer=res["answer"],
        verdict=res.get("verdict"),
        issup=res.get("issup"),
        need_retrieval = res.get("need_retrieval"),
        rewrite_tries = res.get("rewrite_tries"),
        web_docs = res.get("web_docs",[]),
    )
