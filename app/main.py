# app/main.py
import os
import uuid
from typing import Optional, Dict, Any

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from .agent.semantic_scraper import extract_offers
from .agent.agent import run_question
from .agent.models import MonitorRequest, AskRequest


API_KEY = os.getenv("AGENT_API_KEY", "local-dev-key")


def verify_api_key(api_key: Optional[str]):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


api = FastAPI(title="Semantic Price Scraper + Tools (LangChain)")


@api.get("/health")
def health(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    verify_api_key(x_api_key)
    return {
        "status": "ok"
    }


@api.get("/session/new")
def create_session(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    verify_api_key(x_api_key)
    """Создает новый ID сессии"""
    session_id = str(uuid.uuid4())
    return {
        "session_id": session_id,
        "message": "Используйте этот ID в хедере X-Session-Id для группировки запросов"
    }


@api.post("/monitor")
def monitor(
    req: MonitorRequest,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_session_id: Optional[str] = Header(
        None, description="ID сессии для трейсинга"),
    x_user_id: Optional[str] = Header(None, description="ID пользователя")
):
    verify_api_key(x_api_key)
    data = extract_offers(req.url, req.max_items)
    return {"url": req.url, "offers": [o.dict() for o in data.offers]}


@api.post("/ask")
def ask(
    req: AskRequest,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_session_id: Optional[str] = Header(
        None, description="ID сессии для трейсинга"),
    x_user_id: Optional[str] = Header(None, description="ID пользователя")
) -> Dict[str, Any]:
    verify_api_key(x_api_key)
    # Используем session_id из хедера или генерируем новый
    session_id = x_session_id or str(uuid.uuid4())

    # Вызываем агента с сессией
    result = run_question(
        question=req.question,
        urls=req.urls,
        max_items=req.max_items,
        session_id=session_id,
        user_id=x_user_id
    )

    # Добавляем session_id в ответ
    if isinstance(result, dict):
        result["_session_id"] = session_id
        if x_user_id:
            result["_user_id"] = x_user_id

        if "tools_used" not in result:
            result["tools_used"] = []

    return result
