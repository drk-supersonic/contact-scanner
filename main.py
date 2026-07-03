"""
main.py — тестовое задание: веб-инструмент "один вопрос через ИИ".

Логика:
  1. FastAPI отдаёт static/index.html на "/".
  2. Фронт шлёт ответ пользователя на POST /api/respond.
  3. Бэкенд передаёт ответ в LLM (через OpenRouter) с системным промптом
     "исследователя" и возвращает реакцию модели.

Запуск:
    export OPENROUTER_API_KEY="sk-or-..."
    uvicorn main:app --reload
"""

import os
import time

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ════════════════════════════════════════════════════════════════
# КОНСТАНТЫ
# ════════════════════════════════════════════════════════════════

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_REFERER = "https://github.com/osmo/interview-bot"  # можно заменить на свой репозиторий
MODEL = "google/gemini-2.5-flash"

QUESTION = "Расскажите, как вы выбирали последний онлайн-курс?"

SYSTEM_PROMPT = (
    "Ты — исследователь, проводящий короткое интервью. "
    "Респонденту задан вопрос: «Расскажите, как вы выбирали последний онлайн-курс?». "
    "Оцени его ответ.\n\n"
    "Если ответ поверхностный (общие слова, нет конкретики — критериев выбора, "
    "источников информации, сравнения вариантов) — задай РОВНО ОДИН уточняющий вопрос, "
    "который поможет раскрыть детали. Не задавай больше одного вопроса.\n\n"
    "Если ответ подробный (есть конкретика: критерии, источники, сравнение, причины) — "
    "поблагодари респондента и заверши разговор, не задавая вопросов.\n\n"
    "Отвечай коротко, живым разговорным языком, без лишних вступлений."
)

# Серверный ключ — необязательный фолбэк (например, для твоих собственных
# тестов через ask_api_key/run.py). Основной сценарий теперь — ключ приходит
# от клиента в каждом запросе.
SERVER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()

# ════════════════════════════════════════════════════════════════
# ВЫЗОВ LLM (с ретраями, по аналогии с call_llm из tz-drawing-analyzer)
# ════════════════════════════════════════════════════════════════

def call_llm(user_answer: str, api_key: str | None = None, _retry: int = 0) -> str:
    key = (api_key or "").strip() or SERVER_API_KEY
    if not key:
        raise HTTPException(
            status_code=400,
            detail="Не передан API ключ. Введите его в поле на странице.",
        )

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": OPENROUTER_REFERER,
        "X-Title": "Interview Bot — Test Task",
    }
    payload = {
        "model": MODEL,
        "max_tokens": 300,
        "temperature": 0.4,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_answer},
        ],
    }

    try:
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except (requests.exceptions.RequestException, KeyError, IndexError) as e:
        if _retry < 2:
            time.sleep(1.5 * (_retry + 1))
            return call_llm(user_answer, api_key, _retry + 1)
        raise HTTPException(status_code=502, detail=f"LLM недоступна: {e}") from e


# ════════════════════════════════════════════════════════════════
# FASTAPI
# ════════════════════════════════════════════════════════════════

app = FastAPI(title="Interview Bot")


class AnswerIn(BaseModel):
    answer: str
    api_key: str | None = None


class ReplyOut(BaseModel):
    reply: str


@app.get("/api/question")
def get_question():
    return {"question": QUESTION}


@app.post("/api/respond", response_model=ReplyOut)
def respond(payload: AnswerIn):
    answer = payload.answer.strip()
    if not answer:
        raise HTTPException(status_code=400, detail="Пустой ответ")
    reply = call_llm(answer, payload.api_key)
    return {"reply": reply}


# Отдаём фронт (одна страница)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
