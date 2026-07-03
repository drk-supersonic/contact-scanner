"""
main.py — тестовое задание: веб-инструмент "один вопрос через ИИ".

Логика:
  1. FastAPI отдаёт static/index.html на "/".
  2. Фронт шлёт ответ пользователя на POST /api/respond.
  3. Бэкенд передаёт ответ в LLM (через OpenRouter) с системным промптом
     "исследователя" и возвращает реакцию модели.

Запуск:
    uvicorn main:app --reload

API-ключ OpenRouter вводится пользователем прямо на странице, серверу
самому ключ не нужен.
"""

import time

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ════════════════════════════════════════════════════════════════
# КОНСТАНТЫ
# ════════════════════════════════════════════════════════════════

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_REFERER = "https://github.com/osmo/interview-bot"  # можно заменить на свой репозиторий
MODEL = "openai/gpt-5-mini"

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

# ════════════════════════════════════════════════════════════════
# ВЫЗОВ LLM (с ретраями, по аналогии с call_llm из tz-drawing-analyzer)
# ════════════════════════════════════════════════════════════════

def call_llm(user_answer: str, api_key: str, _retry: int = 0) -> str:
    key = (api_key or "").strip()
    if not key:
        raise HTTPException(
            status_code=400,
            detail="Не передан API ключ. Введите его в поле на странице.",
        )

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": OPENROUTER_REFERER,
        "X-Title": "Interview Bot",
    }
    payload = {
        "model": MODEL,
        "max_tokens": 600,
        "temperature": 0.4,
        "reasoning_effort": "minimal",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_answer},
        ],
    }

    try:
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        if _retry < 2:
            time.sleep(1.5 * (_retry + 1))
            return call_llm(user_answer, api_key, _retry + 1)
        # Пытаемся вытащить тело ответа от OpenRouter — там обычно есть причина
        body = ""
        if getattr(e, "response", None) is not None:
            body = e.response.text[:300]
        raise HTTPException(
            status_code=502,
            detail=f"OpenRouter недоступен: {e}. {body}".strip(),
        ) from e
    except ValueError as e:
        # resp.json() не смог распарсить ответ (не-JSON тело)
        raise HTTPException(
            status_code=502,
            detail=f"OpenRouter вернул нечитаемый ответ: {e}",
        ) from e

    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as e:
        # Модель отклонила запрос или вернула ошибку в теле 200-ответа
        raise HTTPException(
            status_code=502,
            detail=f"Неожиданный формат ответа от OpenRouter: {data}",
        ) from e


# ════════════════════════════════════════════════════════════════
# FASTAPI
# ════════════════════════════════════════════════════════════════

app = FastAPI(title="Interview Bot")


@app.exception_handler(Exception)
async def catch_all(request, exc):
    return JSONResponse(status_code=500, content={"detail": f"Внутренняя ошибка: {exc}"})


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
