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

SYSTEM_PROMPT_FIRST = (
    "Ты — исследователь, проводящий короткое интервью. "
    "Респонденту задан вопрос: «Расскажите, как вы выбирали последний онлайн-курс?». "
    "Оцени его ответ.\n\n"
    "Если ответ поверхностный (общие слова, нет конкретики — критериев выбора, "
    "источников информации, сравнения вариантов) — задай РОВНО ОДИН уточняющий вопрос, "
    "который поможет раскрыть детали. Не задавай больше одного вопроса.\n\n"
    "Если ответ подробный (есть конкретика: критерии, источники, сравнение, причины) — "
    "поблагодари респондента и заверши разговор, не задавая вопросов.\n\n"
    "Отвечай коротко, живым разговорным языком, без лишних вступлений. "
    "Не используй тире (—, -) для соединения частей предложения. "
    "Пиши простыми короткими фразами через точку или запятую, без "
    "парантетических вставок через тире."
)

# Промпт для второго (финального) ответа. Лимит на уточняющий вопрос уже
# исчерпан, поэтому модели запрещено задавать ещё один вопрос — независимо
# от того, насколько подробным получился ответ. Это закрывает лазейку:
# без этого модель могла бы решить, что второй ответ всё ещё "поверхностный",
# и снова сформулировать реплику как вопрос.
SYSTEM_PROMPT_FINAL = (
    "Ты — исследователь, проводящий короткое интервью. Респонденту был задан "
    "основной вопрос про выбор последнего онлайн-курса, затем — ОДИН уточняющий "
    "вопрос, на который он только что ответил.\n\n"
    "Лимит уточняющих вопросов исчерпан. Ты ОБЯЗАН завершить разговор: "
    "поблагодари респондента за ответ. Ни при каких обстоятельствах не задавай "
    "больше никаких вопросов, даже если ответ снова кажется неполным.\n\n"
    "Отвечай коротко, живым разговорным языком, без лишних вступлений. "
    "Не используй тире (—, -) для соединения частей предложения. "
    "Пиши простыми короткими фразами через точку или запятую, без "
    "парантетических вставок через тире."
)

# ════════════════════════════════════════════════════════════════
# ВЫЗОВ LLM (с ретраями, по аналогии с call_llm из tz-drawing-analyzer)
# ════════════════════════════════════════════════════════════════

def call_llm(user_answer: str, api_key: str, is_final: bool, _retry: int = 0) -> str:
    key = (api_key or "").strip()
    if not key:
        raise HTTPException(
            status_code=400,
            detail="Не передан API ключ. Введите его в поле на странице.",
        )

    system_prompt = SYSTEM_PROMPT_FINAL if is_final else SYSTEM_PROMPT_FIRST

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
            {"role": "system", "content": system_prompt},
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
            return call_llm(user_answer, api_key, is_final, _retry + 1)
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
    round: int = 0  # 0 — первый ответ на основной вопрос, 1+ — ответ на уточнение


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
    # Уточняющий вопрос допустим только один раз: если это уже ответ на
    # уточнение (round >= 1), лимит исчерпан и разговор обязан завершиться.
    is_final = payload.round >= 1
    reply = call_llm(answer, payload.api_key, is_final)
    return {"reply": reply}


# Отдаём фронт (одна страница)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
