"""
main.py — сканер контактов со скриншотов (2GIS, визитки, карточки организаций и т.п.).

Логика:
  1. FastAPI отдаёт static/index.html на "/".
  2. Фронт присылает скриншоты батчами (до 100 штук за раз) на POST /api/process.
  3. Бэкенд для каждой картинки:
       - считает sha256, точные дубликаты внутри батча не гоняет через ИИ повторно;
       - уменьшает изображение (чтобы не жечь токены и память);
       - шлёт в LLM с vision (через OpenRouter) с промптом на извлечение
         только контактных полей: имя/организация, телефон, email, адрес;
     все запросы к модели идут с ограниченным параллелизмом.
  4. Дедупликация по смыслу (совпадение телефона/email после нормализации,
     объединение самой полной записи) делается на фронте, после того как
     собраны все батчи — так пагинация, сортировка и склейка видны сразу
     и не требуют серверного состояния между запросами.

Запуск:
    uvicorn main:app --reload

API-ключ OpenRouter вводится пользователем прямо на странице, серверу
самому ключ не нужен и нигде не сохраняется.
"""

import asyncio
import hashlib
import io
import json
import time

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# ════════════════════════════════════════════════════════════════
# КОНСТАНТЫ
# ════════════════════════════════════════════════════════════════

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_REFERER = "https://github.com/drk-supersonic/contact-extractor"
MODEL = "google/gemini-3.1-flash-lite"

MAX_FILES_PER_BATCH = 100      # ограничение на один запрос (по договорённости)
MAX_CONCURRENT_REQUESTS = 8    # сколько картинок гоним в ИИ параллельно
MAX_IMAGE_SIDE = 1400          # ресайз перед отправкой, чтобы не жечь токены/память
JPEG_QUALITY = 85

SYSTEM_PROMPT = """Ты обрабатываешь скриншот с карточкой организации (например, 2GIS), \
визиткой, контактом из телефонной книги или похожим источником.

Извлеки СТРОГО следующие поля, если они есть на изображении:
- name: полное имя человека (максимально полное ФИО) ИЛИ название организации, \
если это карточка компании, а не человека
- phone: номер(а) телефона; если их несколько, перечисли через "; "
- email: email адрес, если есть
- address: физический адрес, если есть

Больше ничего извлекать не нужно. Часы работы, сайт, соцсети, мессенджеры, \
отзывы, вакансии, похожие организации, рекламные блоки и любой другой текст \
полностью игнорируй.

Если какого-то из четырёх полей на изображении нет, поставь пустую строку "".

Ответь СТРОГО валидным JSON, без пояснений, без markdown-разметки, \
без текста до или после, ровно такой структуры:
{"name": "", "phone": "", "email": "", "address": ""}
"""

# ════════════════════════════════════════════════════════════════
# ОБРАБОТКА ИЗОБРАЖЕНИЙ
# ════════════════════════════════════════════════════════════════

def prepare_image_b64(raw: bytes) -> str:
    """Уменьшает картинку и кодирует в base64 JPEG, чтобы сэкономить токены и память."""
    img = Image.open(io.BytesIO(raw))
    img = img.convert("RGB")
    w, h = img.size
    scale = min(1.0, MAX_IMAGE_SIDE / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    import base64
    return base64.b64encode(buf.getvalue()).decode("ascii")


def call_llm(image_b64: str, api_key: str, _retry: int = 0) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": OPENROUTER_REFERER,
        "X-Title": "Contact Extractor",
    }
    payload = {
        "model": MODEL,
        "max_tokens": 400,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                    {"type": "text", "text": "Извлеки контактные поля из этого изображения."},
                ],
            },
        ],
    }

    try:
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        if _retry < 2:
            time.sleep(1.5 * (_retry + 1))
            return call_llm(image_b64, api_key, _retry + 1)
        return {"name": "", "phone": "", "email": "", "address": "", "error": f"OpenRouter недоступен: {e}"}

    try:
        content = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError):
        return {"name": "", "phone": "", "email": "", "address": "", "error": f"Неожиданный ответ: {data}"}

    # На случай если модель всё же обернула JSON в ```json ... ```
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return {"name": "", "phone": "", "email": "", "address": "", "error": "Не удалось распарсить JSON от модели"}

    return {
        "name": str(parsed.get("name", "") or "").strip(),
        "phone": str(parsed.get("phone", "") or "").strip(),
        "email": str(parsed.get("email", "") or "").strip(),
        "address": str(parsed.get("address", "") or "").strip(),
    }


async def process_one(raw: bytes, filename: str, api_key: str, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:
        try:
            b64 = await asyncio.to_thread(prepare_image_b64, raw)
        except Exception as e:
            return {"file": filename, "name": "", "phone": "", "email": "", "address": "", "error": f"Не удалось открыть изображение: {e}"}
        result = await asyncio.to_thread(call_llm, b64, api_key)
        result["file"] = filename
        return result


# ════════════════════════════════════════════════════════════════
# FASTAPI
# ════════════════════════════════════════════════════════════════

app = FastAPI(title="Contact Extractor")


@app.exception_handler(Exception)
async def catch_all(request, exc):
    return JSONResponse(status_code=500, content={"detail": f"Внутренняя ошибка: {exc}"})


@app.post("/api/process")
async def process_batch(api_key: str = Form(...), files: list[UploadFile] = File(...)):
    key = (api_key or "").strip()
    if not key:
        raise HTTPException(status_code=400, detail="Не передан API ключ OpenRouter")
    if not files:
        raise HTTPException(status_code=400, detail="Нет файлов")
    if len(files) > MAX_FILES_PER_BATCH:
        raise HTTPException(
            status_code=400,
            detail=f"Максимум {MAX_FILES_PER_BATCH} файлов за один батч, прислано {len(files)}",
        )

    # Точные дубликаты (побайтово одинаковые файлы) внутри батча гоняем через ИИ один раз
    hash_to_result: dict[str, dict] = {}
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def handle(upload: UploadFile) -> dict:
        raw = await upload.read()
        digest = hashlib.sha256(raw).hexdigest()
        if digest in hash_to_result:
            cached = dict(hash_to_result[digest])
            cached["file"] = upload.filename
            cached["duplicate_of_batch"] = True
            return cached
        result = await process_one(raw, upload.filename, key, semaphore)
        hash_to_result[digest] = result
        return result

    results = await asyncio.gather(*(handle(f) for f in files))
    return {"results": results}


# Отдаём фронт (одна страница)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
