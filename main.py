"""
Production-ready Telegram bot: user sends text -> bot returns an MP4 video.

Required env vars:
  TELEGRAM_API_TOKEN     Bot token from BotFather
  WEBHOOK_BASE_URL       Public HTTPS base URL of this app

Recommended env vars:
  TELEGRAM_SECRET_TOKEN  Random string used to verify Telegram webhook requests
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
import imageio_ffmpeg
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

# ============================================================
# Configuration
# ============================================================
TELEGRAM_API_TOKEN = os.getenv("TELEGRAM_API_TOKEN", "").strip()
TELEGRAM_SECRET_TOKEN = os.getenv("TELEGRAM_SECRET_TOKEN", "").strip()
WEBHOOK_BASE_URL = os.getenv("WEBHOOK_BASE_URL", "").rstrip("/")
PUBLIC_WEBHOOK_PATH = "/webhook"

MAX_PROMPT_LENGTH = 500
TELEGRAM_MESSAGE_LIMIT = 4096
HTTP_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
MAX_RETRIES = 3
MAX_CONCURRENT_GENERATIONS = 4

if not TELEGRAM_API_TOKEN:
    raise RuntimeError("TELEGRAM_API_TOKEN environment variable is required")

TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_API_TOKEN}"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("telegram-bot")

_http_client: Optional[httpx.AsyncClient] = None
_generation_semaphore = asyncio.Semaphore(MAX_CONCURRENT_GENERATIONS)
_FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

# ============================================================
# Pydantic models
# ============================================================
class TelegramChat(BaseModel):
    id: int
    type: str
    username: Optional[str] = None
    first_name: Optional[str] = None

class TelegramUser(BaseModel):
    id: int
    is_bot: bool
    first_name: str
    username: Optional[str] = None

class TelegramMessage(BaseModel):
    message_id: int
    chat: TelegramChat
    from_: Optional[TelegramUser] = Field(default=None, alias="from")
    text: Optional[str] = None
    date: int
    model_config = {"populate_by_name": True}

class TelegramUpdate(BaseModel):
    update_id: int
    message: Optional[TelegramMessage] = None

# ============================================================
# Telegram API helpers
# ============================================================
async def _tg_request(
    method: str,
    *,
    json_payload: Optional[dict] = None,
    data: Optional[dict] = None,
    files: Optional[dict] = None,
) -> dict:
    assert _http_client is not None, "HTTP client not initialized"

    url = f"{TELEGRAM_API_URL}/{method}"
    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = await _http_client.post(
                url,
                json=json_payload,
                data=data,
                files=files,
            )
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", "2"))
                logger.warning("Rate limited on %s, waiting %ss", method, retry_after)
                await asyncio.sleep(retry_after)
                continue
            response.raise_for_status()
            body = response.json()
            if not body.get("ok"):
                raise RuntimeError(f"Telegram API error: {body}")
            return body
        except (httpx.HTTPError, RuntimeError) as exc:
            last_error = exc
            wait = 2 ** attempt
            logger.warning(
                "Attempt %s/%s failed for %s: %s (retry in %ss)",
                attempt, MAX_RETRIES, method, exc, wait,
            )
            await asyncio.sleep(wait)

    raise RuntimeError(f"Telegram API {method} failed after retries") from last_error

async def send_message(chat_id: int, text: str) -> None:
    if len(text) > TELEGRAM_MESSAGE_LIMIT:
        text = text[: TELEGRAM_MESSAGE_LIMIT - 3] + "..."
    await _tg_request(
        "sendMessage",
        json_payload={"chat_id": chat_id, "text": text},
    )

async def send_chat_action(chat_id: int, action: str = "upload_video") -> None:
    try:
        await _tg_request(
            "sendChatAction",
            json_payload={"chat_id": chat_id, "action": action},
        )
    except Exception:
        logger.debug("sendChatAction failed (ignored)", exc_info=True)

async def send_video(chat_id: int, video_path: str, caption: Optional[str] = None) -> None:
    with open(video_path, "rb") as video_file:
        files = {"video": (Path(video_path).name, video_file, "video/mp4")}
        payload = {"chat_id": str(chat_id), "supports_streaming": "true"}
        if caption:
            payload["caption"] = caption[:1024]
        await _tg_request("sendVideo", data=payload, files=files)

# ============================================================
# Video generation
# ============================================================
def _escape_ffmpeg_text(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
            .replace(":", "\\:")
            .replace("'", "\\'")
            .replace("%", "\\%")
            .replace("\n", " ")
    )

def generate_mp4_video(prompt: str) -> str:
    workdir = Path(tempfile.gettempdir()) / f"tg_video_{uuid.uuid4().hex}"
    workdir.mkdir(parents=True, exist_ok=True)

    output_path = workdir / "output.mp4"

    wrapped_lines = []
    for line in prompt.splitlines() or [prompt]:
        while len(line) > 40:
            wrapped_lines.append(line[:40])
            line = line[40:]
        wrapped_lines.append(line)
    display_text = "\n".join(wrapped_lines[:8])
    safe_text = _escape_ffmpeg_text(display_text)

    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    font_path = next((f for f in font_candidates if Path(f).exists()), None)

    drawtext = (
        f"drawtext=text='{safe_text}':"
        f"fontcolor=white:fontsize=42:"
        f"box=1:boxcolor=black@0.5:boxborderw=20:"
        f"x=(w-text_w)/2:y=(h-text_h)/2:"
        f"line_spacing=12"
    )
    if font_path:
        drawtext += f":fontfile='{font_path}'"

    vf = (
        "format=yuv420p,"
        "hue=H=2*PI*t/5:s=1.2,"
        f"{drawtext}"
    )

    cmd = [
        _FFMPEG_EXE,
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-f", "lavfi",
        "-i", "gradients=s=720x1280:c0=0x1e3a8a:c1=0x9333ea:c2=0xec4899:"
              "x0=0:y0=0:x1=720:y1=1280:d=5:speed=0.05",
        "-t", "5",
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0 or not output_path.exists():
            raise RuntimeError(
                f"ffmpeg failed (rc={result.returncode}): {result.stderr.strip()}"
            )
    except Exception:
        shutil.rmtree(workdir, ignore_errors=True)
        raise

    return str(output_path)

# ============================================================
# Background message processing
# ============================================================
async def process_message(chat_id: int, text: str) -> None:
    video_path: Optional[str] = None
    try:
        async with _generation_semaphore:
            heartbeat = asyncio.create_task(_heartbeat(chat_id))
            try:
                video_path = await asyncio.to_thread(generate_mp4_video, text)
            finally:
                heartbeat.cancel()
                try:
                    await heartbeat
                except asyncio.CancelledError:
                    pass

            await send_video(
                chat_id,
                video_path,
                caption=f"Video for: {text[:200]}",
            )
            logger.info("Video sent to chat %s", chat_id)

    except Exception:
        logger.exception("Failed to process message for chat %s", chat_id)
        try:
            await send_message(
                chat_id,
                "Sorry, something went wrong while generating the video. Please try again.",
            )
        except Exception:
            logger.exception("Failed to notify user about the error")
    finally:
        if video_path:
            _cleanup_video(video_path)

async def _heartbeat(chat_id: int) -> None:
    try:
        while True:
            await send_chat_action(chat_id, "upload_video")
            await asyncio.sleep(4)
    except asyncio.CancelledError:
        raise

def _cleanup_video(video_path: str) -> None:
    try:
        p = Path(video_path)
        if p.exists():
            p.unlink()
        parent = p.parent
        if parent.name.startswith("tg_video_"):
            shutil.rmtree(parent, ignore_errors=True)
    except Exception:
        logger.warning("Failed to cleanup %s", video_path, exc_info=True)

# ============================================================
# FastAPI lifespan
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http_client
    _http_client = httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True)
    logger.info("HTTP client started")

    if WEBHOOK_BASE_URL:
        webhook_url = f"{WEBHOOK_BASE_URL}{PUBLIC_WEBHOOK_PATH}"
        try:
            payload: dict = {
                "url": webhook_url,
                "allowed_updates": ["message"],
                "drop_pending_updates": True,
            }
            if TELEGRAM_SECRET_TOKEN:
                payload["secret_token"] = TELEGRAM_SECRET_TOKEN
            await _tg_request("setWebhook", json_payload=payload)
            logger.info("Webhook set to %s", webhook_url)
        except Exception:
            logger.exception("Failed to set webhook")
    else:
            logger.warning("WEBHOOK_BASE_URL not set - webhook not registered")

    try:
        yield
    finally:
        if _http_client is not None:
            await _http_client.aclose()
        _http_client = None
        logger.info("HTTP client closed")

app = FastAPI(lifespan=lifespan, title="Telegram Video Bot")

# ============================================================
# Routes
# ============================================================
@app.get("/")
async def root() -> dict:
    return {"status": "ok", "service": "telegram-video-bot"}

@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True}

@app.post(PUBLIC_WEBHOOK_PATH)
async def telegram_webhook(
    update: TelegramUpdate,
    background_tasks: BackgroundTasks,
    request: Request,
) -> dict:
    if TELEGRAM_SECRET_TOKEN:
        header_token = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
        if header_token != TELEGRAM_SECRET_TOKEN:
            logger.warning("Rejected webhook: bad secret token")
            raise HTTPException(status_code=403, detail="Forbidden")

    message = update.message
    if message is None:
        return {"ok": True}

    chat_id = message.chat.id
    text = (message.text or "").strip()

    if not text:
        background_tasks.add_task(
            send_message, chat_id, "Please send a text prompt to generate a video."
        )
        return {"ok": True}

    if text.startswith("/start") or text.startswith("/help"):
        background_tasks.add_task(
            send_message,
            chat_id,
            "Send me any text prompt and I'll generate a short video for you.",
        )
        return {"ok": True}

    if len(text) > MAX_PROMPT_LENGTH:
        background_tasks.add_task(
            send_message,
            chat_id,
            f"Prompt too long ({len(text)} chars). Max is {MAX_PROMPT_LENGTH}.",
        )
        return {"ok": True}

    background_tasks.add_task(
        send_message, chat_id, f"Generating video for: {text[:120]}"
    )
    background_tasks.add_task(process_message, chat_id, text)

    return {"ok": True}
