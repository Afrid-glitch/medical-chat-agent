
import base64
import secrets
from pathlib import Path
from uuid import uuid4

from fastapi import (
    FastAPI,
    File,
    Header,
    HTTPException,
    UploadFile,
)
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field

from google import genai
from google.genai import types

from app.config import settings
from app.agent import graph


ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(
    title="Medical Agent Educational API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins.split(","),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

app.mount(
    "/static",
    StaticFiles(directory=ROOT / "static"),
    name="static",
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)

    history: list[dict[str, str]] = Field(
        default_factory=list,
        max_length=10,
    )

    image_data: str | None = None


# --------------------------------------------------
# GEMINI CLIENT
# --------------------------------------------------

def gemini_client():

    if not settings.gemini_api_key:
        raise HTTPException(
            status_code=503,
            detail="Gemini API key is not configured.",
        )

    return genai.Client(
        api_key=settings.gemini_api_key
    )


# --------------------------------------------------
# AUTHENTICATION
# --------------------------------------------------

def auth(authorization: str | None):

    # Optional authentication for local development.
    if not settings.api_token:
        return

    # Allow requests without a token during local development.
    if not authorization:
        return

    expected_token = f"Bearer {settings.api_token}"

    if not secrets.compare_digest(
        authorization,
        expected_token,
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid API token",
        )


# --------------------------------------------------
# IMAGE ANALYSIS USING GEMINI
# --------------------------------------------------

def inspect_image(data):

    if not data:
        return ""

    try:
        header, payload = data.split(",", 1)

        mime = header.split(";")[0].split(":")[1]

        raw = base64.b64decode(
            payload,
            validate=True,
        )

        if (
            mime not in (
                "image/jpeg",
                "image/png",
                "image/webp",
            )
            or len(raw) > 5_000_000
        ):
            raise ValueError(
                "Only JPEG/PNG/WebP images under 5MB are supported."
            )

    except (ValueError, IndexError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image: {exc}",
        )

    try:
        client = gemini_client()

        result = client.models.generate_content(
            model=settings.gemini_chat_model,
            contents=[
                (
                    "Describe only objectively visible "
                    "medical-relevant features. Do not diagnose, "
                    "estimate severity, or assert absence of illness. "
                    "If the image is unclear, say so."
                ),
                types.Part.from_bytes(
                    data=raw,
                    mime_type=mime,
                ),
            ],
        )

        return result.text or ""

    except HTTPException:
        raise

    except Exception as exc:
        print(
            "Gemini image analysis error:",
            repr(exc),
            flush=True,
        )

        raise HTTPException(
            status_code=502,
            detail="Gemini image analysis is temporarily unavailable.",
        )


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def home():

    return (
        ROOT / "templates" / "index.html"
    ).read_text(encoding="utf-8")


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/api/health")
def health():

    return {
        "status": "ok",
        "configured": bool(
            settings.gemini_api_key
            and settings.pinecone_api_key
        ),
        "tavily": bool(settings.tavily_api_key),
    }


# --------------------------------------------------
# CHAT ENDPOINT
# --------------------------------------------------

@app.post("/api/chat")
def chat(
    payload: ChatRequest,
    authorization: str | None = Header(default=None),
    conversation_id: str | None = Header(
        default=None,
        alias="X-Conversation-ID",
    ),
):

    auth(authorization)

    if (
        not settings.gemini_api_key
        or not settings.pinecone_api_key
    ):
        raise HTTPException(
            status_code=503,
            detail="Set GEMINI_API_KEY and PINECONE_API_KEY in .env",
        )

    history = [
        item
        for item in payload.history
        if (
            item.get("role") in ("user", "assistant")
            and isinstance(item.get("content"), str)
        )
    ][:10]

    thread_id = conversation_id or str(uuid4())

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    try:
        observation = inspect_image(
            payload.image_data
        )

        state = graph.invoke(
            {
                "question": payload.message,
                "history": history,
                "image_description": observation,
                "kb": [],
                "web": [],
                "answer": "",
                "route": "",
            },
            config=config,
        )

        return {
            "answer": state["answer"],

            "sources": (
                [
                    {
                        "type": "pdf",
                        "page": item["page"],
                        "score": item["score"],
                    }
                    for item in state["kb"]
                ]
                +
                [
                    {
                        "type": "web",
                        "url": item["url"],
                        "title": item["title"],
                    }
                    for item in state["web"]
                ]
            ),

            "tools": (
                ["pinecone"]
                + (
                    ["tavily"]
                    if (
                        state["route"] == "web"
                        and settings.tavily_api_key
                    )
                    else []
                )
                + (
                    ["vision"]
                    if observation
                    else []
                )
            ),

            "conversation_id": thread_id,
        }

    except HTTPException:
        raise

    except Exception as exc:
        print(
            "Chat error:",
            repr(exc),
            flush=True,
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "AI service failed. Check server logs, credentials, "
                "Pinecone index, and connectivity."
            ),
        )


# --------------------------------------------------
# VOICE TRANSCRIPTION USING GEMINI
# --------------------------------------------------

@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    authorization: str | None = Header(default=None),
):

    auth(authorization)

    # Read maximum 10 MB + 1 byte to detect oversized uploads.
    data = await file.read(10_000_001)

    if not data:
        raise HTTPException(
            status_code=400,
            detail="Empty audio file.",
        )

    if len(data) > 10_000_000:
        raise HTTPException(
            status_code=413,
            detail="Audio exceeds 10 MB.",
        )

    # Normalize browser MIME type.
    original_mime = (
        file.content_type or "audio/wav"
    ).split(";")[0].strip().lower()

    print(
        "Uploaded audio MIME type:",
        original_mime,
        flush=True,
    )

    # Browser MediaRecorder may identify audio recordings
    # using video MIME types because of their container.
    mime_mapping = {
        "video/webm": "audio/webm",
        "video/mp4": "audio/mp4",
    }

    mime = mime_mapping.get(
        original_mime,
        original_mime,
    )

    allowed_audio_types = {
        "audio/wav",
        "audio/x-wav",
        "audio/mpeg",
        "audio/mp3",
        "audio/aiff",
        "audio/aac",
        "audio/ogg",
        "audio/flac",
        "audio/webm",
        "audio/mp4",
        "audio/m4a",
        "audio/3gpp",
        "application/octet-stream",
    }

    if mime not in allowed_audio_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio format: {original_mime}",
        )

    try:
        client = gemini_client()

        result = client.models.generate_content(
            model=settings.gemini_chat_model,

            contents=[
                (
                    "Transcribe the spoken words in this audio "
                    "accurately. Return only the transcription. "
                    "Do not answer questions contained in the audio."
                ),

                types.Part.from_bytes(
                    data=data,
                    mime_type=mime,
                ),
            ],
        )

        return {
            "text": result.text or ""
        }

    except HTTPException:
        raise

    except Exception as exc:
        print(
            "Gemini transcription error:",
            repr(exc),
            flush=True,
        )

        raise HTTPException(
            status_code=502,
            detail="Gemini speech transcription is unavailable.",
        )


# --------------------------------------------------
# TEXT TO SPEECH USING GEMINI
# --------------------------------------------------

@app.post("/api/speak")
def speak(
    payload: dict,
    authorization: str | None = Header(default=None),
):

    auth(authorization)

    text = str(
        payload.get("text", "")
    )[:3000]

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Missing text",
        )

    try:
        client = gemini_client()

        interaction = client.interactions.create(
            model="gemini-3.1-flash-tts-preview",

            input=[
                {
                    "type": "user_input",
                    "content": [
                        {
                            "type": "text",
                            "text": text,
                        }
                    ],
                }
            ],

            response_format={
                "type": "audio"
            },

            generation_config={
                "speech_config": [
                    {
                        "voice": "Kore"
                    }
                ]
            },
        )

        if (
            not interaction.output_audio
            or not interaction.output_audio.data
        ):
            raise ValueError(
                "Gemini returned no audio data."
            )

        audio_bytes = base64.b64decode(
            interaction.output_audio.data
        )

        return Response(
            content=audio_bytes,
            media_type="audio/wav",
        )

    except HTTPException:
        raise

    except Exception as exc:
        print(
            "Gemini speech generation error:",
            repr(exc),
            flush=True,
        )

        raise HTTPException(
            status_code=502,
            detail="Gemini speech synthesis is unavailable.",
        )
