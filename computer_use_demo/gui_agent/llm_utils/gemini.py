import os
import mimetypes
from typing import Any

import requests

from computer_use_demo.gui_agent.llm_utils.llm_utils import encode_image, is_image_path


def _build_image_part(image_path: str) -> dict[str, Any]:
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/png"
    return {
        "inline_data": {
            "mime_type": mime_type,
            "data": encode_image(image_path),
        }
    }


def run_gemini_interleaved(
    messages: list | str,
    system: str,
    llm: str,
    api_key: str,
    max_tokens: int = 256,
    temperature: float = 0,
):
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set")

    if isinstance(messages, str):
        iterable_messages: list[Any] = [messages]
    else:
        iterable_messages = messages

    contents: list[dict[str, Any]] = []

    for item in iterable_messages:
        parts: list[dict[str, Any]] = []

        if isinstance(item, dict):
            role = item.get("role", "user")
            for cnt in item.get("content", []):
                if isinstance(cnt, str):
                    if is_image_path(cnt):
                        parts.append(_build_image_part(cnt))
                    else:
                        parts.append({"text": cnt})
                elif isinstance(cnt, dict) and cnt.get("type") == "text":
                    parts.append({"text": cnt.get("text", "")})
            if parts:
                contents.append({"role": role, "parts": parts})
            continue

        if isinstance(item, str) and is_image_path(item):
            parts.append(_build_image_part(item))
        else:
            parts.append({"text": str(item)})

        contents.append({"role": "user", "parts": parts})

    payload: dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }

    if system:
        payload["systemInstruction"] = {
            "role": "system",
            "parts": [{"text": system}],
        }

    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{llm}:generateContent",
        params={"key": api_key},
        json=payload,
    )

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise ValueError(
            f"Gemini API request failed: {response.text}"
        ) from exc

    data = response.json()

    text_parts: list[str] = []
    try:
        candidate = data["candidates"][0]
        for part in candidate.get("content", {}).get("parts", []):
            if "text" in part:
                text_parts.append(part["text"])
    except (KeyError, IndexError, TypeError):
        pass

    text_response = "".join(text_parts) if text_parts else str(data)
    usage = data.get("usageMetadata", {})
    token_usage = usage.get("totalTokenCount") or usage.get("totalTokens") or 0

    return text_response, int(token_usage)
