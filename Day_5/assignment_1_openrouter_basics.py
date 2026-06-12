from __future__ import annotations
import dotenv
dotenv.load_dotenv()
import os
from typing import Any


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o-mini"
MODEL_CHOICES = [
    "openai/gpt-4o-mini",
    "google/gemini-2.0-flash-001",
    "anthropic/claude-3.5-haiku",
]
APP_TITLE = "Day 3 Vision Chat"

def resolve_api_key(api_key: str | None = None) -> str:
    """Return an API key from the function argument or environment."""
    # TODO 1: return the stripped api_key if provided.
    # TODO 2: otherwise read OPENROUTER_API_KEY from the environment.
    api_key = (api_key or os.getenv("OPENROUTER_API_KEY") or "").strip()
    # TODO 3: raise ValueError if neither exists.
    if not api_key:
        raise ValueError("API key is required")
    return api_key


def create_openrouter_client(api_key: str) -> Any:
    """Create an OpenAI SDK client configured for OpenRouter."""
    from openai import OpenAI

    # TODO 4: return an openai client using openrouter's base url
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "http://localhost:7860",
            "X-Title": APP_TITLE,
        },
    )
    raise NotImplementedError


def build_text_messages(
    user_prompt: str,
    history: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """Convert prior chat history plus the new prompt into OpenRouter messages."""
    messages: list[dict[str, str]] = []

    for item in history or []:
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            # TODO 5: append the previous message as {"role": role, "content": content.strip()}
            messages.append({"role": role, "content": content.strip()})
            pass

    # TODO 6: append the latest user prompt.
    messages.append(
        {
            "role": "user",
            "content": messages[-1]["content"] if messages else user_prompt.strip(),
        }
    )
    return messages


def ask_text_model(
    prompt: str,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    history: list[dict[str, Any]] | None = None,
) -> str:
    """Send one text-only chat request and return the assistant response."""
    client = create_openrouter_client(resolve_api_key(api_key))

    # TODO 7: call client.chat.completions.create with:
    # model=model
    # messages=build_text_messages(prompt, history)
    # extra_body={"provider": {"data_collection": "deny"}}

    response = client.chat.completions.create(
            model=model,
            messages=build_text_messages(prompt, history),
            extra_body={"provider": {"data_collection": "deny"}},
        )


    # TODO 8: return response.choices[0].message.content or an empty string.
    if response.choices:
        return response.choices[0].message.content or ""
    raise NotImplementedError


if __name__ == "__main__":
    print(ask_text_model("Explain Gradio in one sentence."))
