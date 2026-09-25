"""One place that builds chat models.

OpenAI directly, or any OpenAI-compatible endpoint (OpenRouter for DeepSeek) picked by
HIPPO_PROVIDER / HIPPO_BASE_URL. Everything else in hippo (`bind_tools`, `usage_metadata`,
`ainvoke`) is provider-agnostic, so this is the only file that knows the difference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Optional attribution headers OpenRouter shows on its dashboard; harmless elsewhere.
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/JuneC7020/hippo",
    "X-Title": "hippo",
}


@dataclass
class LLMConfig:
    provider: str  # openai | openrouter | custom
    api_key: str
    base_url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def key_env_name(self) -> str:
        return "OPENROUTER_API_KEY" if self.provider == "openrouter" else "OPENAI_API_KEY"


def resolve_llm(settings: Any, model: str | None = None) -> LLMConfig:
    """Decide provider, key and base_url from settings (and optionally the model slug).

    auto: an explicit HIPPO_BASE_URL wins; else OpenRouter when its key is the only one set or
    when the model looks like "vendor/model" and an OpenRouter key exists; else OpenAI.
    """
    provider = (getattr(settings, "hippo_provider", "auto") or "auto").lower()
    base_url = (getattr(settings, "hippo_base_url", "") or "").strip() or None
    openai_key = getattr(settings, "openai_api_key", "") or ""
    or_key = getattr(settings, "openrouter_api_key", "") or ""
    model = model or getattr(settings, "hippo_model", "") or ""

    if provider == "auto":
        if base_url and base_url.rstrip("/") != OPENROUTER_BASE_URL:
            provider = "custom"
        elif or_key and (not openai_key or "/" in model or base_url):
            provider = "openrouter"
        else:
            provider = "openai"

    if provider == "openrouter":
        return LLMConfig(
            "openrouter", or_key, base_url or OPENROUTER_BASE_URL, dict(OPENROUTER_HEADERS)
        )
    if provider == "custom":
        return LLMConfig("custom", or_key or openai_key, base_url)
    return LLMConfig("openai", openai_key, base_url)


def make_chat(
    model: str,
    *,
    api_key: str,
    base_url: str | None = None,
    temperature: float = 0,
    headers: dict[str, str] | None = None,
    **kwargs: Any,
):
    """ChatOpenAI against OpenAI or an OpenAI-compatible base_url.

    `extra_body={"usage": {"include": True}}` asks OpenRouter to return the billed cost in the
    usage block; OpenAI ignores unknown keys, so it is safe to send always when a base_url is set.
    """
    from langchain_openai import ChatOpenAI

    if headers is None and base_url and base_url.rstrip("/") == OPENROUTER_BASE_URL:
        headers = dict(OPENROUTER_HEADERS)
    params: dict[str, Any] = {"model": model, "temperature": temperature}
    if api_key:  # else langchain falls back to OPENAI_API_KEY in the environment
        params["api_key"] = api_key
    if base_url:
        params["base_url"] = base_url
        params["extra_body"] = {"usage": {"include": True}}
    if headers:
        params["default_headers"] = headers
    params.update(kwargs)
    return ChatOpenAI(**params)


def chat_from_settings(settings: Any, model: str | None = None, **kwargs: Any):
    cfg = resolve_llm(settings, model)
    return make_chat(
        model or settings.hippo_model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        headers=cfg.headers or None,
        **kwargs,
    )
