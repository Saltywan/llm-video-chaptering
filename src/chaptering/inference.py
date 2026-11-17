"""Final-answer-only prompting and sequential OpenAI-compatible smoke requests."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit
from typing import Any


PROMPT_VERSION = "indexed-outline-v1"


def strip_qwen_terminal_tokens(text: str) -> str:
    """Remove only native trailing turn/end-of-text framing, never JSON content."""
    result = text.strip()
    while True:
        for token in ("<|im_end|>", "<|endoftext|>"):
            if result.endswith(token):
                result = result[:-len(token)].rstrip()
                break
        else:
            return result


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Do not forward authenticated requests to redirected origins."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def completion_url(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("Endpoint must be a credential-free HTTPS base URL")
    return endpoint.rstrip("/") + "/chat/completions"


def build_prompt(record: dict[str, Any]) -> str:
    """Read only the inference allowlist, even when passed a labeled record."""
    sentences = record["sentences"]
    if not sentences or any(not isinstance(s, str) or not s.strip() for s in sentences):
        raise ValueError("Expected a nonempty list of nonempty sentences")
    indexed = "\n".join(f"[{i}] {sentence}" for i, sentence in enumerate(sentences))
    return (
        "Divide the transcript into contiguous topical chapters and give each a concise, "
        "specific title supported by its text. Use only the transcript. Do not rewrite it. "
        "The bracketed numbers are global sentence indices. A chapter start is the index "
        "of its first sentence. Include start 0; use strictly increasing integer starts "
        f"from 0 through {len(sentences) - 1}. The chapters must cover the whole transcript. "
        "Choose chapter count from the content. In your final answer return only one JSON "
        'object with exactly this structure: {"chapters":[{"start":0,"title":"..."}]}. '
        "Do not put commentary or a Markdown fence around the final JSON.\n\n"
        f"TRANSCRIPT\n{indexed}"
    )


def final_content(message: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Separate native reasoning channels; never search reasoning text for JSON."""
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("No final text content")
    reasoning = message.get("reasoning_content", message.get("reasoning"))
    info = {"separate_reasoning_observed": isinstance(reasoning, str),
            "reasoning_characters": len(reasoning) if isinstance(reasoning, str) else None,
            "reasoning_source": "api_field" if isinstance(reasoning, str) else None}
    content = content.strip()
    if content.startswith("<think>"):
        end = content.find("</think>")
        if end < 0:
            raise ValueError("Unclosed reasoning block; no final answer")
        info.update(reasoning_characters=end - len("<think>"),
                    reasoning_source="explicit_think_block")
        content = content[end + len("</think>"):].strip()
    if not content:
        raise ValueError("Empty final answer")
    return content, info


def endpoint_prediction(record: dict[str, Any], model: dict[str, Any], settings: dict[str, Any],
                        api_key: str, run_id: str, timeout: float = 180) -> dict[str, Any]:
    from chaptering.evaluation import parse_outline

    result = {"schema_version": 1, "video_id": record["video_id"], "run_id": run_id,
              "status": "error", "chapters": [], "error": None, "usage": {},
              "model_id": model["model_id"], "prompt_version": PROMPT_VERSION,
              "purpose": "endpoint compatibility smoke; not primary ZS/FT evidence"}
    payload = {"model": model["model_id"],
               "messages": [{"role": "user", "content": build_prompt(record)}],
               "max_tokens": settings["max_tokens"], "temperature": settings["temperature"],
               "top_p": settings["top_p"], "seed": settings["seed"],
               "chat_template_kwargs": {"enable_thinking": model["enable_thinking"]}}
    result["settings"] = {key: value for key, value in payload.items()
                          if key not in {"messages", "model"}}
    request = urllib.request.Request(completion_url(model["endpoint"]),
                                     data=json.dumps(payload).encode(),
                                     headers={"Authorization": f"Bearer {api_key}",
                                              "Content-Type": "application/json",
                                              "User-Agent": "llm-video-chaptering/1.0"})
    started = time.perf_counter()
    try:
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=timeout) as response:
            data = json.load(response)
        choice = data["choices"][0]
        result["usage"] = data.get("usage", {})
        result["finish_reason"] = choice.get("finish_reason")
        result["server_model"] = data.get("model")
        result["response_id"] = data.get("id")
        if choice.get("finish_reason") != "stop":
            result.update(status="truncated" if choice.get("finish_reason") == "length" else "error",
                          error="Generation did not finish normally")
        else:
            text, reasoning_info = final_content(choice["message"])
            result.update(reasoning_info)
            try:
                result["chapters"] = parse_outline(text, len(record["sentences"]))
                result["status"] = "ok"
            except (ValueError, TypeError, KeyError) as exc:
                result.update(status="invalid", error=type(exc).__name__ + ": " + str(exc))
    except urllib.error.HTTPError as exc:
        result["error"] = f"HTTP {exc.code}"  # Never echo headers or credential-bearing responses.
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        result["error"] = type(exc).__name__  # No raw request/response dump.
    result["wall_seconds"] = time.perf_counter() - started
    return result
