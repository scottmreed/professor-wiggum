"""Provider-aware LLM helpers for OpenAI, Gemini, and OpenRouter models."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

try:  # pragma: no cover - optional import
    from openai import OpenAI
except ImportError:  # pragma: no cover - fallback handled at runtime
    OpenAI = None  # type: ignore[assignment]

try:  # pragma: no cover - optional import
    from anthropic import Anthropic
except ImportError:  # pragma: no cover - fallback handled at runtime
    Anthropic = None  # type: ignore[assignment]

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def is_gemini_model(model_name: Optional[str]) -> bool:
    """Return True when the model name targets the Gemini API."""
    return bool(model_name) and model_name.lower().startswith("gemini-")


def is_openrouter_model(model_name: Optional[str]) -> bool:
    """Return True when the model should be routed through OpenRouter.

    OpenRouter model IDs contain a slash (e.g. ``anthropic/claude-opus-4.6``,
    ``allenai/olmo-3.1-32b-instruct``).  We also check the catalog provider
    field when the registry is available.
    """
    if not model_name:
        return False
    try:
        from mechanistic_agent.model_registry import get_model_provider
        provider = get_model_provider(model_name)
        if provider:
            return provider == "openrouter"
    except Exception:
        pass
    if "/" in model_name:
        return True
    return False


def is_anthropic_model(model_name: Optional[str]) -> bool:
    """Return True when the model should be routed through Anthropic directly.

    We check the catalog provider field when the registry is available.
    """
    if not model_name:
        return False
    try:
        from mechanistic_agent.model_registry import get_model_provider
        return get_model_provider(model_name) == "anthropic"
    except Exception:
        return False


def is_agent_bridge_model(model_name: Optional[str]) -> bool:
    """Return True when the model is served by the keyless agent bridge.

    The agent bridge lets an external agent/subagent stand in for a hosted model
    (no API key) while receiving exactly the inputs a hosted model would. It is
    selected by the catalog ``provider`` field being ``"agent_bridge"``.
    """
    if not model_name:
        return False
    try:
        from mechanistic_agent.model_registry import get_model_provider
        return get_model_provider(model_name) == "agent_bridge"
    except Exception:
        return False


def _resolve_google_api_key(user_key: Optional[str] = None) -> Optional[str]:
    if user_key:
        return user_key
    for key in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "VERTEX_API_KEY", "VERTTEX_API_KEY"):
        value = os.getenv(key)
        if value:
            return value
    return None


def _resolve_openrouter_api_key(user_key: Optional[str] = None) -> Optional[str]:
    if user_key:
        return user_key
    return os.getenv("OPENROUTER_API_KEY")


def _resolve_openai_api_key(user_key: Optional[str] = None) -> Optional[str]:
    if user_key:
        return user_key
    return os.getenv("OPENAI_API_KEY")


def _resolve_anthropic_api_key(user_key: Optional[str] = None) -> Optional[str]:
    if user_key:
        return user_key
    return os.getenv("ANTHROPIC_API_KEY")


def get_model_api_key(model_name: Optional[str], user_key: Optional[str] = None) -> Optional[str]:
    """Return the provider API key for the given model name."""
    if is_agent_bridge_model(model_name):
        # The bridge is answered by a local agent/subagent, not a hosted API, so
        # there is no credential to resolve. Return a non-empty sentinel so the
        # many callers that gate on "is a key configured?" treat the bridge as
        # available. This value is never sent anywhere or used as a credential.
        return user_key or "agent-bridge-local"
    if is_gemini_model(model_name):
        return _resolve_google_api_key(user_key)
    if is_anthropic_model(model_name):
        return _resolve_anthropic_api_key(user_key)
    if is_openrouter_model(model_name):
        return _resolve_openrouter_api_key(user_key)
    return _resolve_openai_api_key(user_key)


def get_provider_label(model_name: Optional[str]) -> str:
    if is_agent_bridge_model(model_name):
        return "Agent Bridge"
    if is_gemini_model(model_name):
        return "Gemini"
    if is_anthropic_model(model_name):
        return "Anthropic"
    if is_openrouter_model(model_name):
        return "OpenRouter"
    return "OpenAI"


def supports_structured_outputs(model_name: Optional[str]) -> bool:
    """Check if the model supports structured outputs (JSON schema enforcement)."""
    if not model_name:
        return False
    model_lower = model_name.lower()
    return (
        model_lower.startswith("gpt-5")
        or model_lower.startswith("gpt-4o")
        or model_lower == "gpt-4o-mini"
    )


def extract_text_content(message: Any) -> Optional[str]:
    """Normalize message content into plain text."""
    if isinstance(message, str):
        return message.strip() or None
    if isinstance(message, dict):
        content = message.get("content", "")
        return str(content).strip() or None
    content = getattr(message, "content", "")
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and "text" in part
        ).strip() or None
    return str(content).strip() or None


def serialise_chat_messages(messages: Any) -> list[Dict[str, str]]:
    """Normalise chat messages into the ``[{role, content}]`` list sent to providers.

    This is exactly the message representation ``_OpenAIChatAdapter.invoke`` sends
    to a hosted chat-completions API. The keyless agent-bridge provider reuses
    this helper so that an external agent/subagent standing in for the model
    receives precisely the same message view a keyed provider would — no more and
    no less. Keeping a single implementation guarantees the two paths cannot drift.
    """
    serialised: list[Dict[str, str]] = []
    for message in messages or []:
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        role = "user"
        if isinstance(message, dict):
            maybe_role = str(message.get("role") or "").strip().lower()
            if maybe_role in {"system", "user", "assistant", "tool"}:
                role = maybe_role
        else:
            name = message.__class__.__name__.lower()
            if "system" in name:
                role = "system"
            elif "human" in name or "user" in name:
                role = "user"
            elif "assistant" in name or "ai" in name:
                role = "assistant"
        serialised.append({"role": role, "content": str(content or "")})
    return serialised


class _SimpleMessage:
    def __init__(self, content: str, tool_calls=None, usage=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.usage = usage  # Dict[str, int] or None — raw token counts from provider


class _OpenAIChatAdapter:
    def __init__(
        self,
        *,
        model: str,
        temperature: Optional[float],
        timeout: Optional[float],
        model_kwargs: Optional[Dict[str, Any]],
        api_key: str,
        base_url: Optional[str] = None,
    ) -> None:
        if OpenAI is None:
            raise RuntimeError("OpenAI runtime unavailable. Install `openai`.")
        self._model = model
        self._temperature = temperature
        self._timeout = timeout
        self._model_kwargs = model_kwargs or {}
        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = OpenAI(**client_kwargs)

    def invoke(
        self,
        messages: Any,
        config: Any = None,  # noqa: ARG002
        *,
        tools: Any = None,
        tool_choice: Any = None,
    ) -> Any:
        serialised = serialise_chat_messages(messages)

        params: Dict[str, Any] = {"model": self._model, "messages": serialised}
        if self._temperature is not None:
            params["temperature"] = self._temperature
        params.update(self._model_kwargs)
        if tools is not None:
            params["tools"] = tools
        if tool_choice is not None:
            params["tool_choice"] = tool_choice
        try:
            response = self._client.chat.completions.create(**params, timeout=self._timeout)
        except TypeError as exc:
            # OpenAI SDK validates top-level kwargs. OpenRouter-specific controls
            # (e.g. `thinking`, `effort`) must be forwarded via extra_body.
            if not self._model_kwargs or "unexpected keyword argument" not in str(exc):
                raise
            retry_params = dict(params)
            for key in self._model_kwargs:
                retry_params.pop(key, None)
            extra_body = dict(retry_params.get("extra_body") or {})
            extra_body.update(self._model_kwargs)
            retry_params["extra_body"] = extra_body
            response = self._client.chat.completions.create(**retry_params, timeout=self._timeout)
        msg = response.choices[0].message
        text = msg.content or ""
        raw_tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                raw_tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                })
        usage_data = None
        if hasattr(response, "usage") and response.usage is not None:
            u = response.usage
            usage_data = {
                "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
                "total_tokens": getattr(u, "total_tokens", 0) or 0,
                "prompt_cache_hit_tokens": getattr(u, "prompt_cache_hit_tokens", 0) or 0,
                "prompt_cache_miss_tokens": getattr(u, "prompt_cache_miss_tokens", 0) or 0,
            }
        return _SimpleMessage(text, tool_calls=raw_tool_calls, usage=usage_data)


def _openai_messages_to_anthropic(messages: Any) -> tuple[Optional[str], list[dict[str, Any]]]:
    system_prompt: Optional[str] = None
    converted: list[dict[str, Any]] = []

    for message in messages or []:
        role = "user"
        content = getattr(message, "content", None)
        if isinstance(message, dict):
            content = message.get("content")
            maybe_role = str(message.get("role") or "").strip().lower()
            if maybe_role in {"system", "user", "assistant"}:
                role = maybe_role
        else:
            name = message.__class__.__name__.lower()
            if "system" in name:
                role = "system"
            elif "assistant" in name or "ai" in name:
                role = "assistant"

        text = str(content or "")
        if role == "system":
            if system_prompt is None:
                system_prompt = text
            else:
                system_prompt = f"{system_prompt}\n\n{text}"
            continue
        converted.append({"role": role, "content": text})

    return system_prompt, converted


def _openai_tools_to_anthropic(tools: Any) -> Optional[list[dict[str, Any]]]:
    converted: list[dict[str, Any]] = []
    for tool in tools or []:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        fn = tool.get("function", {})
        converted.append(
            {
                "name": str(fn.get("name") or ""),
                "description": str(fn.get("description") or ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return converted or None


def _openai_tool_choice_to_anthropic(tool_choice: Any) -> Optional[dict[str, Any]]:
    if tool_choice is None:
        return {"type": "auto"}
    if isinstance(tool_choice, str):
        if tool_choice == "required":
            return {"type": "any"}
        if tool_choice == "auto":
            return {"type": "auto"}
        if tool_choice == "none":
            return None
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        name = str(tool_choice.get("function", {}).get("name") or "")
        if name:
            return {"type": "tool", "name": name}
    return {"type": "auto"}


class _AnthropicChatAdapter:
    def __init__(
        self,
        *,
        model: str,
        temperature: Optional[float],
        timeout: Optional[float],  # noqa: ARG002
        model_kwargs: Optional[Dict[str, Any]],
        api_key: str,
    ) -> None:
        if Anthropic is None:
            raise RuntimeError("Anthropic runtime unavailable. Install `anthropic`.")
        self._model = model
        self._temperature = temperature
        self._model_kwargs = dict(model_kwargs or {})
        self._client = Anthropic(api_key=api_key)

    def invoke(
        self,
        messages: Any,
        config: Any = None,  # noqa: ARG002
        *,
        tools: Any = None,
        tool_choice: Any = None,
    ) -> Any:
        import json as _json

        system_prompt, converted_messages = _openai_messages_to_anthropic(messages)
        params: Dict[str, Any] = {
            "model": self._model,
            "messages": converted_messages,
            "max_tokens": int(self._model_kwargs.pop("max_tokens", 4096)),
        }
        if self._temperature is not None:
            params["temperature"] = self._temperature
        params.update(self._model_kwargs)
        if system_prompt:
            params["system"] = system_prompt
        anthropic_tools = _openai_tools_to_anthropic(tools)
        if anthropic_tools:
            params["tools"] = anthropic_tools
            anthropic_tool_choice = _openai_tool_choice_to_anthropic(tool_choice)
            if anthropic_tool_choice:
                params["tool_choice"] = anthropic_tool_choice

        response = self._client.messages.create(**params)
        text_parts: list[str] = []
        raw_tool_calls: list[dict[str, Any]] = []
        for block in getattr(response, "content", []) or []:
            block_type = getattr(block, "type", "")
            if block_type == "text":
                text_parts.append(str(getattr(block, "text", "") or ""))
            elif block_type == "tool_use":
                arguments = getattr(block, "input", {})
                raw_tool_calls.append(
                    {
                        "id": getattr(block, "id", ""),
                        "name": getattr(block, "name", ""),
                        "arguments": (
                            _json.dumps(arguments)
                            if isinstance(arguments, (dict, list))
                            else str(arguments or "")
                        ),
                    }
                )

        usage_data = None
        usage = getattr(response, "usage", None)
        if usage is not None:
            prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "output_tokens", 0) or 0)
            cache_hits = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
            usage_data = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "prompt_cache_hit_tokens": cache_hits,
                "prompt_cache_miss_tokens": 0,
                "cache_creation_tokens": int(
                    getattr(usage, "cache_creation_input_tokens", 0) or 0
                ),
            }

        return _SimpleMessage(
            "".join(text_parts),
            tool_calls=raw_tool_calls,
            usage=usage_data,
        )


def _openai_schema_to_gemini(oai_schema: Dict[str, Any]) -> Any:
    """Recursively convert an OpenAI JSON schema dict to a google.genai Schema."""
    from google.genai.types import Schema, Type  # type: ignore[import]

    _type_map = {
        "string": Type.STRING,
        "number": Type.NUMBER,
        "integer": Type.INTEGER,
        "boolean": Type.BOOLEAN,
        "array": Type.ARRAY,
        "object": Type.OBJECT,
    }

    oai_type = oai_schema.get("type", "object")
    g_type = _type_map.get(str(oai_type).lower(), Type.STRING)

    kwargs: Dict[str, Any] = {"type": g_type}

    if "description" in oai_schema:
        kwargs["description"] = oai_schema["description"]

    if "enum" in oai_schema:
        # Gemini tool schemas currently support enum constraints reliably for STRING.
        # For numeric types, preserve intent via descriptions and omit enum to avoid
        # INVALID_ARGUMENT errors.
        if str(oai_type).lower() == "string":
            kwargs["enum"] = [str(v) for v in oai_schema["enum"]]

    if oai_type == "object" and "properties" in oai_schema:
        kwargs["properties"] = {
            k: _openai_schema_to_gemini(v)
            for k, v in oai_schema["properties"].items()
        }
        if "required" in oai_schema:
            kwargs["required"] = list(oai_schema["required"])

    if oai_type == "array" and "items" in oai_schema:
        kwargs["items"] = _openai_schema_to_gemini(oai_schema["items"])

    return Schema(**kwargs)


def _openai_tools_to_gemini(oai_tools: Any) -> Any:
    """Convert a list of OpenAI-format tool dicts to a Gemini Tool object list."""
    from google.genai.types import FunctionDeclaration, Tool  # type: ignore[import]

    declarations = []
    for tool in oai_tools or []:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        fn = tool.get("function", {})
        name = fn.get("name", "")
        description = fn.get("description", "")
        parameters_schema = _openai_schema_to_gemini(fn.get("parameters", {"type": "object", "properties": {}}))
        declarations.append(
            FunctionDeclaration(
                name=name,
                description=description,
                parameters=parameters_schema,
            )
        )
    return [Tool(function_declarations=declarations)]


class _GeminiChatAdapter:
    def __init__(
        self,
        *,
        model: str,
        temperature: Optional[float],
        model_kwargs: Optional[Dict[str, Any]],
        api_key: str,
    ) -> None:
        try:
            from google import genai as _genai  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "Gemini support requires `google-genai`. Install it with: pip install google-genai"
            ) from exc
        self._client = _genai.Client(api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._model_kwargs = model_kwargs or {}

    def invoke(
        self,
        messages: Any,
        config: Any = None,  # noqa: ARG002
        *,
        tools: Any = None,
        tool_choice: Any = None,
    ) -> Any:
        from google.genai.types import (  # type: ignore[import]
            FunctionCallingConfig,
            FunctionCallingConfigMode,
            GenerateContentConfig,
            ToolConfig,
        )

        parts = []
        for message in messages or []:
            role = "user"
            content = getattr(message, "content", None)
            if content is None and isinstance(message, dict):
                content = message.get("content")
                maybe_role = str(message.get("role") or "").strip().lower()
                if maybe_role:
                    role = maybe_role
            elif message is not None and not isinstance(message, dict):
                name = message.__class__.__name__.lower()
                if "system" in name:
                    role = "system"
                elif "assistant" in name or "ai" in name:
                    role = "assistant"
            parts.append(f"[{role}] {str(content or '').strip()}")

        prompt = "\n\n".join(parts)

        gen_config_kwargs: Dict[str, Any] = {}
        if self._temperature is not None:
            gen_config_kwargs["temperature"] = self._temperature
        gen_config_kwargs.update(self._model_kwargs)

        if tools is not None:
            gemini_tools = _openai_tools_to_gemini(tools)
            # Extract the required function name from tool_choice if provided.
            allowed_names = None
            if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
                fn_name = tool_choice.get("function", {}).get("name")
                if fn_name:
                    allowed_names = [fn_name]
            gen_config_kwargs["tools"] = gemini_tools
            gen_config_kwargs["tool_config"] = ToolConfig(
                function_calling_config=FunctionCallingConfig(
                    mode=FunctionCallingConfigMode.ANY,
                    allowed_function_names=allowed_names,
                )
            )

        gen_config = GenerateContentConfig(**gen_config_kwargs) if gen_config_kwargs else None

        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=gen_config,
        )

        # Extract token usage from Gemini response metadata.
        usage_data = None
        um = getattr(response, "usage_metadata", None)
        if um is not None:
            usage_data = {
                "prompt_tokens": getattr(um, "prompt_token_count", 0) or 0,
                "completion_tokens": getattr(um, "candidates_token_count", 0) or 0,
                "total_tokens": getattr(um, "total_token_count", 0) or 0,
                "prompt_cache_hit_tokens": getattr(um, "cached_content_token_count", 0) or 0,
                "prompt_cache_miss_tokens": 0,
            }

        # Extract structured function call result when tools were requested.
        if tools is not None:
            fn_calls = getattr(response, "function_calls", None)
            if fn_calls:
                import json as _json
                fc = fn_calls[0]
                return _SimpleMessage("", tool_calls=[{
                    "id": getattr(fc, "id", fc.name),
                    "name": fc.name,
                    "arguments": _json.dumps(dict(fc.args)),
                }], usage=usage_data)

        # Text fallback — handle both .text shortcut and candidates structure.
        text = getattr(response, "text", None)
        if not isinstance(text, str):
            text = ""
            candidates = getattr(response, "candidates", None) or []
            if candidates:
                cand = candidates[0]
                content_obj = getattr(cand, "content", None)
                if content_obj:
                    for part in getattr(content_obj, "parts", []):
                        if hasattr(part, "text") and part.text:
                            text += part.text
        return _SimpleMessage(text, usage=usage_data)


def get_chat_model(
    model_name: str,
    *,
    temperature: Optional[float] = None,
    timeout: Optional[float] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    user_api_key: Optional[str] = None,
) -> Any:
    """Return a provider-specific chat model wrapper.

    Routes to OpenRouter for Claude and OLMo models, Gemini for Google models,
    and OpenAI for all others.  Pass ``user_api_key`` to override env vars.

    Models whose catalog provider is ``"agent_bridge"`` route to the keyless
    agent bridge, which forwards the request to an external agent/subagent
    instead of a hosted API and therefore needs no API key.
    """
    if is_agent_bridge_model(model_name):
        from .agent_bridge import AgentBridgeAdapter

        return AgentBridgeAdapter(
            model=model_name,
            temperature=temperature,
            timeout=timeout,
            model_kwargs=model_kwargs,
        )

    if is_gemini_model(model_name):
        api_key = _resolve_google_api_key(user_api_key)
        if not api_key:
            raise RuntimeError(
                "Gemini API key not configured. Set GOOGLE_API_KEY or GEMINI_API_KEY."
            )
        return _GeminiChatAdapter(
            model=model_name,
            temperature=temperature,
            model_kwargs=model_kwargs,
            api_key=api_key,
        )

    if is_openrouter_model(model_name):
        api_key = _resolve_openrouter_api_key(user_api_key)
        if not api_key:
            raise RuntimeError(
                "OpenRouter API key not configured. Set OPENROUTER_API_KEY."
            )
        return _OpenAIChatAdapter(
            model=model_name,
            temperature=temperature,
            timeout=timeout,
            model_kwargs=model_kwargs,
            api_key=api_key,
            base_url=_OPENROUTER_BASE_URL,
        )

    if is_anthropic_model(model_name):
        api_key = _resolve_anthropic_api_key(user_api_key)
        if not api_key:
            raise RuntimeError(
                "Anthropic API key not configured. Set ANTHROPIC_API_KEY."
            )
        return _AnthropicChatAdapter(
            model=model_name,
            temperature=temperature,
            timeout=timeout,
            model_kwargs=model_kwargs,
            api_key=api_key,
        )

    api_key = _resolve_openai_api_key(user_api_key)
    if not api_key:
        raise RuntimeError("OpenAI API key not configured. Set OPENAI_API_KEY.")

    return _OpenAIChatAdapter(
        model=model_name,
        temperature=temperature,
        timeout=timeout,
        model_kwargs=model_kwargs,
        api_key=api_key,
    )


def adapter_supports_forced_tools(model_name: Optional[str]) -> bool:
    """Return True when our adapter implementation supports forced tool calling.

    Gemini models use ``_GeminiChatAdapter`` with ``FunctionCallingConfigMode.ANY``.
    OpenAI and OpenRouter models use ``_OpenAIChatAdapter`` with ``tool_choice``.
    OLMo (``allenai/olmo-*``) has ``supports_tools: false`` in the registry and
    falls back to prompted text responses.
    """
    if not model_name:
        return False
    if is_gemini_model(model_name):
        return True  # _GeminiChatAdapter is wired for forced function calling
    try:
        from .model_registry import model_supports_tools
        return model_supports_tools(model_name)
    except Exception:
        return True
