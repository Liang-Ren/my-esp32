"""
OpenAIClient — wraps the OpenAI Responses API (openai >= 1.66).

Conversation threading via previous_response_id:
  - After a successful response the response ID is stored per user_id.
  - On the next call for the same user, only the new user message is sent
    together with previous_response_id. OpenAI maintains history server-side.
  - On any API error the stored ID is cleared and the full history (passed by
    the caller) is sent instead — transparent fallback.
  - user_id="" disables threading (every call is stateless).

Public method:
    generateResponse(input_messages, context, user_id, timeout) -> (text, usage)
"""
import logging
import time
from openai import AsyncOpenAI, APITimeoutError, APIConnectionError, APIStatusError

log = logging.getLogger("xiaozhi.src")

FALLBACK_RESPONSE = "Sorry, I had trouble reaching the AI service."
DEFAULT_TIMEOUT = 30.0
MAX_OUTPUT_TOKENS = 200


class OpenAIClient:
    def __init__(self, api_key: str, model: str):
        self._client = AsyncOpenAI(api_key=api_key)
        self.model = model
        # Maps user_id → most recent response ID for conversation threading.
        # Persists for the lifetime of this object (module-level singleton in
        # gateway.py) so threads survive device reconnects.
        self._prev_ids: dict[str, str] = {}

    async def generateResponse(
        self,
        input_messages: list[dict],
        context: str = "",
        user_id: str = "",
        timeout: float = DEFAULT_TIMEOUT,
    ) -> tuple[str, dict]:
        """
        Call the Responses API and return (response_text, usage_dict).

        Args:
            input_messages: Full conversation turns from build_input().
                            Used verbatim on the first turn or after a fallback.
                            On subsequent turns only the last user message is
                            sent (history lives server-side at OpenAI).
            context:        System instructions string (from prompt_builder).
                            Always sent so memory context is fresh each turn.
            user_id:        Scopes conversation threading. "" = stateless.
            timeout:        Hard deadline in seconds for the HTTP request.

        Returns:
            (text, usage)  text is the model reply; usage contains latency_ms,
                           token counts, model name, response_id, and an "error"
                           key on failure.
        """
        prev_id = self._prev_ids.get(user_id) if user_id else None

        if prev_id:
            # Threaded path: send only the new user message.
            new_msg = _last_user_message(input_messages)
            text, usage = await self._call(
                [new_msg], context, prev_id, timeout
            )
            if usage.get("error"):
                # Stale or expired ID — clear it and retry with full history.
                log.warning(
                    "  [llm] previous_response_id %s invalid (%s) — retrying",
                    prev_id[:16], usage["error"],
                )
                self._prev_ids.pop(user_id, None)
                text, usage = await self._call(input_messages, context, None, timeout)
            else:
                usage["threaded"] = True
        else:
            # First turn for this user, or threading disabled.
            text, usage = await self._call(input_messages, context, None, timeout)

        # Store the new ID for the next turn.
        if user_id and not usage.get("error"):
            new_id = usage.get("response_id", "")
            if new_id:
                self._prev_ids[user_id] = new_id

        return text, usage

    def clear_thread(self, user_id: str) -> None:
        """Discard the stored response ID for a user (e.g. on session reset)."""
        self._prev_ids.pop(user_id, None)

    # ── Internal ────────────────────────────────────────────────────────────────

    async def _call(
        self,
        input_messages: list[dict],
        context: str,
        previous_response_id: str | None,
        timeout: float,
    ) -> tuple[str, dict]:
        t0 = time.time()
        try:
            kwargs: dict = dict(
                model=self.model,
                instructions=context or None,
                input=input_messages,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                temperature=0.7,
                timeout=timeout,
            )
            if previous_response_id:
                kwargs["previous_response_id"] = previous_response_id

            resp = await self._client.responses.create(**kwargs)
            text = resp.output_text.strip()
            latency_ms = round((time.time() - t0) * 1000)
            usage = {
                "api": "responses",
                "model": resp.model,
                "input_tokens": resp.usage.input_tokens if resp.usage else 0,
                "output_tokens": resp.usage.output_tokens if resp.usage else 0,
                "total_tokens": resp.usage.total_tokens if resp.usage else 0,
                "latency_ms": latency_ms,
                "response_id": resp.id,
            }
            return text, usage

        except APITimeoutError:
            return FALLBACK_RESPONSE, {
                "error": f"timeout after {timeout}s",
                "latency_ms": round((time.time() - t0) * 1000),
            }
        except APIConnectionError as exc:
            return FALLBACK_RESPONSE, {
                "error": f"connection error: {exc}",
                "latency_ms": round((time.time() - t0) * 1000),
            }
        except APIStatusError as exc:
            return FALLBACK_RESPONSE, {
                "error": f"API {exc.status_code}: {exc.message}",
                "latency_ms": round((time.time() - t0) * 1000),
            }
        except Exception as exc:
            return FALLBACK_RESPONSE, {
                "error": str(exc),
                "latency_ms": round((time.time() - t0) * 1000),
            }


def _last_user_message(messages: list[dict]) -> dict:
    """Return the last message with role=='user', or the final message."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg
    return messages[-1]
