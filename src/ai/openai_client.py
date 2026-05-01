"""
OpenAIClient — wraps the OpenAI Responses API (openai >= 1.66).

Public method:
    generateResponse(input_messages, context, timeout) -> (text, usage_dict)

`context` is passed as `instructions=` to the Responses API.
`input_messages` is passed as `input=`.
"""
import time
from openai import AsyncOpenAI, APITimeoutError, APIConnectionError, APIStatusError

FALLBACK_RESPONSE = "Sorry, I had trouble reaching the AI service."
DEFAULT_TIMEOUT = 30.0   # seconds
MAX_OUTPUT_TOKENS = 200  # keeps responses concise for voice


class OpenAIClient:
    def __init__(self, api_key: str, model: str):
        self._client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def generateResponse(
        self,
        input_messages: list[dict],
        context: str = "",
        timeout: float = DEFAULT_TIMEOUT,
    ) -> tuple[str, dict]:
        """
        Call the Responses API and return (response_text, usage_dict).

        Args:
            input_messages: Conversation turns [{role, content}, ...].
            context:        System instructions (from prompt_builder).
            timeout:        Seconds before the request is aborted.

        Returns:
            text:  Model reply, suitable for TTS.
            usage: Dict with latency_ms, token counts, model, response_id.
                   Contains an "error" key on failure.
        """
        t0 = time.time()
        try:
            resp = await self._client.responses.create(
                model=self.model,
                instructions=context or None,
                input=input_messages,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                temperature=0.7,
                timeout=timeout,
            )
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
            latency_ms = round((time.time() - t0) * 1000)
            return FALLBACK_RESPONSE, {
                "error": f"timeout after {timeout}s",
                "latency_ms": latency_ms,
            }
        except APIConnectionError as exc:
            latency_ms = round((time.time() - t0) * 1000)
            return FALLBACK_RESPONSE, {
                "error": f"connection error: {exc}",
                "latency_ms": latency_ms,
            }
        except APIStatusError as exc:
            latency_ms = round((time.time() - t0) * 1000)
            return FALLBACK_RESPONSE, {
                "error": f"API {exc.status_code}: {exc.message}",
                "latency_ms": latency_ms,
            }
        except Exception as exc:
            latency_ms = round((time.time() - t0) * 1000)
            return FALLBACK_RESPONSE, {
                "error": str(exc),
                "latency_ms": latency_ms,
            }
