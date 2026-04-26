import os
from openai import AsyncOpenAI

FALLBACK_RESPONSE = "抱歉，我现在遇到了一点问题，请稍后再试。"


class LLMClient:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    async def complete(self, messages: list[dict]) -> tuple[str, dict]:
        """Returns (response_text, usage_dict)."""
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=200,
                temperature=0.7,
            )
            text = resp.choices[0].message.content.strip()
            usage = {
                "model": resp.model,
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }
            return text, usage
        except Exception as e:
            return FALLBACK_RESPONSE, {"error": str(e), "model": self.model}