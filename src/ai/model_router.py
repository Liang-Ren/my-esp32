"""
ModelRouter — decides which prompt mode to use for a given user utterance.
Isolated here so future routing logic (intent detection, tool calls, etc.) has a home.
"""

TRANSLATION_KEYWORDS = frozenset([
    "翻译", "translate", "translation", "怎么说", "英文怎么", "用英语", "用中文",
])


class ModelRouter:
    def detect_mode(self, text: str) -> str:
        """
        Returns:
          "fallback"    — empty or None input
          "translation" — translation request detected
          "normal"      — default conversational reply
        """
        if not text or not text.strip():
            return "fallback"
        lower = text.lower()
        if any(kw in lower for kw in TRANSLATION_KEYWORDS):
            return "translation"
        return "normal"
