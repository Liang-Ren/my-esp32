"""
prompt_builder — assembles the instructions and input for the Responses API.

build_input() returns (instructions, input_messages):
  instructions  → passed as `instructions=` to responses.create()
  input_messages → passed as `input=` to responses.create()
"""
from dataclasses import dataclass, field

BASE_SYSTEM_PROMPT = (
    "你是温哥华小智，一个友善、聪明的AI语音助手。用户通过ESP32设备与你对话。\n"
    "回复规则：\n"
    "- 用简短清晰的中文，50到100字以内\n"
    "- 禁止使用Markdown、列表符号、表格或任何特殊格式\n"
    "- 用自然口语，像朋友一样说话\n"
    "- 你能看到对话历史，不要说\"我无法记住之前的对话\""
)

TRANSLATION_PROMPT = (
    "你是翻译助手。请直接给出翻译结果，不加解释，不加前缀。"
)


@dataclass
class PromptContext:
    """Structured memory context passed from MemoryService to build_input."""
    memory_summary: str = ""
    recent_memory: list[str] = field(default_factory=list)
    user_preferences: str = ""


def build_input(
    user_text: str,
    history: list[dict],
    *,
    memory_summary: str = "",
    recent_memory: list[str] | None = None,
    user_preferences: str = "",
    mode: str = "normal",
) -> tuple[str, list[dict]]:
    """
    Build (instructions, input_messages) for the Responses API.

    Args:
        user_text:        Transcribed user utterance.
        history:          Short-term conversation turns [{role, content}, ...].
        memory_summary:   Long-term user background from memory service.
        recent_memory:    Semantic memory hits relevant to this query.
        user_preferences: Serialized preference string from memory service.
        mode:             "normal" | "translation" | "fallback"

    Returns:
        instructions:    System-level instructions for the model.
        input_messages:  Conversation messages to pass as `input=`.
    """
    if mode == "translation":
        instructions = TRANSLATION_PROMPT
    else:
        parts = [BASE_SYSTEM_PROMPT]
        if memory_summary:
            parts.append(f"\n【用户背景】{memory_summary}")
        if recent_memory:
            parts.append("【近期记忆】" + "；".join(recent_memory[:5]))
        if user_preferences:
            parts.append(f"【用户偏好】{user_preferences}")
        instructions = "\n".join(parts)

    input_messages: list[dict] = list(history)
    input_messages.append({"role": "user", "content": user_text})
    return instructions, input_messages
