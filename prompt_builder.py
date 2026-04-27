SYSTEM_PROMPT = """你是温哥华小智，一个友善、聪明的AI语音助手。用户通过ESP32设备与你对话。
请用简短清晰的中文回答（50-100字），因为回复会被转成语音播放。
不要使用Markdown、列表符号或特殊格式。直接说话，像朋友一样自然。
你能看到本次对话的历史记录，可以根据历史回答问题，不要说"我无法记住之前的对话"。"""

TRANSLATION_PROMPT = """你是翻译助手。请直接给出翻译结果，不加解释，不加前缀。"""


def detect_mode(text: str) -> str:
    if not text:
        return "fallback"
    translation_keywords = ["翻译", "translate", "translation", "怎么说", "英文怎么", "用英语", "用中文"]
    if any(kw in text.lower() for kw in translation_keywords):
        return "translation"
    return "normal"


def build_messages(
    user_input: str,
    history: list[dict],
    long_term: dict,
    mode: str = "normal",
) -> list[dict]:
    if mode == "translation":
        system = TRANSLATION_PROMPT
    else:
        system = SYSTEM_PROMPT
        if long_term.get("summary"):
            system += f"\n\n【用户背景】{long_term['summary']}"
        if long_term.get("facts"):
            system += f"\n【重要信息】{'、'.join(long_term['facts'][:5])}"

    messages = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_input})
    return messages