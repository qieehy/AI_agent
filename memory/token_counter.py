import tiktoken

ENCODING = tiktoken.get_encoding("cl100k_base")

def count_text(text: str) -> int:
    """单段文本token数"""
    if not text:
        return 0
    return len(ENCODING.encode(text))

def count_message(message: dict) -> int:
    """单条 OpenAI 消息的 token 数（含 role/content/tool_calls 开销）。

         参考 OpenAI cookbook：
         https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

         规则：
         - 每条消息基础开销 = 4 tokens
         - content / name / tool_call_id / function.name / function.arguments 各算实际 token
     """
    n = 4   #单个message基础开销

    n += count_text(message.get("role", ""))
    n += count_text(message.get("content", ""))
    n += count_text(message.get("name", ""))

    tool_calls = message.get("tool_calls") or []
    for tc in tool_calls:
        n -= 1          #role 不单独算
        func = tc.get("function", {})
        n += count_text(func.get("name", ""))
        n += count_text(func.get("arguments", ""))

    n += count_text(message.get("tool_call_id", ""))

    return n

def count_messages(messages: list[dict]) -> int:
    """一组消息的总 token 数（含一次 chat 调用开销 3 tokens）。"""
    if not messages:
        return 0
    return sum(count_message(m) for m in messages) + 3

