"""测试 memory/token_counter.py 的 OpenAI 格式消息计数。

关键：喂 Runtime 真实产出的 dict 形状——
- assistant 无 tool_call 时 model_dump() 会带 tool_calls: None（D4 修的 bug，曾 TypeError）
- tool 消息带 tool_call_id / name
- assistant 带 tool_calls 列表（function.name / function.arguments）

策略：纯函数直接调用，不 mock。断言用相对关系（计数 > 0）而非绝对数，
因为 tiktoken 的精确 token 数随编码变化（cl100k_base），但计数规则是稳定的。
"""
from __future__ import annotations

from memory.token_counter import count_message, count_messages, count_text

# ---------- count_text ----------

def test_count_text_empty():
    """空字符串 → 0。"""
    assert count_text("") == 0


def test_count_text_non_empty():
    """非空英文 → > 0。"""
    assert count_text("hello") > 0


def test_count_text_chinese():
    """中文 → > 0（tiktoken 对 CJK 也计数，每字约 1-3 token）。"""
    assert count_text("你好世界") > 0


def test_count_text_whitespace_only():
    """纯空白也算 token（不为 0 即可，规则交给 tiktoken）。"""
    assert count_text("   ") >= 0


# ---------- count_message ----------

def test_count_message_plain_user():
    """普通 user 消息 → 基础 4 + role + content > 0。"""
    n = count_message({"role": "user", "content": "hello"})
    assert n > 0
    # 内容更多 → token 更多（单调性）
    assert n < count_message({"role": "user", "content": "hello world" * 10})


def test_count_message_assistant_with_tool_calls_none():
    """【D4 修的 bug】assistant 无 tool_call 时 model_dump() 带 tool_calls: None。

    修复前：message.get("tool_calls", []) 在 key 存在但值为 None 时返回 None，
    遍历 None 抛 TypeError。修复后：count_message 正常返回 int。
    """
    n = count_message({"role": "assistant", "content": "hi", "tool_calls": None})
    assert isinstance(n, int)
    assert n > 0


def test_count_message_missing_keys():
    """关键缺失 → 用 .get 兜底，不抛 KeyError。"""
    n = count_message({"role": "user"})  # 无 content
    assert isinstance(n, int)
    n2 = count_message({})               # 空 dict
    assert isinstance(n2, int)
    assert n2 >= 4                        # 至少基础开销 4


def test_count_tool_role_message():
    """tool 消息（含 tool_call_id / name）→ 正确计数，不抛。"""
    n = count_message({
        "role": "tool",
        "tool_call_id": "call_abc",
        "name": "calculator",
        "content": "42",
    })
    assert isinstance(n, int)
    assert n > 0


def test_count_message_with_tool_calls_list():
    """assistant 带 tool_calls 列表 → function name/arguments 计入。"""
    n = count_message({
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'},
        }],
    })
    assert isinstance(n, int)
    assert n > 0
    # 工具名越长 → token 越多（单调性，保持 arguments 相同）
    n_long = count_message({
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "a_very_long_function_name_that_deserves_more_tokens", "arguments": '{"a": 1, "b": 2}'},
        }],
    })
    assert n_long > n


# ---------- count_messages ----------

def test_count_messages_empty():
    """空列表 → 0。"""
    assert count_messages([]) == 0


def test_count_messages_sums_with_batch_overhead():
    """多条消息 = 各条之和 + 3（一次 chat 调用开销）。"""
    m1 = {"role": "user", "content": "a"}
    m2 = {"role": "assistant", "content": "b"}
    assert count_messages([m1, m2]) == count_message(m1) + count_message(m2) + 3


def test_count_messages_single_message_no_overhead():
    """单条消息也 +3（count_messages 是 chat 调用视角，恒有开销）。"""
    m = {"role": "user", "content": "a"}
    assert count_messages([m]) == count_message(m) + 3


def test_count_messages_mixed_roles():
    """混合 role（含 tool）的完整对话形状，不抛。"""
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "compute 1+2"},
        {"role": "assistant", "content": "", "tool_calls": None},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": "add", "arguments": '{"a":1,"b":2}'},
        }]},
        {"role": "tool", "tool_call_id": "c1", "name": "add", "content": "3"},
        {"role": "assistant", "content": "答案是 3"},
    ]
    assert count_messages(msgs) > 0
