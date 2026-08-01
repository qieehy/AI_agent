"""测试 memory/short_term.py BufferMemory 的 4 条截断不变量。

D4 验收标准核心：锁死截断行为 — system 永不被淘汰、最新消息永不删、
单条超大允许超预算、tool 回合块整体淘汰。
"""
from __future__ import annotations

import pytest

from memory.short_term import BufferMemory
from memory.token_counter import count_message


# ---------- 辅助 ----------

def _make_msg(role: str = "user", content: str = "hello", **extra):
    """构造 OpenAI 格式消息 dict。"""
    msg = {"role": role, "content": content}
    msg.update(extra)
    return msg


def _make_tool_result(call_id: str, name: str, content: str, role: str = "tool"):
    return {"role": role, "tool_call_id": call_id, "name": name, "content": content}


def _make_tool_calls(name: str, arguments: str, call_id: str = "c1"):
    """构造 assistant 消息（带 tool_calls）。"""
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }],
    }


# ---------- 测试：构造 ----------

def test_init_without_system():
    """无 system_message 时正常构造。"""
    m = BufferMemory()
    assert m.get_system_message() is None
    assert m.messages == []
    assert m.total_tokens == 0


def test_init_with_system_counts_tokens():
    """有 system_message 时计入 _total_tokens。"""
    m = BufferMemory(system_message={"role": "system", "content": "test"})
    assert m.total_tokens > 0


def test_init_with_max_tokens_different_defaults():
    """max_tokens 为构造参数，不强制 4000。"""
    m = BufferMemory(max_tokens=100)
    assert m.total_tokens == 0  # 无 system


# ---------- 测试：add_message + get_context ----------

def test_add_message_increments_tokens():
    """add_message 后 total_tokens 增加。"""
    m = BufferMemory()
    before = m.total_tokens
    m.add_message(_make_msg())
    assert m.total_tokens > before


def test_get_context_returns_system_plus_window():
    """get_context = [system] + 窗口（defensive copy）。"""
    m = BufferMemory(system_message={"role": "system", "content": "bot"})
    m.add_message(_make_msg("user", "hello"))
    ctx = m.get_context()
    assert ctx[0] == {"role": "system", "content": "bot"}
    assert ctx[1] == {"role": "user", "content": "hello"}


def test_get_context_without_system():
    """无 system 时 get_context 只返回窗口。"""
    m = BufferMemory()
    m.add_message(_make_msg())
    assert m.get_context() == [{"role": "user", "content": "hello"}]


# ---------- 测试：防御性拷贝 ----------

def test_messages_property_is_defensive_copy():
    """改 messages 返回值不影响内部状态。"""
    m = BufferMemory()
    m.add_message(_make_msg("user", "original"))
    msgs = m.messages
    msgs.append(_make_msg("user", "injected"))
    assert len(m.messages) == 1


def test_get_context_is_defensive_copy():
    """改 get_context 返回值不影响内部状态。"""
    m = BufferMemory()
    m.add_message(_make_msg())
    ctx = m.get_context()
    ctx.pop()
    assert len(m.messages) == 1


# ---------- 测试：不变量 1 — system 永不被淘汰 ----------

def test_system_message_survives_truncation():
    """超限截断后 system 仍在 get_context 里。"""
    m = BufferMemory(
        system_message={"role": "system", "content": "persistent"},
        max_tokens=20,
    )
    # 塞满，触发截断
    for i in range(20):
        m.add_message(_make_msg("user", f"msg-{i}"))
    ctx = m.get_context()
    assert ctx[0] == {"role": "system", "content": "persistent"}


# ---------- 测试：不变量 2 — 最新消息永不删 ----------

def test_newest_message_survives_truncation():
    """截断后最新一条消息还在。"""
    m = BufferMemory(max_tokens=30)
    for i in range(10):
        m.add_message(_make_msg("user", f"msg {i}"))
    msgs = m.messages
    assert msgs[-1]["content"] == "msg 9"


def test_list_never_gets_empty():
    """截断不会删到空（至少留 1 条）。"""
    m = BufferMemory(max_tokens=5)
    m.add_message(_make_msg("user", "only one"))
    assert len(m.messages) == 1


# ---------- 测试：不变量 3 — 单条超大允许超预算 ----------

def test_single_oversized_message_survives():
    """单条消息超过 max_tokens 时不崩不丢。"""
    m = BufferMemory(max_tokens=10)
    m.add_message(_make_msg("user", "x" * 200))
    assert len(m.messages) == 1
    # 允许超预算
    assert m.total_tokens > 10


def test_second_message_added_after_oversized():
    """超大消息后面再加新消息，仍正常运作。"""
    m = BufferMemory(max_tokens=10)
    m.add_message(_make_msg("user", "x" * 200))  # 超预算
    m.add_message(_make_msg("user", "new"))
    # 至少还有消息
    assert len(m.messages) >= 1


# ---------- 测试：不变量 4 — tool 回合块整体淘汰 ----------

def test_tool_turn_block_evicted_together():
    """assistant tool_calls + 它的 tool 结果一起淘汰，不留悬空引用。"""
    m = BufferMemory(max_tokens=30)
    m.add_message(_make_msg("user", "compute 1+2"))
    m.add_message(_make_tool_calls("add", '{"a":1,"b":2}', call_id="c1"))
    m.add_message(_make_tool_result("c1", "add", "3"))
    # 再加一条超级大消息触发截断
    m.add_message(_make_msg("user", "x" * 500))

    msgs = m.messages
    # 验证：如果 tool_calls 消息还在，tool 结果也必须还在（反之亦然）
    tool_result_ids = {msg.get("tool_call_id") for msg in msgs if msg.get("role") == "tool"}
    assistant_call_ids = set()
    for msg in msgs:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                assistant_call_ids.add(tc["id"])

    # 每个 assistant 声明的 tool_call_id 都必须有对应的 tool 结果
    # 每个 tool 结果的 tool_call_id 都必须有对应的 assistant 声明
    orphan_assistant = assistant_call_ids - tool_result_ids
    orphan_tool = tool_result_ids - assistant_call_ids
    assert not orphan_assistant, f"orphan assistant calls: {orphan_assistant}"
    assert not orphan_tool, f"orphan tool results: {orphan_tool}"


def test_cross_turn_tool_blocks_not_mixed():
    """多个回合交错时的 tool 块不会被错误合并。"""
    m = BufferMemory(max_tokens=200)
    # 回合 1
    m.add_message(_make_msg("user", "q1"))
    m.add_message(_make_tool_calls("add", '{"a":1}', call_id="c1"))
    m.add_message(_make_tool_result("c1", "add", "3"))
    # 回合 2
    m.add_message(_make_msg("user", "q2"))
    m.add_message(_make_tool_calls("multiply", '{"a":2}', call_id="c2"))
    m.add_message(_make_tool_result("c2", "multiply", "6"))
    # 发一段长消息触发截断
    m.add_message(_make_msg("user", "y" * 200))

    msgs = m.messages
    # 检查 tool 消息的 call_id 都能找到对应的 assistant
    tool_ids = {msg["tool_call_id"] for msg in msgs if msg.get("role") == "tool"}
    assistant_ids = set()
    for msg in msgs:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                assistant_ids.add(tc["id"])
    assert not (tool_ids - assistant_ids), f"orphan tools: {tool_ids - assistant_ids}"
    assert not (assistant_ids - tool_ids), f"orphan assistant: {assistant_ids - tool_ids}"


# ---------- 测试：计数不变量 ----------

def test_recount_invariant_after_truncation():
    """每次 truncation 后 total_tokens == sum(count_message(m) for m in messages)。"""
    m = BufferMemory(max_tokens=50)
    for i in range(10):
        m.add_message(_make_msg("user", f"message number {i}"))
    expected = sum(count_message(msg) for msg in m.messages)
    assert m.total_tokens == expected


def test_total_tokens_not_beyond_budget_unless_oversized():
    """截断后 total_tokens <= max_tokens，除非最后一条本身就超预算。"""
    m = BufferMemory(max_tokens=50)
    for i in range(10):
        m.add_message(_make_msg("user", f"message {i}"))
    if len(m.messages) > 1:
        assert m.total_tokens <= 50


# ---------- 测试：clear ----------

def test_clear_resets_messages_and_tokens():
    """clear() 后 messages == [], _total_tokens 回到只有 system 的状态。"""
    m = BufferMemory(system_message={"role": "system", "content": "bot"})
    m.add_message(_make_msg("user", "a"))
    m.add_message(_make_msg("user", "b"))
    m.clear()
    assert m.messages == []
    # 只有 system 消息的 token
    assert m.total_tokens == count_message({"role": "system", "content": "bot"})


def test_clear_without_system():
    """无 system 时 clear 回到 0。"""
    m = BufferMemory()
    m.add_message(_make_msg())
    m.clear()
    assert m.messages == []
    assert m.total_tokens == 0
