"""D23 验收基准：证明异步架构的 QPS 提升 ≥ 5x。

【为什么这样设计】

1. 验收对象是「线程池基线」，串行基线只是参考
   - 串行基线（一个线程顺序跑）代表"没做任何并发"的旧形态，只作参考
   - sync Runtime 的真实并发形态是 thread-per-request：每个请求占一个线程，
     所以公平的对手是 ThreadPoolExecutor 并发跑 sync run()
   - async 赢在"等待不占线程"，这只有在并发下才可测量：
       串行总时长   ≈ N × DELAY                = 30s
       线程池总时长 ≈ ceil(N / WORKERS) × DELAY =  3s
       异步总时长   ≈ DELAY + 每请求开销         ≈ 0.5~0.6s
     → 结构化加速比 ≈ ceil(N / WORKERS) = 6

2. 测量卫生（不清理这两项，异步基线的边距会被吃掉）
   - logger.remove()：loguru 每条日志都写 stderr，60 个请求 × 数条日志的
     同步 I/O 全落在事件循环线程上。真实 LLM 延迟是 1~30s，这点 I/O 可忽略，
     但在 0.5s 假延迟的基准里会被放大
   - SessionStore(":memory:")：默认 create_memory_manager() 落盘
     data/sessions.db，同样是每请求 I/O 落在循环线程上

3. 每个请求一个独立 Runtime + 独立 :memory: 记忆
   - 线程池基线：sqlite 连接不是线程安全的，共享会被 check_same_thread 打爆
   - 异步基线：60 个协程共享一个 BufferMemory 会互相污染
   - 两边对称地付 Runtime 构建成本，才是同口径对比

用法（必须在仓库根目录）：python -m scripts.bench_async
退出码：0 = 达标，1 = 未达标（可直接挂 CI 验收）
"""
import asyncio
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from memory import MemoryManager, SessionStore
from observability import logger
from runtime import RunStatus, Runtime
from tools import Executor, ToolRegistry

N = 60                # 请求总数
WORKERS = 10          # 线程池基线的并发度
DELAY = 0.5           # 假 LLM 单次延迟（秒）
TARGET_SPEEDUP = 5.0  # 验收线


# ---------- 假 LLM（DI 注入，不碰真实 API） ----------

def _fake_response(text: str = "done") -> SimpleNamespace:
    """鸭子类型的 LLM 响应：只实现 Runtime 消费的三个点。

    Runtime 读取 response.choices[0].message 的
    .content / .tool_calls / .model_dump()（见 runtime/runtime.py:104）。
    """
    message = SimpleNamespace(
        content=text,
        tool_calls=[],  # 空 = 无工具调用，跑一轮就 FINISHED
        model_dump=lambda: {"role": "assistant", "content": text},
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _sync_llm(messages, tools=None):
    """sync 基线的假 LLM：time.sleep 占住当前线程 0.5s。"""
    time.sleep(DELAY)
    return _fake_response()


async def _async_llm(messages, tools=None):
    """async 基线的假 LLM：asyncio.sleep 让出事件循环 0.5s。

    同样是 0.5s 的"等待"，但线程不被占住——这就是 QPS 提升的全部来源。
    """
    await asyncio.sleep(DELAY)
    return _fake_response()


def _poison_llm(messages, tools=None):
    """不该被调用的 sync 槽位：异步基线跑了它 = 接线错误。"""
    raise AssertionError("async 基线的 Runtime 走了 sync LLM 槽位")


def _make_runtime(llm_call=None, llm_call_async=None) -> Runtime:
    """每请求一个 Runtime + 独立 :memory: 记忆（理由见模块 docstring 第 3 点）。"""
    return Runtime(
        llm_call=llm_call,
        llm_call_async=llm_call_async,
        tool_executor=Executor(ToolRegistry(), mode="serial"),
        memory_manager=MemoryManager(SessionStore(":memory:")),
    )


# ---------- 三条基线 ----------

def bench_serial() -> tuple[float, list]:
    """串行基线（仅参考）：一个线程顺序跑完 N 个请求。"""
    start = time.perf_counter()
    states = [_make_runtime(llm_call=_sync_llm).run("hi") for _ in range(N)]
    return time.perf_counter() - start, states


def bench_sync_pool() -> tuple[float, list]:
    """线程池基线（验收对象）：thread-per-request，sync Runtime 的真实并发形态。"""
    def run_one(_: int):
        return _make_runtime(llm_call=_sync_llm).run("hi")

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        states = list(pool.map(run_one, range(N)))
    return time.perf_counter() - start, states


async def _bench_async() -> tuple[float, list]:
    """异步基线：单事件循环并发 N 个 run_async，等待不占线程。"""
    async def run_one(_: int):
        runtime = _make_runtime(llm_call=_poison_llm, llm_call_async=_async_llm)
        return await runtime.run_async("hi")

    start = time.perf_counter()
    states = await asyncio.gather(*(run_one(i) for i in range(N)))
    return time.perf_counter() - start, states


# ---------- 主流程 ----------

def main() -> int:
    logger.remove()  # 关掉 loguru：日志 I/O 会吃掉异步基线的边距（docstring 第 2 点）

    print(f"D23 异步架构基准 — N={N} 请求, 假 LLM 延迟 {DELAY}s, 线程池 {WORKERS} 并发")
    print()

    serial, s_states = bench_serial()
    pooled, p_states = bench_sync_pool()
    async_elapsed, a_states = asyncio.run(_bench_async())

    # 验证：所有请求都必须正常完成，否则数据无效
    for name, states in [("串行", s_states), ("线程池", p_states), ("异步", a_states)]:
        bad = [s for s in states if s.status != RunStatus.FINISHED]
        if bad:
            print(f"[FAIL] {name} 基线有 {len(bad)} 个请求未 FINISHED（示例 status={bad[0].status}）")
            return 1

    speedup = pooled / async_elapsed
    print(f"串行基线   (sync, 1 线程)      : {serial:6.2f}s   （仅参考）")
    print(f"线程池基线 (sync, {WORKERS} 线程)     : {pooled:6.2f}s   （验收对象）")
    print(f"异步基线   (async, 1 事件循环) : {async_elapsed:6.2f}s")
    print()
    print(f"加速比 = {pooled:.2f}s / {async_elapsed:.2f}s = {speedup:.2f}x")
    print(f"验收线: >= {TARGET_SPEEDUP}x  ->  {'PASS' if speedup >= TARGET_SPEEDUP else 'FAIL'}")
    return 0 if speedup >= TARGET_SPEEDUP else 1


if __name__ == "__main__":
    sys.exit(main())
