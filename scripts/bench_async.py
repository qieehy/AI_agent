"""D24 验收基准：异步架构 QPS 提升 ≥ 5x（sync Runtime 删除后的新口径）。

【为什么这样设计】

1. 验收对象是「线程池基线」，串行基线只是参考
   - D24 删除了 sync Runtime，线程池基线改为「每线程一个事件循环」：
     ThreadPoolExecutor 的每个线程里 asyncio.run(run_async(...))。
     这正是"老架构 thread-per-request"在纯 async 世界的等价物——
     每条请求独占一个线程（含一个事件循环），互不共享。
   - 异步基线：单事件循环 asyncio.gather 并发跑同一份 run_async。
     两条基线跑的代码完全相同，唯一区别是"等待"如何被调度，
     而这正是 QPS 提升的全部来源：
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
   - sqlite 连接不是线程安全的，线程池基线共享会被 check_same_thread 打爆
   - 异步基线 60 个协程共享一个 BufferMemory 会互相污染
   - 两边对称地付 Runtime 构建成本，才是同口径对比

用法（必须在仓库根目录）：python -m scripts.bench_async
退出码：0 = 达标，1 = 未达标（可直接挂 CI 验收）
输出只用 ASCII：Windows GBK 控制台打不出 ✅/❌（D23 教训）
"""
import asyncio
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from memory import MemoryManager, SessionStore
from observability import logger
from runtime import LoopGuard, LoopPolicy, RunStatus, Runtime, RuntimeState
from tools import Executor, ToolCallValidator, ToolRegistry, ToolRoute

N = 60                # 请求总数
WORKERS = 10          # 线程池基线的并发度
DELAY = 0.5           # 假 LLM 单次延迟（秒）
TARGET_SPEEDUP = 5.0  # 验收线


# ---------- 假 LLM（DI 注入，不碰真实 API） ----------

def _fake_response(text: str = "done") -> SimpleNamespace:
    """鸭子类型的 LLM 响应：只实现 Runtime 消费的三个点。

    Runtime 读取 response.choices[0].message 的
    .content / .tool_calls / .model_dump()（见 runtime/runtime.py）。
    """
    message = SimpleNamespace(
        content=text,
        tool_calls=[],  # 空 = 无工具调用，跑一轮就 FINISHED
        model_dump=lambda: {"role": "assistant", "content": text},
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


async def _async_llm(messages, tools=None):
    """唯一的假 LLM：asyncio.sleep 让出事件循环 0.5s。

    三条基线共用它——sync Runtime 已删，线程池基线里每个线程
    自己的事件循环跑同一段 await，这就是"等待不占线程"的对照组。
    """
    await asyncio.sleep(DELAY)
    return _fake_response()


class _NoToolsRouter:
    """Keep the benchmark independent from embedding latency."""

    async def route(self, query: str, schemas: list[dict]) -> ToolRoute:
        return ToolRoute((), (), "benchmark", 0.0)


def _make_runtime() -> Runtime:
    """每请求一个 Runtime + 独立 :memory: 记忆（理由见模块 docstring 第 3 点）。"""
    executor = Executor(ToolRegistry(), mode="serial")
    return Runtime(
        llm_call_async=_async_llm,
        tool_executor=executor,
        memory_manager=MemoryManager(SessionStore(":memory:")),
        loop_guard=LoopGuard(LoopPolicy()),
        validator=ToolCallValidator(executor.get_schemas()),
        tool_router=_NoToolsRouter(),
    )


# ---------- 三条基线 ----------

def _run_in_fresh_loop() -> RuntimeState:
    """线程池基线的一个请求：本线程新建事件循环跑 run_async。"""
    return asyncio.run(_make_runtime().run_async("hi"))


def bench_serial() -> tuple[float, list]:
    """串行基线（仅参考）：单线程顺序跑完 N 个请求。"""
    start = time.perf_counter()
    states = [_run_in_fresh_loop() for _ in range(N)]
    return time.perf_counter() - start, states


def bench_thread_pool() -> tuple[float, list]:
    """线程池基线（验收对象）：thread-per-request，每线程一个事件循环。"""
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        states = list(pool.map(lambda _: _run_in_fresh_loop(), range(N)))
    return time.perf_counter() - start, states


async def _bench_async() -> tuple[float, list]:
    """异步基线：单事件循环并发 N 个 run_async，等待不占线程。"""
    async def run_one(_: int):
        return await _make_runtime().run_async("hi")

    start = time.perf_counter()
    states = await asyncio.gather(*(run_one(i) for i in range(N)))
    return time.perf_counter() - start, states


# ---------- 主流程 ----------

def main() -> int:
    logger.remove()  # 关掉 loguru：日志 I/O 会吃掉异步基线的边距（docstring 第 2 点）

    print(f"D24 异步架构基准 — N={N} 请求, 假 LLM 延迟 {DELAY}s, 线程池 {WORKERS} 并发")
    print()

    serial, s_states = bench_serial()
    pooled, p_states = bench_thread_pool()
    async_elapsed, a_states = asyncio.run(_bench_async())

    # 验证：所有请求都必须正常完成，否则数据无效
    for name, states in [("串行", s_states), ("线程池", p_states), ("异步", a_states)]:
        bad = [s for s in states if s.status != RunStatus.FINISHED]
        if bad:
            print(f"[FAIL] {name} 基线有 {len(bad)} 个请求未 FINISHED（示例 status={bad[0].status}）")
            return 1

    speedup = pooled / async_elapsed
    print(f"串行基线   (每线程一循环, 1 线程)      : {serial:6.2f}s   （仅参考）")
    print(f"线程池基线 (每线程一循环, {WORKERS} 线程)     : {pooled:6.2f}s   （验收对象）")
    print(f"异步基线   (async, 1 事件循环)         : {async_elapsed:6.2f}s")
    print()
    print(f"加速比 = {pooled:.2f}s / {async_elapsed:.2f}s = {speedup:.2f}x")
    print(f"验收线: >= {TARGET_SPEEDUP}x  ->  {'PASS' if speedup >= TARGET_SPEEDUP else 'FAIL'}")
    return 0 if speedup >= TARGET_SPEEDUP else 1


if __name__ == "__main__":
    sys.exit(main())
