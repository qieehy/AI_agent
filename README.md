# AI Agent Framework

生产级 Agent 框架——从脚本到产品的 37 天演进。

**v0.3.0** — 工具 + RAG + 多模态：文件/网络/Shell 工具、混合检索 + 精排、带引用的 RAG 问答、视觉服务。

## 架构

```
                    User (CLI)
                         ↓
                     main.py
                         ↓
                  Agent Runtime
           ┌─────────┼──────────┬──────────┐
           ↓         ↓          ↓          ↓
      ToolRouter  LLMClient   Executor   EventBus
      (BGE 检索)  (重试/退避)  (串/并行)   (Hook)
           ↓         ↓          ↓
      候选 schemas OpenAI API ToolRegistry
```

## 模块总览

| 模块 | 路径 | 职责 |
|---|---|---|
| `runtime/` | `state.py` `step.py` `event.py` `runtime.py` | Agent 主循环：State → LLM → Tool → Event |
| `llm/` | `client.py` | OpenAI 客户端：指数退避重试（最多 3 次） |
| `tools/` | `registry.py` `router.py` `executor.py` | Tool 注册、Embedding 候选检索与执行 |
| `memory/` | `short_term.py` `token_counter.py` | BufferMemory 滑动窗口（4 条截断不变量） |
| `errors/` | `exceptions.py` | AgentError → LLMError / ToolError / ConfigError / MemoryError |
| `config/` | `settings.py` | pydantic-settings 配置中心 |

## 快速开始

```bash
# 安装
git clone <repo-url>
cd AI_agent
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[rag]"

# 配置
cp .env.example .env
# 编辑 .env，填入 API_KEY、BASE_URL；Tool Router 参数可按需覆盖

# 运行
python main.py
> 1+2 等于多少？
```

## 运行测试

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

真实 BGE Worker 集成测试默认跳过，避免 CI 意外下载模型。确认模型已缓存后，
可在 PowerShell 中显式验收生产进程链路：

```powershell
$env:RUN_REAL_EMBEDDING_INTEGRATION = "1"
$env:HF_HUB_OFFLINE = "1"
.\.venv\Scripts\python.exe -m pytest tests/test_embedding_worker_integration.py -q -p no:cacheprovider
```

## Tool Router

`Runtime.run_async()` 在每个 LLM 步骤前执行一次 Embedding Router：

1. 使用工具名、描述和参数 Schema 构建能力文本；
2. 用本地 BGE 模型批量计算查询与未缓存工具向量；
3. 按余弦相似度、阈值和 `top-k` 选择候选工具；
4. 只把候选 Schema 发送给 LLM，Validator 同时拒绝本轮未入选的调用。

本地 BGE 由持久的 `rag.embedding_worker` 子进程独占。Worker 在第一次路由时
惰性启动；父进程通过有界 JSONL 协议发送批量请求，并分别限制锁等待、模型
启动、推理和关闭时间。推理超时、调用取消或协议损坏会终止当前 Worker，
下一次路由启动新的 generation。CLI 正常退出、异常和取消都会关闭 Worker，
生产路径不会回退到无法硬终止的线程推理。

父进程为 Worker 生命周期写入结构化日志。事件覆盖启动与 READY、推理开始与
完成、锁/启动/推理/关闭超时、取消、协议违规、generation 重启和最终关闭；
字段包括 `component`、`event`、`operation`、`outcome`、`generation`、PID、
批量大小、向量维度和耗时（适用时）。日志不会记录原始文本、向量、环境变量、
Token 或 Worker stderr 内容；stderr 只保留在客户端的有界诊断尾部中。

Worker 边界可通过 `.env` 配置：

```text
TOOL_ROUTER_WORKER_STARTUP_TIMEOUT_S=180
TOOL_ROUTER_WORKER_INFERENCE_TIMEOUT_S=30
TOOL_ROUTER_WORKER_LOCK_TIMEOUT_S=5
TOOL_ROUTER_WORKER_SHUTDOWN_TIMEOUT_S=5
TOOL_ROUTER_WORKER_MAX_REQUEST_BYTES=262144
TOOL_ROUTER_WORKER_MAX_RESPONSE_BYTES=4194304
TOOL_ROUTER_WORKER_MAX_STDERR_CHARS=4096
```

没有工具达到阈值时返回空候选集，LLM 继续生成无工具回答；路由或
Embedding 失败则在 LLM 调用前终止 Run，不会回退为暴露全部工具。

每一步的路由结果按顺序记录在 `RuntimeState.metadata["tool_routes"]`。
未启用 Planner 时，第一步使用用户输入，后续步骤使用 Memory 中最后一条消息；
`plan_execute` 模式则在每一步使用经过验证的总目标和全部节点目标。当前尚无活动
节点状态，因此 Router 使用的是完整计划，而不是单个 `StepGoal`。

## Planner Agent

`plan_execute` 模式会在每次 Run 的执行循环前调用独立 Planner。Planner 获得用户输入
和工具能力摘要，只返回计划 JSON，不执行工具。框架随后检查顶层与任务字段、资源上限、
任务 ID、依赖引用、自依赖和依赖环；只有通过检查的数据才会转换为不可变 `TaskPlan`。

成功计划会：

- 写入 `RuntimeState.metadata["plan"]`，并记录 `planner_duration_ms`；
- 发出一次 `plan.created` 事件；
- 作为 Tool Router 的查询意图；
- 作为临时 system context 交给执行模型，但不写入持久 Memory。

Planner 超时、供应商失败、响应结构错误或非法计划统一成为 `PlannerError`。Runtime 以
`error_source="planner"` 终止，不调用 Router、执行模型或工具；上层取消仍按
`CancelledError` 传播，并进入 Runtime 的 `CANCELED` 终态。同一 session 的规划位于
`SessionCoordinator` lease 内，不会与同 session 的另一 Run 交错。

配置：

```text
PATTERN=plan_execute
PLANNER_TIMEOUT_S=30
PLANNER_MAX_TASKS=12
PLANNER_MAX_GOAL_CHARS=1000
```

当前计划是经过验证的执行上下文，不是节点级调度器：Runtime 尚未记录每个节点的
`pending/running/succeeded/failed`，也不提供节点并行、结果绑定或断点恢复。

## Reflection Agent

`reflection` 模式在执行模型生成无工具调用的候选答案后，交给独立 Critic 评审。候选答案
只有在 Critic 返回 `accept` 后才写入 Memory；`revise` 会把经过严格 JSON 校验的反馈作为
临时上下文交给执行模型，并在配置的预算内重新生成。被拒绝的草稿不会写入 Memory，也不会
通过 `llm.token` 流式事件发送给用户。

Critic 非法响应、调用失败或超时统一转换为 `ReflectionError`，并以
`error_source="critic"` 结束 Run。修订预算耗尽时进入 `reflection_limit`，失败终态不会携带
或显示旧的用户消息作为最终答案。同一 session 的生成、评审和修订位于同一个 lease 内。

配置：

```text
PATTERN=reflection
CRITIC_TIMEOUT_S=30
CRITIC_MAX_FEEDBACK_CHARS=2000
REFLECTION_REVISION_ROUNDS=1
```

`CRITIC_MAX_FEEDBACK_CHARS` 不得超过 `TOOL_ROUTER_MAX_QUERY_CHARS`；配置在启动时验证。
Reflection 模式使用缓冲输出：CLI 在答案通过评审后一次性显示最终文本，不会提前显示候选
草稿。每次评审会形成一个不含候选正文和反馈正文的 `CRITIQUE` Step，并占用
`max_steps` 预算。当前实现不提供整个 Run 的统一总超时，也不声称 Critic 必然提高答案
正确率。

## 技术栈

- **LLM**: OpenAI-compatible API（MiniMax / GPT / Claude）
- **Token 计数**: tiktoken（cl100k_base）
- **配置**: pydantic-settings + .env
- **测试**: pytest + unittest.mock
- **Python**: ≥3.10

## v0.1.0 特性

- [x] 异常体系：AgentError → LLMError / ToolError / ConfigError / MemoryError
- [x] Runtime 主循环：RUNNING → AWAITING_TOOL → FINISHED / FAILED
- [x] Tool 注册表 + 执行器：完整 OpenAI Schema（Annotated description、Literal enum、Optional nullable、list/dict）
- [x] BufferMemory：4 条截断不变量（system 不淘汰、最新不丢、单条超限允许、tool 回合块整体淘汰）
- [x] LLM 客户端重试：指数退避 + jitter，临时异常重试、永久异常直接抛
- [x] 81 测试 + 1 万轮压力测试

## v0.3.0 特性

- [x] RAG 管线：PDF 切块 → BGE 向量化 → FAISS 检索 → 带引用生成
- [x] 混合检索 + 精排：BM25 + RRF 融合召回，cross-encoder 两段式精排
- [x] 引用溯源：答案携带 [n] 编号来源（文件 / 页码 / 分数）
- [x] 视觉服务：VisionService 接口 + OpenAIVisionService（待接真实端点验收）
- [x] 工具扩展：文件 / 网络 / Shell 三件套（沙箱 + 白名单）
- [x] 工程化：日志 trace_id / SQLite 持久化 / pydantic-settings / CLI / CI

## 路线图

37 天分 6 周：[完整计划](doc/37天_Agent_Framework_终极学习路径.md)

| 周 | 主题 | 里程碑 |
|---|---|---|
| Week 1 | Agent Runtime 重构（防御优先） | v0.1 ✅ |
| Week 2 | 工程化升级（日志/持久化/CI/CLI） | v0.2 ✅ |
| Week 3 | 工具扩展 + RAG + 多模态 | v0.3 ✅ |
| Week 4 | 高级 Agent（Streaming/异步/Planner/Reflection） | v0.4 |
| Week 5 | MCP + Multi-Agent + 安全 | v0.5 |
| Week 6 | 产品化（FastAPI/WebUI/Docker） | v1.0 |

## 设计原则

- **防御优先**：异常/Token/安全早于功能堆叠
- **SRP 拆分**：Registry 只管注册，Executor 只管执行，Runtime 只管调度
- **组合根**：`main.py` 组装依赖，不用模块级单例
- **Duck typing**：窄接口用 Protocol/Callable，不做 ABC 过度抽象
- **O(1) 增量**：Token 计数增量更新，不重复计算全列表

## License

MIT
