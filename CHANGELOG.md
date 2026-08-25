# Changelog

所有重要变更记录于此。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Added
- **Streaming 输出**（D22）：SSE token 流 + `LLM_TOKEN` 事件 + CLI 实时打字机；连接重试与中途异常分离
- **异步架构**（D23）：`AsyncLLMClient`（async chat/stream）+ Executor 异步执行（gather + `asyncio.to_thread`）+ `Runtime.run_async` 全链路；CLI 打字机走异步路径

### Fixed
- 流式工具名重装改为覆盖语义：provider 每 fragment 重复携带 name 时不再指数翻倍

### Planned (Week 4)
- Planner / Reflection

## [0.3.0] - 2026-08-22

v0.2（D8-D14 工程化）未单独发版记录，此版本一并收录 D8-D21。

### Added
- **RAG 管线**（D20）：PDF 加载 → 切块 → 嵌入 → FAISS 检索 → LLM 生成；重索引幂等（双存储先删旧账）
- **引用溯源**（D21）：`Answer(text, sources)` 编号来源 [n]，prompt 编号与 sources 一一对应
- **混合检索**（D21）：BM25Index（中文 bigram）+ RRF 融合 dense/sparse 两路召回
- **精排**（D21）：CrossEncoderReranker（bge-reranker-v2-m3），两段式：fetch 召回 → 精排 top_k
- **视觉服务**（D21）：VisionService ABC + OpenAIVisionService（base64 data URL，默认 gpt-4o-mini，懒加载）
- **向量化 / 向量库**（D18/D19）：BGE 本地嵌入服务 + FAISS CRUD
- **工具扩展**（D15-D17）：文件工具（路径沙箱）、网络工具（SSRF 沙箱）、Shell 工具（白名单 + 超时 + 输出截断）
- **工程化**（D8-D14）：loguru 结构化日志 + trace_id、SQLite 持久化、pydantic-settings 配置中心、typer + rich CLI、GitHub Actions（ruff / mypy / pytest）

### Fixed
- CI 注入假环境变量供 pydantic-settings 校验

### Tests
- 188 tests；真 BGE / reranker 模型测试经 importorskip 门控，CI 轻量运行

## [0.1.0] - 2026-08-03

### Added
- **异常体系**（D2）：`AgentError` → `LLMError` / `ToolError` / `ConfigError` / `MemoryError`，含 context + to_dict 序列化
- **Executor SRP 拆分**（D3）：ToolRegistry 只管注册，Executor 只管执行，serial/parallel 双模式
- **BufferMemory**（D4）：滑动窗口 + 4 条截断不变量（system 独立字段永不淘汰 / 最新消息 guard / 单条超大允许超预算 / tool 回合块整体淘汰）
- **Token 计数**（D4）：tiktoken 增量 O(1) 计数，防御性修复 `tool_calls=None` 越界 bug
- **Tool Schema 增强**（D5）：`_py_to_json` 支持 Literal enum / Optional nullable / list[T] array / dict object / Annotated description / 默认值
- **LLM 重试**（D6）：指数退避 + 30% jitter，临时异常重试 3 次，永久异常不重试

### Changed
- `memory/memory.py` 重构为 `memory/short_term.py`（BufferMemory）
- `_py_to_json` 返回 `dict` 替代原 `str`
- `_generate_schema` 使用 `get_type_hints(func, include_extras=True)` 解析 PEP 563 字符串注解

### Fixed
- `count_message` 对 `tool_calls=None`（Runtime `model_dump()` 产出）从 TypeError 修复为正确返回 int

### Tests
- 81 tests + 1 万轮压力测试（0.89s），覆盖所有模块
