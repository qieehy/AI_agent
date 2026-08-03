# Changelog

所有重要变更记录于此。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Planned (Week 2)
- 日志系统：loguru / structlog + JSON 输出 + trace_id
- Memory 持久化：SQLite + MemoryManager 工厂
- 测试体系：覆盖率 ≥ 70%
- CI/CD：GitHub Actions lint + test
- 配置中心：pydantic-settings 替代 os.getenv
- CLI 升级：typer + rich

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
