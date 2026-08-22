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
              ┌──────────┼──────────┐
              ↓          ↓          ↓
          LLMClient   Executor   EventBus
          (重试/退避)  (串行/并行)  (Hook)
              ↓          ↓
         OpenAI API   ToolRegistry
                      ↓
                  calculator
                      ...
```

## 模块总览

| 模块 | 路径 | 职责 |
|---|---|---|
| `runtime/` | `state.py` `step.py` `event.py` `runtime.py` | Agent 主循环：State → LLM → Tool → Event |
| `llm/` | `client.py` | OpenAI 客户端：指数退避重试（最多 3 次） |
| `tools/` | `registry.py` `executor.py` | Tool 注册表（完整 OpenAI Schema）+ 执行器（serial/parallel） |
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
pip install -e .

# 配置
cp .env.example .env
# 编辑 .env，填入 API_KEY 和 BASE_URL

# 运行
python main.py
> 1+2 等于多少？
```

## 运行测试

```bash
pip install -e ".[dev]"
pytest tests/ -v        # 188 tests
```

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
