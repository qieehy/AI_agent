# Changelog

所有重要变更记录于此。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Planned (Week 1)
- 异常体系：AgentError / LLMError / ToolError / ConfigError / MemoryOverflowError
- Memory token 管理（tiktoken）
- LLM 客户端重试（指数退避）
- Agent Loop 加固（多 tool_call 并行、错误回填）

## [0.1.0] - 2026-07-27

### Added
- 项目脚手架：pyproject.toml / .gitignore / .env.example / pre-commit
- README 草稿
- Runtime 分层设计（State / Step / Event / Runtime 数据类）
- 路线规划文档（37 天终极版 + 审计报告）

### Security
- `.env` 加入 `.gitignore`，保护 API 凭据
- pre-commit 钩子检测私钥
