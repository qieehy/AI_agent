# Day 27：Reflection Agent 与有界自动修正

> 状态：**阶段 4 完成：Runtime、配置和生产组合根已经接线。**

## 原始问题

当前 `reflection` prompt 只要求执行模型在输出前自行检查。这个文字约定无法证明检查真的
发生，也没有独立的结构化结果、失败归因、修订次数上限或可观测事件。因此它是 prompt
pattern，不是生产级 Reflection Agent。

Day 27 的目标是让一个独立 Critic 检查候选最终答案，并让 Runtime 在有界预算内要求执行
模型修正。Critic 只评审，不调用工具、不修改 Memory、不直接替换答案。

## 参与者与所有权

- **执行模型**：生成候选答案，并根据受验证的反馈修订；
- **Critic**：读取用户请求、候选答案和可用证据，返回结构化评审；
- **Runtime**：拥有候选答案、评审轮数、预算、状态转换、事件和唯一终止路径；
- **Memory**：只有答案被接受后才持久化最终 assistant 消息，避免草稿污染会话。

Critic 输出属于不可信模型数据。只有通过字段、枚举、组合关系和资源上限校验的结果，才可
进入 Runtime。

## 阶段 1：评审数据契约

第一阶段只定义以下不可变内部模型：

```text
CritiqueResult
├── decision: accept | revise
└── feedback: None | 非空的有界文本
```

必须满足：

- 顶层字段精确为 `decision` 和 `feedback`；
- `accept` 必须搭配 `feedback=null`；
- `revise` 必须搭配非空反馈；
- 反馈去除首尾空白后仍非空，且不超过配置上限；
- 未知决策、错误类型、非法 JSON 一律 fail closed；
- 解析错误使用稳定的 `ReflectionError`，并用异常链保留 JSON 根因。

`feedback` 描述缺陷而不是携带替代答案。这保持职责边界：Critic 评价，执行模型生成，
Runtime 决策。

## 目标生命周期

1. 执行模型按现有工具循环生成一个不含 tool call 的候选答案；
2. Runtime 不立即写入 Memory，也不进入 `FINISHED`；
3. Critic 在独立超时内评审候选答案；
4. `accept`：Runtime 写入候选答案并走唯一成功终止路径；
5. `revise`：Runtime 注入临时、只读的反馈上下文，让执行模型再次生成；
6. 修订预算耗尽：按明确策略终止，不能无限循环或静默接受未通过答案。

## 失败、超时与取消

- 非法 Critic 输出、供应商失败和 Critic 超时转换为 `ReflectionError`，归因为
  `error_source="critic"`；
- 调用方取消必须原样传播，由 Runtime 进入既有 `CANCELED` 路径；
- Critic 失败时不应把候选草稿保存成最终回答；
- 整个 Run 的未来总超时仍应覆盖执行、工具、Critic 和修订，不被局部超时替代。

## 并发与清理

同一 session 的既有 lease 覆盖完整“生成—评审—修订”周期，因此两个 Run 不得交错草稿
或反馈。不同 session 可以并发。临时反馈由 Runtime 单次调用持有，不进入持久 Memory；
超时、失败和取消后不得遗留后台 Critic 任务。

## 可观测性

Runtime 将 `critique.decision`、评审轮次和 Critic 耗时写入 `state.metadata["critiques"]`，
并发送 `critique.completed` 事件。事件不包含完整候选答案或反馈。Critic 失败继续由统一
`run.error` 终态事件表达，并使用 `error_source="critic"` 归因。

每次 Critic 调用形成一个 `StepKind.CRITIQUE`。成功 Step 只记录候选 LLM Step 索引、评审
轮次、决策和耗时；失败或取消 Step 记录稳定错误类别，不保存候选答案或完整反馈。Critic
Step 占用 `max_steps` 预算，Runtime 在预算不足时不会启动新的评审。`critique.completed`
事件使用 Critic Step 自己的索引，因此不会和下一次 LLM Step 发生编号碰撞。

## 流式输出语义

启用 Critic 时，执行模型的流仍会被 Runtime 消费和组装，但 `llm.token` 不会在评审前发送，
避免把随后被拒绝的草稿泄露给用户。被接受的答案写入 Memory，并通过成功终态事件的
`final` 字段提供。未启用 Critic 时保持原有逐 token 输出行为。

失败终态的 `final` 固定为 `null`；只有 `FINISHED` 且最后一条消息是文本 assistant 消息时，
终态事件才携带最终答案。

## Router 修订查询

正常轮次继续使用现有用户内容或 Planner 路由查询。修订轮只使用经过结构校验和长度限制的
Critic feedback 选择工具，不把原查询与反馈再次拼接，避免 Runtime 自己把已合法的查询扩张
到 Router 上限之外。执行模型仍通过临时 Reflection system message 获得原上下文、候选答案
和反馈。

配置层强制 `critic_max_feedback_chars <= tool_router_max_query_chars`。

## 生产组合根

`build_runtime()` 仅在 `pattern="reflection"` 时构造并注入 Critic。Critic 与执行模型共享同一
异步 LLM 客户端，但调用在一次 session lease 内按顺序发生。以下配置在启动时验证：

- `critic_timeout_s`：正数且有限；
- `critic_max_feedback_chars`：`1..16384`，且不得超过 Router 查询上限；
- `reflection_revision_rounds`：`0..10`。

## 分阶段实现顺序

1. **数据模型与解析边界（完成）**：完成 `CritiqueDecision`、`CritiqueResult`、
   `parse_critique_result` 和 `ReflectionError`；
2. **Critic 调用边界（完成）**：`review(context_messages, candidate_answer)` 使用独立
   system prompt，把上下文和候选答案编码为 JSON 数据，并在不暴露工具的情况下调用 LLM；
   同时覆盖响应形状检查、独立超时、取消、清理和异常转换；
3. **Runtime 接线（完成）**：候选答案暂存、接受/修订状态转换、预算和唯一终止路径；
4. **组合根与配置（完成）**：仅在 `reflection` 模式创建 Critic，并验证配置；
5. **自动化验收（完成）**：覆盖正常接受、一次修订、预算耗尽、非法输出、失败、超时、
   取消、流式草稿隔离、Router 反馈以及 session 并发。

## 当前不提供的保证

当前不提供跨进程或分布式 session 串行保证，也没有覆盖整个 Run 的统一总超时。Critic 与
执行模型共享 provider 客户端，不提供故障域隔离。当前缓冲策略也不提供“审核通过后再逐
token 重放”的伪流式体验。

自动化行为测试证明状态机和边界契约，但不证明 Reflection 一定降低答案错误率；该结论仍需
固定评测集、无 Critic 基线、质量指标、延迟与成本数据。

## 小练习

为 Critic Step 增加聚合指标：分别统计接受率、修订率、超时率和 P95 耗时，同时思考怎样在
不记录候选答案和反馈正文的前提下定位质量回归。
