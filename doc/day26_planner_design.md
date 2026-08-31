# Day 26：Planner Agent 与受验证的子任务 DAG

> 状态：**已实现并接入生产组合根**。本文同时保留分阶段教学顺序与当前能力边界。

## 目标与边界

D26 把 `plan_execute` 从提示词约定升级为生产接线：每次 Run 在执行前由独立 Planner
调用把用户目标拆成子任务 DAG。Planner 不执行工具、也不修改会话；Runtime 拥有计划、
验证结果和终止状态。

本阶段不实现节点级并行调度、持久化恢复或 Critic。DAG 用于约束执行模型并为 Tool
Router 提供完整任务意图。Reflection/Critic 属于 D27。

## 不变量与信任边界

模型输出是不可信输入。`runtime/planner.py` 在计划进入 Runtime 前保证：

- 顶层和任务字段精确匹配协议；
- 任务数、目标文本和 ID 均有界；
- ID 唯一，依赖存在且不重复；
- 不允许自依赖或依赖环；
- Planner 超时和失败转为稳定的 `PlannerError`；
- 调用取消原样传播，不包装为业务失败。

Planner 失败时 Runtime 以 `error_source="planner"` 终止，不调用 Router、执行模型或工具。

## 当前生命周期

1. composition root 仅在 `plan_execute` 模式创建 Planner；
2. Runtime 获取 session lease 并初始化本轮用户消息；
3. Planner 收到用户输入与工具能力摘要，返回严格 JSON；
4. 校验后的计划写入 `RuntimeState.metadata["plan"]` 并发出 `plan.created`；
5. Router 用总目标和全部节点目标选择候选工具；
6. 执行模型在原会话上下文之外收到只读的计划 system context；
7. 原有验证、工具执行、循环保护和唯一终止路径继续生效。

同一 session 仍由 `SessionCoordinator` 串行化；不同 session 可以并发规划。计划不写入
持久 Memory，避免内部控制消息污染后续用户会话。

## 可观测性与限制

状态记录计划和 `planner_duration_ms`，事件记录计划与耗时，日志记录任务数和耗时。
当前尚无节点级开始/完成/失败状态，因此不能声称 DAG 被确定性调度或可断点恢复。
最有价值的下一步是引入 Runtime 拥有的节点状态机，并把工具结果绑定到活动节点。

## 教学实现顺序

1. **计划数据模型**：完成 `PlanTask`、`TaskPlan` 及其只读转换；
2. **基础解析边界**：把 JSON 文本转换为模型，并拒绝错误字段、错误类型、空目标和
   超出配置上限的输入；
3. **DAG 语义校验**：检查 ID、未知依赖、自依赖和依赖环；
4. **异步 Planner 调用**：加入超时、取消和异常转换；
5. **Runtime 接线**：最后才把已经验证的计划交给 Router 和执行模型。

第二阶段的关键不是“会调用 `json.loads`”，而是理解转换前后的信任差异：JSON 文本来自
模型，属于不可信输入；只有完整通过字段、类型和资源上限检查后，才允许构造框架内部的
不可变 `TaskPlan`。
