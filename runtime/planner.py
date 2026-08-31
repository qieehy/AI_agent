import asyncio
import math
import re
import json
from dataclasses import dataclass
from typing import Callable, Awaitable, Any, TypeAlias

from errors import PlannerError


_TASK_ID = re.compile(
    r"^[A-Za-z][A-Za-z0-9_-]{0,63}$"
)

@dataclass(frozen=True, slots=True)
class PlanTask:
    id: str
    goal: str
    dependencies: tuple[str, ...] = ()  #本task所依赖的task (存id) : 默认为空tuple

    def to_dict(self) -> dict[str, object]:
        return {"id": self.id, "goal": self.goal, "dependencies": list(self.dependencies)}

@dataclass(frozen=True, slots=True)
class TaskPlan:
    goal: str
    tasks: tuple[PlanTask, ...]

    def to_dict(self) -> dict[str, object]:
        return {"goal": self.goal, "tasks": [task.to_dict() for task in self.tasks]}

    def routing_query(self) -> str:
        goals = [self.goal]

        for task in self.tasks:
            goals.append(task.goal)

        return "\n".join(goals)

    def execution_context(self) -> str:
        plan_json = json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))

        return (
            "Use this validated execution DAG. "
            "Respect dependencies and do not invent tasks.\n"
            + plan_json
        )

def parse_task_plan(content: object, *, max_tasks: int = 12, max_goal_chars: int = 1000) -> TaskPlan:
    """将 planner 请求 llm 返回的文本 加工成 TaskPlan"""
    if not isinstance(content, str):
        raise PlannerError("planner response content must be JSON text")

    try:
        raw = json.loads(content)
    except json.JSONDecodeError as e:
        raise PlannerError("planner response content is not valid JSON") from e

    if not isinstance(raw, dict) or set(raw) != {"goal", "tasks"}:
        raise PlannerError("planner response must contain only goal and tasks")

    raw_tasks = raw["tasks"]

    if not isinstance(raw_tasks, list) or not 1 <= len(raw_tasks) <= max_tasks:
        raise PlannerError("planner task count is outside of configured bounds", context={"max_tasks": max_tasks})

    goal = _bounded_text(raw["goal"], "goal", max_goal_chars)

    tasks: list[PlanTask] = []
    for index, item in enumerate(raw_tasks):
        if (
                not isinstance(item, dict)
                or set(item) != {"id", "goal", "dependencies"}
        ):
            raise PlannerError(
                "each planner task must contain only "
                "id, goal, and dependencies",
                context={"task_index": index},
            )

        task_id = item["id"]
        if (
                not isinstance(task_id, str)
                or not _TASK_ID.fullmatch(task_id)
        ):
            raise PlannerError(
                "planner task id is invalid",
                context={"task_index": index},
            )

        task_goal = _bounded_text(item["goal"], "task goal", max_goal_chars)

        raw_dependencies = item["dependencies"]

        if (
                not isinstance(raw_dependencies, list)
                or not all(isinstance(dep, str) for dep in raw_dependencies)
                or len(raw_dependencies) != len(set(raw_dependencies))
        ):
            raise PlannerError(
                "planner dependencies must be a list of unique task ids"
            )
        tasks.append(
            PlanTask(
                id=task_id,
                goal=task_goal,
                dependencies=tuple(raw_dependencies)
            )
        )
    ids = [task.id for task in tasks]
    if len(ids) != len(set(ids)):
        raise PlannerError(
            "planner task ids must be unique"
        )
    known_ids = set(ids)
    for task in tasks:
        unknown = set(task.dependencies) - known_ids
        if unknown:
            raise PlannerError(
                "planner task references unknown dependencies",
                context={
                    "task_id": task.id,
                    "unknown": sorted(unknown),
                },
            )
        if task.id in task.dependencies:
            raise PlannerError(
                "planner task cannot depend on itself",
                context={"task_id": task.id},
            )
    _validate_no_dependency_cycle(tasks)
    return TaskPlan(
        goal=goal,
        tasks=tuple(tasks),
    )

def _bounded_text(value: object, field: str, max_goal_chars: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlannerError(
            f"planner {field} must be non-empty text"
        )

    normalized = value.strip()
    if len(normalized) > max_goal_chars:
        raise PlannerError(f"planner {field} exceeds the configured size limit",
        context={"max_goal_chars": max_goal_chars},)
    return normalized

def _validate_no_dependency_cycle(tasks: list[PlanTask]) -> None:
    remaining = {
        task.id: set(task.dependencies)
        for task in tasks
    }
    resolved: set[str] = set()
    while remaining:
        ready = [
            task_id
            for task_id, dependencies in remaining.items()
            if dependencies.issubset(resolved)
        ]
        if not ready:
            raise PlannerError(
                "planner tasks contain a dependency cycle"
            )
        for task_id in ready:
            resolved.add(task_id)
            del remaining[task_id]

PlannerLLMCallable: TypeAlias = Callable[
    [list[dict], list[dict] | None],
    Awaitable[Any],
]

class Planner:
    def __init__(
            self,
            llm_call: PlannerLLMCallable,
            *,
            timeout_s: float = 30.0,
            max_tasks: int = 12,
            max_goal_chars: int = 1000,
    ) -> None:
        if not math.isfinite(timeout_s):
            raise ValueError(
                "timeout_s must be finite"
            )

        if timeout_s <= 0:
            raise ValueError(
                "timeout_s must be greater than 0"
            )

        if max_tasks <= 0:
            raise ValueError(
                "max_tasks must be greater than 0"
            )

        if max_goal_chars <= 0:
            raise ValueError(
                "max_goal_chars must be greater than 0"
            )
        self._llm_call = llm_call
        self._timeout_s = timeout_s
        self._max_tasks = max_tasks
        self._max_goal_chars = max_goal_chars

    async def plan(self, user_input: str, tool_schemas: list[dict]) -> TaskPlan:
        messages = [
            {
                "role": "system",
                "content": self._prompt(tool_schemas),
            },
            {
                "role": "user",
                "content": user_input,
            },
        ]
        try:
            response = await asyncio.wait_for(
                self._llm_call(messages, None),
                timeout=self._timeout_s,
            )
        except asyncio.TimeoutError as exc:
            # Planner 自己的超时策略
            raise PlannerError(
                "planner request timed out",
                context={"timeout_s": self._timeout_s},
            ) from exc
        except asyncio.CancelledError:
            # 上层取消，必须原样传播
            raise
        except Exception as exc:
            # 其他 LLM/供应商异常
            raise PlannerError(
                "planner request failed"
            ) from exc
        try:
            content = response.choices[0].message.content
        except (
                AttributeError,
                IndexError,
                TypeError,
                KeyError,
        ) as exc:
            raise PlannerError(
                "planner response has an invalid shape"
            ) from exc

        return parse_task_plan(
            content,
            max_tasks=self._max_tasks,
            max_goal_chars=self._max_goal_chars,
        )



    @staticmethod
    def _prompt(tool_schemas: list[dict]) -> str:
        capabilities = [
            {
                "name": schema.get("function", {}).get("name"),
                "description": schema.get(
                    "function", {}
                ).get("description", ""),
            }
            for schema in tool_schemas
        ]
        return (
                "You are a planning component. "
                "Decompose the request into the smallest useful DAG. "
                "Return JSON only, with exactly this shape: "
                '{"goal":"...","tasks":['
                '{"id":"task_1","goal":"...",'
                '"dependencies":[]}]} . '
                "Use one task for a simple request. "
                "Dependencies must reference task ids. "
                "Do not execute tasks and do not claim results. "
                "Available capabilities: "
                + json.dumps(
            capabilities,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        )

