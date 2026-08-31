# AGENTS.md

## 1. Project mission

This repository is a learning-oriented, enterprise-grade Python Agent Framework.

The project is educational because every important design is explained, tested, reviewed, and documented—not because the production implementation is simplified into a toy.

The target is to learn how real Agent systems should be designed and operated, including:

- Agent Runtime and state-machine design;
- tool registration, routing, validation, permissions, and execution;
- async execution, streaming, cancellation, backpressure, and concurrency safety;
- prompt patterns and context engineering;
- session memory and persistent state;
- RAG, hybrid retrieval, reranking, citation, and evaluation;
- error taxonomy, observability, security, performance, and deployment readiness.

Correctness, explicit boundaries, operational safety, and maintainability take priority over feature count and superficial demonstrations.

## 2. Enterprise-grade learning policy

Do not weaken production design merely to make a concept easier to teach.

Teaching must happen through:

- clear explanations;
- execution timelines;
- focused diagrams;
- small vertical implementation slices;
- tests for invariants and failure paths;
- comparison with alternatives;
- documentation of decisions and limitations;
- isolated educational examples outside production paths.

The production path must not use a knowingly unsafe, incomplete, or misleading implementation when a proper design is within scope.

A learning example may be simplified only when:

- it is clearly labeled as an isolated example;
- it is not wired into the production Runtime;
- its omitted guarantees are stated explicitly;
- the enterprise implementation is still explained;
- it cannot be mistaken for the recommended production design.

“Minimal change” means narrow scope and reviewable diff. It does not mean weaker validation, missing cleanup, unsafe defaults, or untested failure paths.

“Build it in stages” means delivering small but complete vertical slices. It does not mean leaving half-connected classes that appear implemented but are unused in production.

Enterprise-grade does not mean adding distributed infrastructure or abstractions without a demonstrated need. Avoid both toy implementations and premature overengineering.

## 3. Core engineering principles

When working in this repository:

- Preserve existing user changes and uncommitted work.
- Keep each change focused on the requested problem.
- Do not refactor unrelated modules unless explicitly requested.
- Prefer explicit designs over clever or hidden behavior.
- Prefer narrow interfaces and clear ownership.
- Make invalid states difficult to represent.
- Use secure and defensive defaults.
- Fail closed at security and permission boundaries.
- Do not use prompts as a substitute for validation, authorization, concurrency control, or isolation.
- Verify production wiring, not only class existence and unit tests.
- Treat tests, types, schemas, logs, and documentation as external project memory.
- Record intentional technical debt instead of disguising it as completed functionality.
- Ask before destructive operations, external writes, new production dependencies, or material expansion of scope.

## 4. Request modes

Determine the request type before acting.

### Explain or learn

When asked to explain:

- inspect the actual implementation;
- do not modify files unless explicitly requested;
- trace the real control flow from the production entry point;
- explain object ownership and lifecycle;
- identify every important `await`, state transition, exception boundary, and cleanup path;
- distinguish code behavior from comments, documentation, and assumptions;
- explain what guarantees exist and which guarantees are absent;
- compare the implementation with the nearest production alternative;
- identify any teaching shortcut that would be unacceptable in production;
- provide small exercises when useful.

Do not replace technical accuracy with an overly simplified analogy. An analogy may support the explanation, but the real mechanism must also be explained.

### Review or diagnose

When asked to review or diagnose:

- inspect implementation, wiring, configuration, tests, and relevant runtime behavior;
- do not implement a fix unless requested;
- cite concrete files and code locations;
- prioritize findings as P0, P1, or P2;
- describe the triggering conditions;
- describe the user-visible or operational consequence;
- distinguish confirmed defects from plausible risks;
- review architecture boundaries, types, concurrency, error handling, defensive design, testability, observability, performance, security, and technical debt;
- recommend the smallest high-value next change.

### Plan or design

For a non-trivial change, establish:

1. current behavior;
2. desired behavior;
3. actors and ownership;
4. state and lifecycle;
5. invariants;
6. trust boundaries;
7. normal path;
8. invalid-input path;
9. failure path;
10. timeout path;
11. cancellation path;
12. concurrency path;
13. recovery path;
14. observability requirements;
15. alternatives and tradeoffs;
16. acceptance tests;
17. known limitations.

Do not propose an architecture merely because it sounds enterprise-grade. Tie every component to a concrete invariant, threat, scale assumption, or operational need.

### Implement or fix

When asked to implement:

- implement the smallest complete production-quality slice;
- preserve existing behavior unless the change intentionally modifies it;
- add or update behavioral tests;
- run narrow tests first;
- run broader regression checks when practical;
- verify production construction and invocation;
- include failure, timeout, cancellation, and concurrency behavior where relevant;
- update documentation if architecture or public behavior changes;
- report blockers instead of hiding or bypassing them.

## 5. Evidence and hallucination control

Classify important conclusions as:

- **Observed:** directly supported by code, configuration, test output, logs, or tool output;
- **Inferred:** reasoned from evidence but not directly executed or observed;
- **Unknown:** evidence is insufficient.

In reports:

- cite concrete files and code locations;
- state material assumptions;
- never invent command output, test results, API behavior, file contents, benchmarks, or deployment guarantees;
- never claim tests passed unless successful output was observed;
- report skipped, blocked, timed-out, flaky, or uncollected tests;
- treat README files and comments as declared intent;
- treat executable code, production wiring, and observed tests as current behavior;
- identify disagreement between documentation and implementation;
- verify current external facts with authoritative sources.

A class plus unit tests does not prove a feature is implemented. Confirm that the production composition root creates it, injects it, invokes it, handles its failures, and observes its results.

## 6. Architecture boundaries

Keep responsibilities explicit:

- `main.py`: composition root and CLI wiring;
- `runtime/`: run lifecycle, state transitions, orchestration, loop protection, cancellation, and termination;
- `llm/`: provider adapters, API communication, streaming assembly, retries, timeout, and error translation;
- `tools/registry.py`: tool metadata, registration, and schema generation;
- `tools/router.py`: candidate-tool retrieval and routing decisions;
- `tools/validator.py`: validation of untrusted model-generated calls;
- `tools/executor.py`: execution scheduling, isolation, deadlines, and result translation;
- `memory/`: conversation context, session ownership, persistence, and retention;
- `prompts/`: prompt construction and behavioral contracts;
- `rag/`: ingestion, chunking, embedding, indexing, retrieval, fusion, reranking, citation, and evaluation;
- `observability/`: structured logs, trace context, metrics, events, and audit data;
- `errors/`: stable framework exception taxonomy;
- `config/`: validated configuration and environment boundaries.

Do not move responsibilities between modules without explaining the dependency direction and operational consequence.

Production dependency construction belongs in the composition root. Avoid hidden module-level mutable singletons.

## 7. Type and boundary standards

- Add precise type annotations at public and component boundaries.
- Prefer typed dataclasses, enums, protocols, generics, or Pydantic models over expanding `Any` and raw dictionaries.
- Convert provider SDK objects into internal models at adapter boundaries.
- Do not spread OpenAI-compatible SDK object shapes throughout Runtime and Tool modules.
- Validate untrusted data before execution.
- Validate configuration at startup and fail fast on invalid values.
- Preserve exception causes with `raise ... from exc`.
- Do not silently coerce unsupported values with `str()` merely to avoid an error.
- Define serialization contracts explicitly.
- Use stable identifiers for run, session, step, tool call, and trace relationships.

Type safety must help preserve behavior across refactors; it is not only for satisfying a checker.

## 8. Runtime and state-machine standards

Runtime behavior must have:

- explicit initial, running, waiting, successful, failed, canceled, and guarded terminal states;
- one clearly defined termination path;
- structured stop reasons;
- bounded steps and loop protection;
- consistent event emission;
- deterministic message ordering;
- explicit ownership of session mutation;
- cancellation-safe cleanup;
- no silent fall-through from invalid states.

State transitions must be testable. Invalid transitions should fail explicitly or be impossible by construction.

Do not maintain multiple unsynchronized sources of truth for messages, steps, status, or errors.

## 9. Async and concurrency standards

- Do not block the event loop with synchronous I/O, subprocesses, model inference, embedding, reranking, or CPU-heavy work.
- Run blocking synchronous work through an explicitly bounded worker mechanism.
- Await native async work directly.
- Treat cancellation as a first-class terminal path.
- Use `try/finally` or `async with` for resource ownership.
- Never release a lock that the current operation did not successfully acquire.
- Protect complete consistency boundaries rather than individual container mutations.
- Same-session runs must not interleave conversation turns.
- Different sessions should remain concurrent where safe.
- Separate lock-acquisition timeout, LLM timeout, tool timeout, and whole-run timeout.
- Apply both global and per-resource concurrency limits where needed.
- Prevent unbounded task creation and unbounded queues.
- Define backpressure behavior.
- State whether a guarantee applies to one coroutine, one event loop, one process, or a distributed deployment.

An in-memory `asyncio.Lock` is not a distributed lock. Multiprocess or multi-instance guarantees require an external coordination mechanism, fencing/versioning, or a different ownership architecture.

## 10. Tool and security standards

- Treat model-generated tool calls as untrusted input.
- Validate tool name, arguments, schema, permissions, size, and policy before execution.
- Never use `shell=True` with model-generated input.
- Prefer dedicated typed tools over generic command-string execution.
- Apply executable and argument allowlists.
- Enforce filesystem roots with resolved paths and symlink-aware checks.
- Do not expose secrets through logs, prompts, errors, tool results, fixtures, or snapshots.
- Network tools must account for redirects, DNS resolution, rebinding, private addresses, timeouts, content types, and response-size limits.
- Represent tool failures as structured failures, not successful strings containing error text.
- Add idempotency protection for side-effecting operations where appropriate.
- Record audit information without logging sensitive payloads.
- Give high-risk tools stricter isolation, permission, timeout, and concurrency policies.

A model instruction such as “only run safe commands” is not a security boundary.

## 11. Router and prompt standards

Tool routing must be evaluated as a retrieval system, not judged from a few examples.

- Verify Router integration in the Runtime production path.
- Build tool representations from meaningful capability information.
- Version embedding models and cached vectors.
- Define top-k, threshold, fallback, and no-match behavior explicitly.
- Measure recall, precision, latency, and tool-exposure rate with representative queries.
- Avoid silently exposing all tools when a secure fail-closed policy is required.
- Do not retain unused keyword-routing code after intentionally adopting embedding-only routing.
- Keep prompts concise and avoid duplicate rules.
- Expose only tools relevant to the current request.
- Treat prompt patterns as behavior contracts; do not name a static string “Planner” or “Reflection” unless Runtime behavior supports those stages.

## 12. RAG standards

Separate:

- document loading;
- normalization;
- chunking;
- embedding;
- indexing;
- retrieval;
- hybrid fusion;
- reranking;
- context construction;
- answer generation;
- citation verification;
- evaluation.

Requirements:

- validate document size, encoding, vector dimensions, and batch size;
- avoid blocking async request paths;
- keep dense, sparse, metadata, and document ledgers consistent;
- make document replacement atomic or provide rollback;
- support deterministic IDs and versioning;
- treat retrieved content as untrusted data, never as system instructions;
- enforce context budgets;
- preserve source metadata needed to verify citations;
- measure retrieval quality with an evaluation dataset;
- distinguish retrieval score from calibrated confidence;
- define behavior for no result, weak result, partial failure, and unavailable reranker.

Do not choose thresholds or claim retrieval quality based only on intuition.

## 13. Error-handling standards

- Use a stable error taxonomy.
- Preserve root causes internally.
- Return safe, structured error information across module boundaries.
- Do not leak secrets or sensitive system details.
- Distinguish retryable, permanent, invalid-input, policy, timeout, canceled, and internal errors.
- Bound retries with exponential backoff and jitter.
- Do not retry non-idempotent operations unless idempotency is guaranteed.
- Do not retry forever.
- Make partial failure semantics explicit.
- Ensure errors do not leave locks, sessions, indexes, files, or message histories in inconsistent states.

## 14. Observability standards

Important operations should expose:

- `trace_id`;
- `run_id`;
- `session_id`;
- step index;
- component and operation;
- duration;
- outcome;
- structured error category;
- retry count;
- queue or lock wait time;
- selected tools and routing metadata where safe;
- model and configuration version;
- token or usage metrics where available.

Logs must be structured, bounded, and sanitized.

Logs are not enough for production readiness. Identify useful metrics, traces, audit events, and alerts for important failure modes.

## 15. Performance standards

Do not optimize from intuition alone.

Before changing performance-sensitive code:

- identify the expected workload;
- determine complexity;
- measure a baseline;
- define latency, throughput, memory, and cost targets;
- benchmark representative inputs;
- avoid benchmarks dominated by test setup or network noise;
- preserve correctness while optimizing.

Watch for:

- repeated full-list token counting;
- quadratic BM25 calculations;
- repeated model loading;
- unbounded embedding batches;
- unnecessary schema transmission;
- blocking inference inside async paths;
- repeated serialization;
- unbounded caches;
- N+1 API or storage calls.

Report benchmark conditions with results.

## 16. Testing and quality gates

Use the project environment when available:

```powershell
.\.venv\Scripts\python.exe -m pytest <relevant-tests> -q -p no:cacheprovider
```

Then, when practical:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy runtime tools memory rag prompts llm errors
```

Testing requirements:

- test public behavior and invariants;
- cover normal, invalid-input, failure, timeout, cancellation, concurrency, and recovery paths where relevant;
- use deterministic events, barriers, and fake clocks instead of fragile timing-only assertions;
- keep unit tests independent of real network and credentials;
- classify external API and model tests as integration tests;
- add regression tests for every confirmed defect;
- test production wiring separately from component behavior;
- verify cleanup and absence of leaked resources;
- test security controls with bypass attempts;
- do not weaken assertions to make tests pass;
- do not mock away the behavior the test is intended to verify.

If the full suite is blocked by a pre-existing unrelated problem, run the relevant subset and report:

- the passing focused result;
- the full-suite blocker;
- why it is unrelated;
- what remains unverified.

## 17. Enterprise readiness review

Do not call a component enterprise-grade solely because it has:

- classes and interfaces;
- async syntax;
- retries;
- logs;
- unit tests;
- a security comment;
- an allowlist;
- a design-pattern name.

Evaluate readiness across:

- correctness;
- isolation;
- type safety;
- concurrency safety;
- failure recovery;
- cancellation;
- security and trust boundaries;
- configuration validation;
- observability;
- performance and resource bounds;
- test depth;
- upgrade and migration behavior;
- operational ownership;
- deployment model;
- documentation and supportability.

If an enterprise requirement is intentionally deferred, record it as a limitation or technical-debt item with its triggering scale or deployment condition.

## 18. Learning-first explanation requirements

After a non-trivial implementation, explain:

1. the original problem;
2. the violated invariant;
3. the selected production design;
4. object ownership;
5. chronological execution;
6. normal path;
7. invalid-input path;
8. failure path;
9. timeout path;
10. cancellation path;
11. concurrency behavior;
12. cleanup behavior;
13. observability;
14. alternatives rejected and why;
15. guarantees added;
16. guarantees not provided;
17. how tests prove the behavior;
18. a small exercise for the user.

Do not simplify the production implementation for explanation. Explain the real implementation at the user’s current level, introducing prerequisite concepts as needed.

## 19. Documentation as external memory

Persist stable knowledge in:

- `AGENTS.md`: recurring Codex working rules;
- `README.md`: setup and user-facing behavior;
- `doc/architecture.md`: architecture, boundaries, and control flow;
- `doc/glossary.md`: Python, async, Agent, RAG, and operations terminology;
- `doc/decisions/`: architectural decision records;
- `doc/runbooks/`: operational diagnosis and recovery;
- `doc/threat-model.md`: assets, trust boundaries, threats, and mitigations;
- `doc/roadmap.md`: planned capability and explicit technical debt;
- `CHANGELOG.md`: user-visible changes;
- tests: executable behavioral contracts.

Documentation must describe current production behavior. Clearly label planned or unimplemented behavior.

When the user corrects a recurring assumption, propose a concise update to project instructions or architecture documentation.

## 20. Git and workspace discipline

Before editing:

- inspect `git status`;
- identify modified and untracked files;
- understand whether they belong to the user;
- avoid overlapping unrelated changes.

During editing:

- keep diffs focused and reviewable;
- avoid formatting unrelated files;
- do not delete or revert user work;
- do not hide failures by changing unrelated configuration;
- do not commit, push, reset, or switch branches unless explicitly requested.

After editing:

- inspect the final diff;
- run `git diff --check`;
- list task-owned changes;
- identify unrelated pre-existing work that remains.

## 21. Definition of done

A change is complete only when:

- the requested behavior is implemented;
- production wiring is verified;
- relevant invariants are enforced in code;
- behavioral and regression tests cover the important paths;
- focused validation passes;
- broader validation is attempted when practical;
- errors and cancellation leave consistent state;
- security boundaries do not depend on model cooperation;
- observability is sufficient to diagnose material failures;
- documentation matches actual behavior;
- no unrelated user work was overwritten;
- remaining limitations and technical debt are stated honestly.

The final report should contain:

1. outcome;
2. important design decisions;
3. changed files;
4. production wiring;
5. validation performed and exact results;
6. blocked or unverified checks;
7. guarantees added;
8. enterprise limitations;
9. technical debt;
10. the most valuable next step.

Keep routine reports concise. Provide full teaching detail when the user asks to understand the implementation.

## 22. Teaching-pair workflow

By default, the user implements production code incrementally. Codex should:

- explain the underlying concept before introducing specialized terminology;
- write behavioral tests that define the next production requirement;
- write teaching and design documentation, clearly marking planned behavior as unimplemented;
- review the user's implementation and explain test, type, and runtime failures;
- not create or modify production code unless the user explicitly authorizes direct implementation.
