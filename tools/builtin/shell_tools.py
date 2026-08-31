"""shell_exec — 只读 git 白名单执行（P0-1：彻底取消 shell 解释器）。

【历史漏洞】旧实现 subprocess.run(command, shell=True) 且只校验第一个空白 token：
`echo FIRST && echo PWNED` 首 token `echo` 在白名单内，第二段命令照常执行。
任何 `&&`/`&`/`|`（Unix 上还有 `;`/`$()`/反引号）都能注入任意命令。

【修复原则】
1. shell=False —— 不经过任何 shell，`&&`/`|`/`>` 退化为普通参数，无注入面
2. 参数化解析 —— shlex.split(posix=True) 处理引号 / 空格 / Unicode
3. 逐命令策略 —— CommandPolicy 只允许有限可执行文件 + 有限子命令（allowlist）
4. 危险参数拒绝 —— git 的 -C / -c / --git-dir / --ext-diff / --output 等一律拒绝
5. 环境白名单 —— 不继承父进程，只给最小集合 + GIT_CONFIG_NOSYSTEM
6. 固定 cwd（settings.exec_cwd 注入）+ 超时 + 输出字节上限

【演进路径】下一版拆成 git_status() / git_diff() / git_log() 专用工具，
把"任意命令字符串"从工具协议里彻底删除；CommandPolicy + _run_git 已铺路。
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from config import get_settings

# ── 硬限制 ──────────────────────────────────────────────────────────
_MAX_OUTPUT_BYTES = 5_000     # 单流 stdout / stderr 字节上限
_MAX_ARG_COUNT = 8            # 参数数量上限
_MAX_ARG_LENGTH = 200         # 单参数长度上限
_FORBIDDEN_EXTENSIONS = frozenset({".bat", ".cmd", ".ps1", ".vbs", ".com", ".exe"})

# git 长选项危险开关：改工作目录 / 覆盖配置 / 触发外部程序 / 重定向 / 变更操作
_GIT_FORBIDDEN_ARGS = frozenset({
    "-C", "-c",                              # 改 cwd / 覆盖配置（-c 可触发外部程序）
    "--git-dir", "--work-tree", "--exec-path",
    "-d", "-D", "-m", "-M", "--delete", "--move", "--copy",      # branch 等变更
    "--set-upstream-to", "--unset-upstream", "--edit-description",
    "--ext-diff", "--textconv", "--output", "-o", "--paginate",  # 外部程序 / 重定向
})
# git 短选项前缀：-cfoo=bar 这种附着值写法，按前两字符拦截（-d/-m 等在本工具
# 允许的只读子命令里没有合法用途，宁可过度拦截）
_GIT_FORBIDDEN_SHORT = frozenset({"-c", "-C", "-d", "-D", "-m", "-M", "-o", "-O"})


@dataclass(frozen=True)
class CommandPolicy:
    """单个可执行文件的执行策略：允许哪些子命令、超时多久。"""

    executable: str
    allowed_subcommands: frozenset[str]
    timeout_seconds: float


_COMMAND_POLICIES: dict[str, CommandPolicy] = {
    "git": CommandPolicy(
        executable="git",
        allowed_subcommands=frozenset({
            "status", "diff", "log", "show", "branch",
        }),
        timeout_seconds=30.0,
    ),
}

# 环境白名单：只继承这些键，其余父进程环境一概不带（防 GIT_DIR / GIT_PAGER 劫持）
_ENV_WHITELIST = ("PATH", "SystemRoot", "TEMP", "TMP", "HOME", "USERPROFILE")

_EXECUTABLE_CACHE: dict[str, str] = {}


def _exec_cwd() -> Path:
    """执行目录由配置注入（settings.exec_cwd），默认项目根。用户无法指定。"""
    configured = get_settings().exec_cwd.strip()
    return Path(configured).resolve() if configured else Path.cwd().resolve()


def _resolve_executable(name: str) -> str:
    """固定解析可执行文件绝对路径并缓存（防 PATH 劫持；缺则 fail-fast）。"""
    if name not in _EXECUTABLE_CACHE:
        path = shutil.which(name)
        if path is None:
            raise RuntimeError(f"executable not found: {name}")
        _EXECUTABLE_CACHE[name] = os.path.abspath(path)
    return _EXECUTABLE_CACHE[name]


def _build_env() -> dict[str, str]:
    """构造最小环境：白名单键 + 禁系统 gitconfig + 稳定输出编码。"""
    env = {k: os.environ[k] for k in _ENV_WHITELIST if k in os.environ}
    env["GIT_CONFIG_NOSYSTEM"] = "1"   # 防系统级 core.pager / alias 被外部配置劫持
    env["LC_ALL"] = "C"
    return env


# ── 纯校验层（不执行任何命令，可直接单测） ──────────────────────────


def _parse_argv(command: str) -> list[str]:
    """原始字符串 → 参数数组；拒绝空输入。

    posix=True 正确剥离引号（`git diff "a b.txt"` → ["a b.txt"]）；
    代价是反斜杠视为转义，Windows 路径须用 `/`（git 兼容）。
    """
    if not command.strip():
        raise ValueError("命令不能为空")
    argv = shlex.split(command, posix=True)
    if not argv:
        raise ValueError("命令不能为空")
    return argv


def _split_subcommand(argv: list[str], policy: CommandPolicy) -> tuple[str, list[str]]:
    """校验子命令属于策略 allowlist，返回 (subcommand, 剩余参数)。"""
    if len(argv) < 2:
        raise ValueError(f"命令缺少子命令: {argv[0]}")
    subcommand = argv[1]
    if subcommand not in policy.allowed_subcommands:
        raise ValueError(
            f"不允许的子命令: {subcommand} "
            f"(允许: {', '.join(sorted(policy.allowed_subcommands))})"
        )
    return subcommand, argv[2:]


def _arg_is_forbidden(arg: str) -> bool:
    """三通道拦截：精确 token / `--opt=value` 拆等号 / `-cfoo` 短选项前缀。"""
    if arg in _GIT_FORBIDDEN_ARGS:
        return True
    if arg.split("=", 1)[0] in _GIT_FORBIDDEN_ARGS:
        return True
    return arg[:2] in _GIT_FORBIDDEN_SHORT


def _validate_args(args: list[str]) -> None:
    """参数校验：数量 / 长度上限 + 危险参数拒绝 + Windows 脚本文件后缀。"""
    if len(args) > _MAX_ARG_COUNT:
        raise ValueError(f"参数过多: {len(args)} > {_MAX_ARG_COUNT}")
    for arg in args:
        if len(arg) > _MAX_ARG_LENGTH:
            raise ValueError(f"参数过长: {len(arg)} > {_MAX_ARG_LENGTH} 字符")
        if _arg_is_forbidden(arg):
            raise ValueError(f"禁止的参数: {arg}")
        if Path(arg).suffix.lower() in _FORBIDDEN_EXTENSIONS:
            raise ValueError(f"禁止引用可执行/脚本文件: {arg}")


# ── 执行层 ──────────────────────────────────────────────────────────


def _cap_output(text: str) -> str:
    """单流输出字节上限，超限截断并标注。"""
    if len(text) > _MAX_OUTPUT_BYTES:
        return text[:_MAX_OUTPUT_BYTES] + f"\n...[trimmed, original {len(text)} chars]"
    return text


def shell_exec(command: Annotated[str, "要执行的只读 git 命令，如 'git status'、'git log --oneline -5'、'git diff main'。仅允许 status/diff/log/show/branch；参数含空格需加引号；路径用 / 分隔。"] ) -> dict:
    """执行白名单内的只读 git 命令，返回 {command, return_code, stdout, stderr}。

    安全边界（P0-1）：shell=False + 参数化解析 + 逐子命令 allowlist +
    危险参数拒绝 + 环境白名单 + 固定 cwd，详见模块 docstring。
    """
    argv = _parse_argv(command)
    cmd_name = argv[0]
    policy = _COMMAND_POLICIES.get(cmd_name)
    if policy is None:
        raise ValueError(f"不允许的命令: {cmd_name} (仅允许: {', '.join(sorted(_COMMAND_POLICIES))})")

    subcommand, args = _split_subcommand(argv, policy)
    _validate_args(args)

    full_argv = [
        _resolve_executable(policy.executable),
        "--no-pager",        # 强制关分页，防 core.pager / PAGER 劫持
        subcommand,
        *args,
    ]

    try:
        result = subprocess.run(
            full_argv,
            shell=False,     # 无 shell = 无注入面
            cwd=_exec_cwd(),  # 由配置注入，用户无法指定
            env=_build_env(),
            capture_output=True,
            timeout=policy.timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return {
            "command": command,
            "return_code": -1,
            "stdout": "",
            "stderr": f"命令超时 ({policy.timeout_seconds}s)，已终止",
        }

    return {
        "command": command,
        "return_code": result.returncode,
        "stdout": _cap_output(result.stdout.decode("utf-8", errors="replace")),
        "stderr": _cap_output(result.stderr.decode("utf-8", errors="replace")),
    }


TOOLS = [shell_exec]
