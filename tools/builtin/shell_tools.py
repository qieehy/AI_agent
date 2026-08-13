import subprocess
from typing import Annotated

_ALLOWED_COMMANDS: frozenset[str] = frozenset({
    "git", "ls", "dir", "cat", "type", "echo",
    "pwd", "find", "head", "tail", "wc", "grep",
})

_SHELL_TIMEOUT = 30  # 命令超时（秒）
_MAX_OUTPUT_CHARS = 5_000  # stdout + stderr 总截断上限

def _check_whitelist(command: str) -> str:
    cmd_name = command.strip().split()[0]
    if cmd_name not in _ALLOWED_COMMANDS:
        raise ValueError(f"Invalid command: {cmd_name}")
    return cmd_name


def shell_exec(command: Annotated[str, "要执行的shell命令"]) -> dict:
    """执行白名单内的系统命令，返回 {command, return_code, stdout, stderr}。"""
    _check_whitelist(command)

    try:
        result = subprocess.run(
            command,
            shell=True,          # 让 shell 解析命令（支持 git status 这种带空格的）
            capture_output=True,     # 捕获 stdout + stderr
            timeout=_SHELL_TIMEOUT,
            text=True,          # 返回 str 而不是 bytes
        )
    except subprocess.TimeoutExpired:
        return {
            "command": command,
            "return_code": -1,
            "stdout": "",
            "stderr": f"命令超时 ({_SHELL_TIMEOUT}s)，已终止",
        }
    stdout = result.stdout
    stderr = result.stderr

    if len(stdout) > _MAX_OUTPUT_CHARS:
        stdout = stdout[:_MAX_OUTPUT_CHARS] + f"\n... [trimmed, original {len(stdout)} chars]"
    if len(stderr) > _MAX_OUTPUT_CHARS:
        stderr = stderr[:_MAX_OUTPUT_CHARS] + f"\n... [trimmed, original {len(stderr)} chars]"

    return {
        "command": command,
        "return_code": result.returncode,
        "stdout": stdout,
        "stderr": stderr,
    }

TOOLS = [shell_exec]
