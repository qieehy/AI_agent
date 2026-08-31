"""P0-1: shell_exec 安全契约测试。

策略：
- 纯校验层（_parse_argv / _split_subcommand / _validate_args）直接测，不碰真命令
- 执行层用 monkeypatch 替身：断言传入 subprocess 的是「参数数组 + shell=False」，
  结构上保证 `&&`/`|`/`>` 只会是普通参数、永不被 shell 解析
- 真 git 集成测试用 skipif 守卫（无 git 环境自动跳过），并验证无副作用文件
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from tools.builtin import shell_tools
from tools.builtin.shell_tools import (
    _COMMAND_POLICIES,
    _GIT_FORBIDDEN_ARGS,
    CommandPolicy,
    _arg_is_forbidden,
    _parse_argv,
    _split_subcommand,
    _validate_args,
    shell_exec,
)

GIT_AVAILABLE = shutil.which("git") is not None

requires_git = pytest.mark.skipif(not GIT_AVAILABLE, reason="git not on PATH")


# ── 策略表 ──────────────────────────────────────────────────────────


def test_git_policy_only_allow_read_only_subcommands():
    """v1 只允许只读子命令，变更类子命令必须在 allowlist 之外。"""
    policy = _COMMAND_POLICIES["git"]
    assert isinstance(policy, CommandPolicy)
    assert policy.allowed_subcommands == {"status", "diff", "log", "show", "branch"}
    # 变更/配置类子命令不允许
    assert {"config", "alias", "clean", "reset", "checkout", "restore",
            "push", "fetch", "pull"} & policy.allowed_subcommands == set()


def test_git_forbidden_args_cover_spec_requirement():
    """spec 要求的 git 危险参数必须全部在禁止集。

    变更/配置类子命令（config/alias/clean/reset/checkout/restore/push/fetch/pull）
    由 _split_subcommand 的 allowlist 拦截；这里只验证参数级危险项。
    """
    assert {"-C", "-c", "--git-dir", "--work-tree", "--exec-path"} <= _GIT_FORBIDDEN_ARGS
    # 我们额外封堵的外部程序 / 重定向 / 变更参数
    assert {"--ext-diff", "--textconv", "--output", "-d", "-D", "-m", "-M"} <= _GIT_FORBIDDEN_ARGS


# ── 解析层：引号 / 空格 / Unicode ──────────────────────────────────


def test_parse_argv_strips_quotes_and_keeps_spaces():
    assert _parse_argv('git diff "a b.txt"') == ["git", "diff", "a b.txt"]


def test_parse_argv_handles_unicode():
    assert _parse_argv("git log --grep=你好") == ["git", "log", "--grep=你好"]


def test_parse_argv_rejects_empty():
    for bad in ["", "   "]:
        with pytest.raises(ValueError, match="不能为空"):
            _parse_argv(bad)


def test_parse_argv_rejects_unbalanced_quote():
    with pytest.raises(ValueError):
        _parse_argv('git diff "unclosed')


# ── 子命令 allowlist ───────────────────────────────────────────────


def test_split_subcommand_accepts_allowed():
    policy = _COMMAND_POLICIES["git"]
    sub, args = _split_subcommand(["git", "log", "--oneline", "-5"], policy)
    assert sub == "log"
    assert args == ["--oneline", "-5"]


def test_split_subcommand_rejects_unknown():
    policy = _COMMAND_POLICIES["git"]
    with pytest.raises(ValueError, match="不允许的子命令"):
        _split_subcommand(["git", "checkout", "main"], policy)


def test_split_subcommand_rejects_missing():
    policy = _COMMAND_POLICIES["git"]
    with pytest.raises(ValueError, match="缺少子命令"):
        _split_subcommand(["git"], policy)


# ── 危险参数三通道拦截 ─────────────────────────────────────────────


@pytest.mark.parametrize("arg", [
    "-C", "-c", "--git-dir", "--work-tree", "--exec-path",
    "--ext-diff", "--textconv", "--output", "--paginate",
    "-d", "-D", "-m", "-M", "--delete", "--move", "--copy",
])
def test_arg_is_forbidden_exact(arg):
    assert _arg_is_forbidden(arg)


@pytest.mark.parametrize("arg", [
    "--git-dir=/tmp/repo",       # 等号连写
    "--work-tree=/tmp/wt",
    "--output=out.patch",
    "-c core.pager=cat",         # 附着值短选项
    "-C/tmp",
    "-cfoo.bar=1",
])
def test_arg_is_forbidden_eq_and_attached_forms(arg):
    assert _arg_is_forbidden(arg)


def test_arg_is_forbidden_allows_read_only_flags():
    for ok in ["--oneline", "-5", "--stat", "--porcelain", "-s", "-w", "-U5",
               "--grep=bug", "main", "HEAD", ".", "src/" , "README.md"]:
        assert not _arg_is_forbidden(ok), f"{ok!r} 不应被拦截"


# ── validate_args：数量 / 长度 / 脚本后缀 ──────────────────────────


def test_validate_args_rejects_too_many():
    with pytest.raises(ValueError, match="参数过多"):
        _validate_args([str(i) for i in range(9)])


def test_validate_args_rejects_too_long():
    with pytest.raises(ValueError, match="参数过长"):
        _validate_args(["x" * 201])


def test_validate_args_rejects_script_extension():
    with pytest.raises(ValueError, match="可执行/脚本文件"):
        _validate_args(["run.bat"])


def test_validate_args_rejects_forbidden_token():
    with pytest.raises(ValueError, match="禁止的参数"):
        _validate_args(["--git-dir=/tmp"])


# ── 执行层：注入载荷必须变成普通参数，绝不经过 shell ───────────────


def test_exec_passes_argv_and_shell_false(monkeypatch):
    """`git status & whoami` → subprocess 收到的是参数数组 & shell=False。"""
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(argv, 0, stdout=b"ok\n", stderr=b"")

    monkeypatch.setattr(shell_tools.subprocess, "run", fake_run)
    monkeypatch.setattr(shell_tools, "_resolve_executable", lambda name: f"/usr/bin/{name}")

    result = shell_exec("git status & whoami")

    assert captured["kwargs"]["shell"] is False      # 无 shell = 无注入
    assert captured["kwargs"]["cwd"] is not None     # 固定 cwd
    # `&` 退化为 git 的普通参数，而不是 shell 连接符
    assert captured["argv"] == ["/usr/bin/git", "--no-pager", "status", "&", "whoami"]
    assert result["return_code"] == 0
    assert result["stdout"] == "ok\n"


def test_exec_passes_pipe_and_redirect_as_args(monkeypatch):
    """`git status | whoami` 与 `git status > x.txt` 同样只是参数。"""
    captured = []

    def fake_run(argv, **kwargs):
        captured.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(shell_tools.subprocess, "run", fake_run)
    monkeypatch.setattr(shell_tools, "_resolve_executable", lambda name: f"/usr/bin/{name}")

    shell_exec("git status | whoami")
    shell_exec("git status > x.txt")

    assert captured[0] == ["/usr/bin/git", "--no-pager", "status", "|", "whoami"]
    assert captured[1] == ["/usr/bin/git", "--no-pager", "status", ">", "x.txt"]


def test_exec_rejects_config_override_before_running(monkeypatch):
    """git -c / --git-dir 必须在执行前被拒绝（monkeypatch 不应被调到）。

    全局危险选项落在子命令位（argv[1]）时按"不允许的子命令"拒绝；
    放在子命令之后时按"禁止的参数"拒绝——两条路径都拦截、都不执行。
    """
    called = []

    def fake_run(argv, **kwargs):
        called.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(shell_tools.subprocess, "run", fake_run)

    # 子命令位（-c / --git-dir 出现在 log / status 之前）
    with pytest.raises(ValueError):
        shell_exec("git -c core.pager=cat log")
    with pytest.raises(ValueError):
        shell_exec("git --git-dir=/tmp/x status")
    # 子命令之后（走 _validate_args 的"禁止的参数"路径）
    with pytest.raises(ValueError, match="禁止的参数"):
        shell_exec("git status --git-dir=/tmp/x")

    assert called == []      # 一次都没执行


def test_exec_rejects_branch_mutation(monkeypatch):
    """`git branch -D x` 违反只读承诺，必须拒绝。"""
    monkeypatch.setattr(shell_tools.subprocess, "run", lambda *a, **k: None)
    with pytest.raises(ValueError, match="禁止的参数"):
        shell_exec("git branch -D feature/x")


def test_exec_rejects_unknown_command(monkeypatch):
    monkeypatch.setattr(shell_tools.subprocess, "run", lambda *a, **k: None)
    with pytest.raises(ValueError, match="不允许的命令"):
        shell_exec("python -c 'print(1)'")


def test_exec_rejects_unknown_subcommand(monkeypatch):
    monkeypatch.setattr(shell_tools.subprocess, "run", lambda *a, **k: None)
    with pytest.raises(ValueError, match="不允许的子命令"):
        shell_exec("git checkout main")


def test_exec_rejects_empty_command(monkeypatch):
    monkeypatch.setattr(shell_tools.subprocess, "run", lambda *a, **k: None)
    with pytest.raises(ValueError, match="不能为空"):
        shell_exec("")


# ── 超时 / 输出上限 ────────────────────────────────────────────────


def test_exec_timeout_returns_error_dict(monkeypatch):
    def fake_run(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, timeout=kwargs["timeout"])

    monkeypatch.setattr(shell_tools.subprocess, "run", fake_run)
    monkeypatch.setattr(shell_tools, "_resolve_executable", lambda name: f"/usr/bin/{name}")

    result = shell_exec("git status")
    assert result["return_code"] == -1
    assert "超时" in result["stderr"]


def test_exec_truncates_overlong_output(monkeypatch):
    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout=b"x" * 20_000, stderr=b"")

    monkeypatch.setattr(shell_tools.subprocess, "run", fake_run)
    monkeypatch.setattr(shell_tools, "_resolve_executable", lambda name: f"/usr/bin/{name}")

    result = shell_exec("git log")
    assert len(result["stdout"]) <= shell_tools._MAX_OUTPUT_BYTES + 64
    assert "trimmed" in result["stdout"]


# ── 真 git 集成（无副作用文件验证） ────────────────────────────────


@requires_git
def test_exec_real_git_status_works():
    """真环境：只读 git status 正常返回，return_code 0。"""
    result = shell_exec("git status")
    assert result["return_code"] == 0


@requires_git
def test_exec_injection_creates_no_side_effect_file(tmp_path, monkeypatch):
    """`git status > x.txt` 经 shell=False 后 `>` 是 pathspec，不能写文件。"""
    monkeypatch.setattr(shell_tools, "_exec_cwd", lambda: tmp_path)
    target = tmp_path / "p0_side_effect.txt"

    result = shell_exec(f"git status > {target.name}")

    assert result["return_code"] != 0        # git 报 pathspec 错误
    assert not target.exists()               # 且没有副作用文件


@requires_git
def test_exec_real_pipe_is_not_a_shell_pipe(tmp_path, monkeypatch):
    """`git status | whoami`：whoami 永不执行（stdout 里没有当前用户名）。"""
    monkeypatch.setattr(shell_tools, "_exec_cwd", lambda: tmp_path)
    result = shell_exec("git status | whoami")

    assert "whoami" in result["stderr"] or result["return_code"] != 0
