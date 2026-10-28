from __future__ import annotations

from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from gpurelay import __version__
from gpurelay.cli import app

runner = CliRunner()

# Every command from spec §15 (all implemented).
IMPLEMENTED = {
    "init",
    "run-local", "analyze", "submit", "up", "status", "logs", "quota", "pause", "resume",
    "cancel", "fetch", "colab", "dashboard", "export-state", "import-state", "serve",
    "app", "gateway", "helper",
}  # fmt: skip

# Minimal valid arguments for invoking each command.


def test_registered_commands_match_spec() -> None:
    group = typer.main.get_command(app)
    assert isinstance(group, typer.core.TyperGroup)
    assert set(group.commands) == IMPLEMENTED


def test_root_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    for name in IMPLEMENTED:
        assert name in result.output


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


@pytest.mark.parametrize("command", sorted(IMPLEMENTED))
def test_command_help(command: str) -> None:
    result = runner.invoke(app, [command, "--help"])
    assert result.exit_code == 0, result.output
    assert "Usage" in result.output


def test_init_local_needs_no_accounts(tmp_path: Path) -> None:
    env = {"GPURELAY_HOME": str(tmp_path), "HF_TOKEN": ""}
    before = runner.invoke(app, ["status"], env=env)
    assert before.exit_code == 1 and "gpurelay init --local" in before.output
    result = runner.invoke(app, ["init", "--local"], env=env)
    assert result.exit_code == 0, result.output
    assert 'kind = "localfs"' in (tmp_path / "config.toml").read_text()
    after = runner.invoke(app, ["status"], env=env)
    assert after.exit_code == 0, after.output
    assert "local" in after.output


def test_state_moves_between_homes(tmp_path: Path) -> None:
    laptop = {"GPURELAY_HOME": str(tmp_path / "laptop"), "HF_TOKEN": ""}
    vm = {"GPURELAY_HOME": str(tmp_path / "vm"), "HF_TOKEN": ""}
    assert runner.invoke(app, ["init", "--local"], env=laptop).exit_code == 0
    script = tmp_path / "job.py"
    script.write_text("print('hi')\n")
    submitted = runner.invoke(app, ["submit", str(script), "--name", "moved"], env=laptop)
    assert submitted.exit_code == 0, submitted.output
    job_id = submitted.output.split("submitted ")[1].split()[0]
    archive = tmp_path / "state.tgz"
    result = runner.invoke(app, ["export-state", str(archive)], env=laptop)
    assert result.exit_code == 0, result.output
    assert runner.invoke(app, ["import-state", str(archive)], env=vm).exit_code == 0
    again = runner.invoke(app, ["import-state", str(archive)], env=vm)
    assert again.exit_code == 1 and "--force" in again.output
    assert runner.invoke(app, ["import-state", str(archive), "--force"], env=vm).exit_code == 0
    status = runner.invoke(app, ["status"], env=vm)
    assert status.exit_code == 0 and job_id in status.output


def test_bad_inputs_get_clean_errors(tmp_path: Path) -> None:
    env = {"GPURELAY_HOME": str(tmp_path / "home"), "HF_TOKEN": ""}
    assert runner.invoke(app, ["init", "--local"], env=env).exit_code == 0
    notes = tmp_path / "notes.txt"
    notes.write_text("hello\n")
    broken = tmp_path / "broken.py"
    broken.write_text("def f(:\n")
    garbage = tmp_path / "state.tgz"
    garbage.write_text("not an archive")
    cases = [
        (["submit", str(notes)], "submit a Python script"),
        (["submit", str(broken)], "syntax error"),
        (["import-state", str(garbage), "--force"], "not a GPU Relay state archive"),
        (["import-state", str(tmp_path / "missing.tgz")], "does not exist"),
        (["status", "job-nope"], "no job matches"),
    ]
    for args, message in cases:
        result = runner.invoke(app, args, env=env)
        assert result.exit_code != 0, args
        assert message in " ".join(result.output.split()), (args, result.output)
        assert not isinstance(result.exception, AssertionError | TypeError | AttributeError)
