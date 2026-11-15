"""Tests for scripts/check_setup.py (no data, key or network needed)."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_setup.py"


@pytest.fixture()
def mod():
    """Load the script as a module."""
    spec = importlib.util.spec_from_file_location("check_setup", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_python_ok(mod):
    assert mod.check_python()[0] == "OK"


def test_key_from_environment(mod, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    assert mod.check_key()[0] == "OK"


def test_key_missing(mod, monkeypatch, tmp_path):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(mod, "WEEK3", tmp_path)
    assert mod.check_key()[0] == "FAIL"


def test_key_from_env_file_is_not_printed(mod, monkeypatch, tmp_path):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    (tmp_path / ".env").write_text("GROQ_API_KEY=secret123\n", encoding="utf-8")
    monkeypatch.setattr(mod, "WEEK3", tmp_path)
    status, text = mod.check_key()
    assert status == "OK" and "secret123" not in text


def test_missing_database_fails(mod, tmp_path):
    s = mod.Settings(db_path=tmp_path / "nope", topics_path=tmp_path / "t.json", datasets_path=tmp_path / "d.json")
    rows = mod.check_data(s)
    assert rows[0][0] == "FAIL"
    assert all(r[0] == "WARN" for r in rows[1:])
