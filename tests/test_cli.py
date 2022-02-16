"""Tests for ki command line interface (CLI)."""
import os
import pytest
from importlib.metadata import version
from click.testing import CliRunner
from cli_test_helpers import shell
from ki import ki


def invoke(*args, **kwargs) -> int:
    """Wrap click CliRunner invoke()."""
    return CliRunner().invoke(*args, **kwargs)


def test_hello_world():
    result = invoke(ki, ["Peter"])
    assert result.exit_code == 0
    assert result.output == "Hello Peter!\n"


def test_runas_module():
    """Can this package be run as a Python module?"""
    result = shell("python -m ki --help")
    assert result.exit_code == 0


def test_entrypoint():
    """Is entrypoint script installed? (setup.py)"""
    result = shell("ki --help")
    assert result.exit_code == 0


def test_version():
    """Does --version display information as expected?"""
    expected_version = version("ki")
    result = shell("ki --version")

    assert result.stdout == f"ki, version {expected_version}{os.linesep}"
    assert result.exit_code == 0


def test_clone_command():
    """Is command available?"""
    result = shell("ki baz --help")
    assert result.exit_code == 0
