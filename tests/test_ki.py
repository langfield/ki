"""Tests for ki command line interface (CLI)."""
import os
import pytest
from importlib.metadata import version
from click.testing import CliRunner
from cli_test_helpers import shell
import ki


# HELPER FUNCTIONS


def invoke(*args, **kwargs) -> int:
    """Wrap click CliRunner invoke()."""
    return CliRunner().invoke(*args, **kwargs)


# CLI


def test_bad_command_is_bad():
    """Typos should result in errors."""
    result = invoke(ki.ki, ["clome"])
    assert result.exit_code == 2
    assert "Error: No such command 'clome'." in result.output


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


def test_command_availability():
    """Are commands available?"""
    results = []
    results.append(shell("ki clone --help"))
    results.append(shell("ki pull --help"))
    results.append(shell("ki push --help"))
    for result in results:
        assert result.exit_code == 0


def test_cli():
    """Does CLI stop execution w/o a command argument?"""
    with pytest.raises(SystemExit):
        ki.ki()
        pytest.fail("CLI doesn't abort asking for a command argument")


# CLONE


def test_get_default_clone_directory():
    """Does it generate the right path?"""
    path = ki.get_default_clone_directory("collection.anki2")
    assert path == os.path.abspath("./collection")

    path = ki.get_default_clone_directory("sensors.anki2")
    assert path == os.path.abspath("./sensors")


def test_clone_creates_directory():
    """Does it create the directory?"""
    pass


def test_clone_errors_when_directory_already_exists():
    """Does it disallow overwrites?"""
    pass


def test_clone_generates_expected_notes():
    """Do generated note files match content of an example collection?"""
    pass
