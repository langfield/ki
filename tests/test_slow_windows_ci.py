#!/usr/bin/env python3
"""Tests for ki command line interface (CLI)."""
import os
from importlib.metadata import version

from beartype import beartype
from beartype.typing import List

import ki
from tests.test_ki import invoke


# CLI

import subprocess


@beartype
def capture(args: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        text=True,
        capture_output=True,
    )


def test_version():
    """Does --version display information as expected?"""
    expected_version = version("ki")
    result = invoke(ki.ki, ["--version"])

    assert result.stdout.rstrip() == f"ki, version {expected_version}"
    assert result.exit_code == 0


def test_version_clickless():
    """Does --version display information as expected?"""
    expected_version = version("ki")
    p = capture(["ki", "--version"])

    assert p.stdout.rstrip() == f"ki, version {expected_version}"
    assert p.returncode == 0
