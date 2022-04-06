#!/usr/bin/env python3
"""
Push architecture redesign.
"""

# pylint: disable=invalid-name, unused-import, missing-class-docstring

import os
import re
import sys
import json
import glob
import pprint
import shutil
import logging
import tarfile
import hashlib
import sqlite3
import tempfile
import warnings
import itertools
import functools
import subprocess
import collections
import unicodedata
import configparser
from enum import Enum
from pathlib import Path
from dataclasses import dataclass

import git
import click
import gitdb
import prettyprinter as pp
from tqdm import tqdm
from loguru import logger
from pyinstrument import Profiler

from bs4 import MarkupResemblesLocatorWarning

from lark import Lark, Transformer
from lark.lexer import Token

import anki

from apy.anki import Anki
from apy.convert import markdown_to_html, plain_to_html

from beartype import beartype
from beartype.typing import (
    Set,
    List,
    Dict,
    Any,
    Iterator,
    Sequence,
    Iterable,
    Optional,
    Union,
    Tuple,
    Generator,
    Type,
)

from ki.note import KiNote
from ki.transformer import NoteTransformer, FlatNote

logging.basicConfig(level=logging.INFO)

FieldDict = Dict[str, Any]
NotetypeDict = Dict[str, Any]
TemplateDict = Dict[str, Union[str, int, None]]

BATCH_SIZE = 500
HTML_REGEX = r"</?\s*[a-z-][^>]*\s*>|(\&(?:[\w\d]+|#\d+|#x[a-f\d]+);)"
REMOTE_NAME = "anki"
BRANCH_NAME = "main"
CHANGE_TYPES = "A D R M T".split()
TQDM_NUM_COLS = 70
MAX_FIELNAME_LEN = 30
HINT = (
    "hint: Updates were rejected because the tip of your current branch is behind\n"
    + "hint: the Anki remote collection. Integrate the remote changes (e.g.\n"
    + "hint: 'ki pull ...') before pushing again."
)
IGNORE = [".git", ".ki", ".gitignore", ".gitmodules", "models.json"]
MODELS_FILENAME = "models.json"


class ExtantFile(type(Path())):
    """Indicates that file *was* extant when it was resolved."""


class ExtantDir(type(Path())):
    """Indicates that dir *was* extant when it was resolved."""


class Hashes(str):
    """Hashes content."""


@beartype
@dataclass(frozen=True)
class MaybePath:
    """A pathlib.Path or an error message."""
    value: Union[Path, str]
    type: Type = Path


@beartype
@dataclass(frozen=True)
class MaybeRepo:
    """A git.Repo or an error message."""
    value: Union[git.Repo, str]
    type: Type = git.Repo


@beartype
@dataclass(frozen=True)
class MaybeHashes:
    """A Hashes string or an error message."""
    value: Union[Hashes, str]
    type: Type = Hashes


@beartype
@dataclass
class MaybeExtantFile:
    value: Union[Path, ExtantFile, str]
    type: Type = ExtantFile

    def __post_init__(self):
        """Validate input."""
        # Beartype ensures this is Path or str. If str, then it's an error message.
        if not isinstance(self.value, Path):
            return

        # Resolve path.
        path = self.value.resolve()

        # Check that path exists and is a file.
        if not self.value.exists():
            self.value = f"File or directory not found: {path}"
        elif self.value.is_dir():
            self.value = f"Expected file, got directory: {path}"
        elif not self.value.is_file():
            self.value = f"Extant but not a file: {path}"

        # Must be an extant file.
        else:
            self.value = ExtantFile(path)


@beartype
@dataclass
class MaybeExtantDir:
    value: Union[Path, ExtantDir, str]
    type: Type = ExtantDir

    def __post_init__(self):
        """Validate input."""
        # Beartype ensures this is Path or str. If str, then it's an error message.
        if not isinstance(self.value, Path):
            return

        # Resolve path.
        path = self.value.resolve()

        # Check that path exists and is a directory.
        if not self.value.exists():
            self.value = f"File or directory not found: {path}"
        elif self.value.is_file():
            self.value = f"Expected directory, got file: {path}"
        elif not self.value.is_dir():
            self.value = f"Extant but not a directory: {path}"

        # Must be an extant file.
        else:
            self.value = ExtantDir(path)


Maybe = Union[MaybePath, MaybeRepo, MaybeExtantFile, MaybeExtantDir, MaybeHashes]


@beartype
def get_colpath(root: ExtantDir) -> MaybeExtantFile:
    """Get collection path from `.ki/` directory."""
    # Check that config file exists.
    config_path = root / ".ki/" / "config"
    if not config_path.is_file():
        return MaybeExtantFile(f"File not found: {config_path}")

    # Parse config file.
    config = configparser.ConfigParser()
    config.read(config_path)
    colpath = Path(config["remote"]["path"])

    if not colpath.is_file():
        return MaybeExtantFile(f"Not found or not a file: {colpath}")

    return MaybeExtantFile(colpath)


@beartype
def get_ki_repo(cwd: ExtantDir) -> MaybeRepo:
    """Get the containing ki repository of ``path``."""
    current = cwd
    while not (current / ".ki").is_dir() and current.resolve() != Path("/"):
        current = current.parent.resolve()
    if current.resolve() == Path("/"):
        msg = "fatal: not a ki repository (or any parent up to mount point /)\n"
        msg += "Stopping at filesystem boundary."
        return MaybeRepo(msg)
    try:
        repo = git.Repo(current)
    # pylint: disable=broad-except
    except Exception as err:
        return MaybeRepo(str(err))
    return MaybeRepo(repo)


@beartype
def lock(colpath: ExtantFile) -> sqlite3.Connection:
    """Acquire a lock on a SQLite3 database given a path."""
    con = sqlite3.connect(colpath)
    con.isolation_level = "EXCLUSIVE"
    con.execute("BEGIN EXCLUSIVE")
    return con


@beartype
def md5(path: ExtantFile) -> str:
    """Compute md5sum of file at `path`."""
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


KI = ".ki"
HASHES_FILE = "hashes"


@beartype
def get_ki_hashes(root: ExtantDir) -> MaybeHashes:
    """Get hashes content from ki directory file."""
    hashes_path = root / KI / HASHES_FILE
    if not hashes_path.is_file():
        return MaybeHashes(f"File not found: {hashes_path}")

    # Like ``Just`` in haskell.
    hashes = Hashes(hashes_path.read_text().split("\n")[-1])
    return MaybeHashes(hashes)


@beartype
def updates_rejected_message(colpath: Path) -> str:
    """Generate need-to-pull message."""
    return f"Failed to push some refs to '{colpath}'\n{HINT}"


@beartype
def IO(maybe: Maybe) -> Optional[Any]:
    """UNSAFE: Handles errors and aborts."""
    logger.debug(f"Type of maybe.value '{maybe.value}': {type(maybe.value)}")
    if isinstance(maybe.value, maybe.type):
        return maybe.value
    assert isinstance(maybe.value, str)
    logger.error(maybe.value)
    raise ValueError(maybe.value)


@beartype
def _push() -> None:
    """Push a ki repository into a .anki2 file."""
    cwd: ExtantDir = IO(MaybeExtantDir(Path.cwd()))
    repo: MaybeRepo = get_ki_repo(cwd)
    repo: git.Repo = IO(repo)
    root: ExtantDir = IO(MaybeExtantDir(Path(repo.working_dir)))

    colpath: MaybeExtantFile = get_colpath(root)
    colpath: ExtantFile = IO(colpath)
    con: sqlite3.Connection = lock(colpath)
    md5sum: str = md5(colpath)

    hashes: Hashes = IO(get_ki_hashes(root))
    if md5sum not in hashes:
        IO(updates_rejected_message(colpath))
