#!/usr/bin/env python3
"""
Push architecture redesign.
"""

# pylint: disable=invalid-name, unused-import, missing-class-docstring, broad-except
# pylint: disable=too-many-return-statements

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

FS_ROOT = Path("/")

KI = ".ki"
CONFIG_FILE = "config"
HASHES_FILE = "hashes"
BACKUPS_DIR = "backups"
LAST_PUSH_FILE = "last_push"

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
LOCAL_SUFFIX = Path("ki/local")

REMOTE_CONFIG_SECTION = "remote"
COLLECTION_FILE_PATH_CONFIG_FIELD = "path"


class ExtantFile(type(Path())):
    """UNSAFE: Indicates that file *was* extant when it was resolved."""


class ExtantDir(type(Path())):
    """UNSAFE: Indicates that dir *was* extant when it was resolved."""


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


@beartype
@dataclass(frozen=True)
class KiRepo:
    """
    UNSAFE: A ki repository, including:
    - .ki/hashes
    - .ki/config

    Existence of collection path is guaranteed.
    """

    repo: git.Repo
    root: ExtantDir
    ki_dir: ExtantDir
    col_file: ExtantFile
    backups_dir: ExtantDir
    config_file: ExtantFile
    hashes_file: ExtantFile
    last_push_file: ExtantFile


@beartype
@dataclass(frozen=True)
class MaybeKiRepo:
    """A KiRepo or an error message."""

    value: Union[KiRepo, str]
    type: Type = KiRepo


@beartype
def get_ki_repo(cwd: ExtantDir) -> MaybeKiRepo:
    """Get the containing ki repository of ``path``."""
    current = cwd
    while not (current / KI).is_dir() and current.resolve() != FS_ROOT:
        current = current.parent.resolve()
    if current.resolve() == FS_ROOT:
        msg = "fatal: not a ki repository (or any parent up to mount point /)\n"
        msg += "Stopping at filesystem boundary."
        return MaybeKiRepo(msg)
    try:
        repo = git.Repo(current)
    except Exception as err:
        return MaybeKiRepo(str(err))

    # Root directory and ki directory of repo now guaranteed to exist.
    root = ExtantDir(current)
    ki_dir = ExtantDir(root / KI)

    # Check that config file exists.
    config_file = ki_dir / CONFIG_FILE
    if not config_file.is_file():
        return MaybeKiRepo(f"File not found: {config_file}")
    config_file = ExtantFile(config_file)

    # Parse config file.
    config = configparser.ConfigParser()
    config.read(config_file)
    col_file = Path(config[REMOTE_CONFIG_SECTION][COLLECTION_FILE_PATH_CONFIG_FIELD])
    if not col_file.is_file():
        return MaybeKiRepo(f"Not found or not a file: {col_file}")
    col_file = ExtantFile(col_file)

    # TODO: Consider moving file creation out of this function, and handling
    # non-existence like we do for config above.

    # Get path to backups file (and possible create it).
    backups_dir = ki_dir / BACKUPS_DIR
    if not backups_dir.is_dir():
        if backups_dir.exists():
            return MaybeKiRepo(f"File at '{backups_dir}' should be directory")
        backups_dir.mkdir()
    backups_dir = ExtantDir(backups_dir)

    # Get path to hashes file (and possible create it).
    hashes_file = ki_dir / HASHES_FILE
    if not hashes_file.is_file():
        if hashes_file.exists():
            return MaybeKiRepo(f"Directory at '{hashes_file}' should be file")
        hashes_file.touch()
    hashes_file = ExtantFile(hashes_file)

    # Get path to last_push file (and possible create it).
    last_push_file = ki_dir / LAST_PUSH_FILE
    if not last_push_file.is_file():
        if last_push_file.exists():
            return MaybeKiRepo(f"Directory at '{last_push_file}' should be file")
        last_push_file.touch()
    last_push_file = ExtantFile(last_push_file)

    return MaybeKiRepo(
        KiRepo(
            repo,
            root,
            ki_dir,
            col_file,
            backups_dir,
            config_file,
            hashes_file,
            last_push_file,
        )
    )


@beartype
def lock(col_file: ExtantFile) -> sqlite3.Connection:
    """Acquire a lock on a SQLite3 database given a path."""
    con = sqlite3.connect(col_file)
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


@beartype
def updates_rejected_message(col_file: Path) -> str:
    """Generate need-to-pull message."""
    return f"Failed to push some refs to '{col_file}'\n{HINT}"


@beartype
@dataclass(frozen=True)
class KiRepoSHA:
    """
    UNSAFE: A repo-commit pair, where ``sha`` is guaranteed to be an extant
    commit hash of ``repo``.
    """

    kirepo: KiRepo
    sha: str


@beartype
def get_ephemeral_kirepo(suffix: Path, kirepo_sha: KiRepoSHA, md5sum: str) -> MaybeKiRepo:
    """
    Given a KiRepoSHA, i.e. a pair of the form (kirepo, SHA), we clone
    ``kirepo.repo`` into a temp directory and hard reset to the given commit
    hash.

    Parameters
    ----------
    suffix : pathlib.Path
        /tmp/.../ path suffix, e.g. ``ki/local/``.
    kirepo_sha : KiRepoSHA
        The ki repository to clone, and a commit for it.
    md5sum : str
        The md5sum of the associated anki collection.

    Returns
    -------
    KiRepo
        The cloned repository.
    """
    root = Path(tempfile.mkdtemp()) / suffix
    root.mkdir(parents=True)
    root = ExtantDir(root)

    # Git clone `repo` at latest commit in `/tmp/.../<suffix>/<md5sum>`.
    repo: git.Repo = kirepo_sha.kirepo.repo
    branch = repo.active_branch
    target: Path = root / md5sum
    ephem = git.Repo.clone_from(repo.working_dir, target, branch=branch, recursive=True)

    # Do a reset --hard to the given SHA.
    ephem.git.reset(kirepo_sha.sha, hard=True)

    # UNSAFE: We copy the .ki directory under the assumptions that:
    # - it's still extant
    # - the target directory does not exist
    # TODO: Consider writing typechecked wrappers around shutil methods.
    shutil.copytree(kirepo_sha.kirepo.ki_dir, target / KI)
    kirepo: MaybeKiRepo = get_ki_repo(target)

    return kirepo


Maybe = Union[
    MaybePath,
    MaybeRepo,
    MaybeExtantFile,
    MaybeExtantDir,
    MaybeKiRepo,
]


@beartype
def IO(maybe: Maybe) -> Optional[Any]:
    """UNSAFE: Handles errors and aborts."""
    if isinstance(maybe.value, maybe.type):
        return maybe.value
    assert isinstance(maybe.value, str)
    logger.error(maybe.value)
    raise ValueError(maybe.value)


@beartype
def _push() -> None:
    """Push a ki repository into a .anki2 file."""
    # Check that we are inside a ki repository, and get the associated collection.
    cwd: ExtantDir = ExtantDir(Path.cwd())
    kirepo: KiRepo = IO(get_ki_repo(cwd))
    con: sqlite3.Connection = lock(kirepo.col_file)

    md5sum: str = md5(kirepo.col_file)
    hashes: str = kirepo.hashes_file.read_text().split("\n")[-1]
    if md5sum not in hashes:
        IO(updates_rejected_message(kirepo.col_file))

    kirepo_sha = KiRepoSHA(kirepo, str(kirepo.repo.head.commit))
    staging_repo: KiRepo = IO(get_ephemeral_kirepo(LOCAL_SUFFIX, kirepo_sha, md5sum))
