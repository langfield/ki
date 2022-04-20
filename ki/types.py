#!/usr/bin/env python3
"""
Python package `ki` is a command-line interface for the version control and
editing of `.anki2` collections as git repositories of markdown files.
Rather than providing an interactive UI like the Anki desktop client, `ki` aims
to allow natural editing *in the filesystem*.

In general, the purpose of `ki` is to allow users to work on large, complex
Anki decks in exactly the same way they work on large, complex software
projects.
.. include:: ./DOCUMENTATION.md
"""

# pylint: disable=invalid-name, missing-class-docstring, broad-except
# pylint: disable=too-many-return-statements, too-many-lines

import os
import re
import json
import copy
import shutil
import logging
import secrets
import hashlib
import sqlite3
import tempfile
import functools
import subprocess
import unicodedata
import configparser
from enum import Enum
from pathlib import Path
from dataclasses import dataclass

import git
import click
import markdownify
import prettyprinter as pp
from tqdm import tqdm
from lark import Lark
from loguru import logger
from result import Result, Err, Ok, OkErr

import anki
from anki import notetypes_pb2
from anki.collection import Collection, Note, OpChangesWithId

from apy.convert import markdown_to_html, plain_to_html, html_to_markdown

from beartype import beartype
from beartype.typing import (
    Set,
    List,
    Dict,
    Any,
    Optional,
    Union,
    Tuple,
    Generator,
    Callable,
)

from ki.safe import safe
from ki.transformer import NoteTransformer, FlatNote

logging.basicConfig(level=logging.INFO)

ChangeNotetypeInfo = notetypes_pb2.ChangeNotetypeInfo
ChangeNotetypeRequest = notetypes_pb2.ChangeNotetypeRequest
NotetypeDict = Dict[str, Any]

# Type alias for OkErr types. Subscript indicates the Ok type.
Res = List

FS_ROOT = Path("/")

GIT = ".git"
GITIGNORE_FILE = ".gitignore"
GITMODULES_FILE = ".gitmodules"

KI = ".ki"
CONFIG_FILE = "config"
HASHES_FILE = "hashes"
BACKUPS_DIR = "backups"
LAST_PUSH_FILE = "last_push"
NO_SM_DIR = "no_submodules_tree"

MODELS_FILE = "models.json"

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
IGNORE = [GIT, KI, GITIGNORE_FILE, GITMODULES_FILE, MODELS_FILE]
LOCAL_SUFFIX = Path("ki/local")
STAGE_SUFFIX = Path("ki/stage")
REMOTE_SUFFIX = Path("ki/remote")
DELETED_SUFFIX = Path("ki/deleted")
FIELD_HTML_SUFFIX = Path("ki/fieldhtml")

REMOTE_CONFIG_SECTION = "remote"
COLLECTION_FILE_PATH_CONFIG_FIELD = "path"

GENERATED_HTML_SENTINEL = "data-original-markdown"

MD = ".md"

# Emoji regex character classes.
EMOJIS = "\U0001F600-\U0001F64F"
PICTOGRAPHS = "\U0001F300-\U0001F5FF"
TRANSPORTS = "\U0001F680-\U0001F6FF"
FLAGS = "\U0001F1E0-\U0001F1FF"

# Regex to filter out bad stuff from filenames.
SLUG_REGEX = re.compile(r"[^\w\s\-" + EMOJIS + PICTOGRAPHS + TRANSPORTS + FLAGS + "]")


BACKUPS_DIR_INFO = """
This is the '.ki/backups' directory, used to store backups of the '.anki2'
collection database file before ki overwrites it during a push. It may be
missing because the current ki repository has become corrupted.
"""

CONFIG_FILE_INFO = """
This is the '.ki/config' file, used to store the path to a '.anki2' collection
database file. It may be missing because the current ki repository has become
corrupted.
"""

HASHES_FILE_INFO = """
This is the '.ki/hashes' file, used to store recent md5sums of the '.anki2'
collection database file, which allow ki to determine when updates should be
rejected, i.e. when the user must pull remote changes before they can push
local ones. It may be missing because the current ki repository has become
corrupted.
"""

MODELS_FILE_INFO = """
This is the top-level 'models.json' file, which contains serialized notetypes
for all notes in the current repository. Ki should always create this during
cloning. If it has been manually deleted, try reverting to an earlier commit.
Otherwise, it may indicate that the repository has become corrupted.
"""

LAST_PUSH_FILE_INFO = """
This is the '.ki/last_push' file, used internally by ki to keep diffs and
eliminate unnecessary merge conflicts during pull operations. It should never
be missing, and if it is, the repository may have become corrupted.
"""

NO_MODULES_DIR_INFO = """
This is the '.ki/no_submodules_tree' file, used internally by ki to keep track
of what notes need to be written/modified/deleted in the Anki database. It
should never be missing, and if it is, the repository may have become
corrupted.
"""

COL_FILE_INFO = """
This is the '.anki2' database file that contains all the data for a user's
collection. This path was contained in the '.ki/config' file, indicating that
the collection this repository previously referred to has been moved or
deleted. The path can be manually fixed by editing the '.ki/config' file.
"""


# TYPES


class ExtantFile(type(Path())):
    """UNSAFE: Indicates that file *was* extant when it was resolved."""


class ExtantDir(type(Path())):
    """UNSAFE: Indicates that dir *was* extant when it was resolved."""


class EmptyDir(ExtantDir):
    """UNSAFE: Indicates that dir *was* empty (and extant) when it was resolved."""


class NoPath(type(Path())):
    """UNSAFE: Indicates that path *was not* extant when it was resolved."""


class Singleton(type(Path())):
    """UNSAFE: A path consisting of a single component (e.g. `file`, not `dir/file`)."""


class ExtantStrangePath(type(Path())):
    """
    UNSAFE: Indicates that path was extant but weird (e.g. a device or socket)
    when it was resolved.
    """


# ENUMS


class GitChangeType(Enum):
    """Enum for git file change types."""

    ADDED = "A"
    DELETED = "D"
    RENAMED = "R"
    MODIFIED = "M"
    TYPECHANGED = "T"


# DATACLASSES


@beartype
@dataclass(frozen=True)
class Delta:
    """The git delta for a single file."""

    status: GitChangeType
    path: ExtantFile
    relpath: Path


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
    models_file: ExtantFile
    last_push_file: ExtantFile
    no_modules_repo: git.Repo


@beartype
@dataclass(frozen=True)
class Field:
    """A typechecked version of `anki.models.FieldDict` for use within ki."""

    name: str
    ord: Optional[int]


@beartype
@dataclass(frozen=True)
class Template:
    """A typechecked version of `anki.models.TemplateDict` for use within ki."""

    name: str
    qfmt: str
    afmt: str
    ord: Optional[int]


@beartype
@dataclass(frozen=True)
class Notetype:
    """A typechecked version of `anki.models.NotetypeDict` for use within ki."""

    id: int
    name: str
    type: int
    flds: List[Field]
    tmpls: List[Template]
    sortf: Field

    # A copy of the `NotetypeDict` object as it was returned from the Anki
    # database. We keep this around to preserve extra keys that may not always
    # exist, but the ones above should be required for Anki to function.
    dict: Dict[str, Any]


@beartype
@dataclass(frozen=True)
class ColNote:
    """A note that exists in the Anki DB."""

    n: Note
    new: bool
    deck: str
    title: str
    old_nid: int
    markdown: bool
    notetype: Notetype
    sortf_text: str


@beartype
@dataclass(frozen=True)
class KiRepoRef:
    """
    UNSAFE: A repo-commit pair, where `sha` is guaranteed to be an extant
    commit hash of `repo`.
    """

    kirepo: KiRepo
    sha: str


@beartype
@dataclass(frozen=True)
class RepoRef:
    """
    UNSAFE: A repo-commit pair, where `sha` is guaranteed to be an extant
    commit hash of `repo`.
    """

    repo: git.Repo
    sha: str


@beartype
@dataclass(frozen=True)
class Leaves:
    root: ExtantDir
    files: Dict[str, ExtantFile]
    dirs: Dict[str, EmptyDir]


# EXCEPTIONS


class MissingFileError(FileNotFoundError):
    @beartype
    def __init__(self, path: Path, info: str = ""):
        msg = f"File not found: '{path}'{info}"
        super().__init__(msg)


class MissingDirectoryError(Exception):
    @beartype
    def __init__(self, path: Path, info: str = ""):
        msg = f"Directory not found: '{path}'{info}"
        super().__init__(msg)


class ExpectedFileButGotDirectoryError(FileNotFoundError):
    @beartype
    def __init__(self, path: Path, info: str = ""):
        msg = "A file was expected at this location, but got a directory: "
        msg += f"'{path}'{info}"
        super().__init__(msg)


class ExpectedDirectoryButGotFileError(Exception):
    @beartype
    def __init__(self, path: Path, info: str = ""):
        msg = "A directory was expected at this location, but got a file: "
        msg += f"'{path}'{info}"
        super().__init__(msg)


class ExpectedEmptyDirectoryButGotNonEmptyDirectoryError(Exception):
    @beartype
    def __init__(self, path: Path, info: str = ""):
        msg = "An empty directory was expected at this location, but it is nonempty: "
        msg += f"'{path}'{info}"
        super().__init__(msg)


class StrangeExtantPathError(Exception):
    @beartype
    def __init__(self, path: Path, info: str = ""):
        msg = "A normal file or directory was expected, but got a weird pseudofile "
        msg += "(e.g. a socket, or a device): "
        msg += f"'{path}'{info}"
        super().__init__(msg)


class NotKiRepoError(Exception):
    @beartype
    def __init__(self):
        msg = "fatal: not a ki repository (or any parent up to mount point /)\n"
        msg += "Stopping at filesystem boundary."
        super().__init__(msg)


class UpdatesRejectedError(Exception):
    @beartype
    def __init__(self, col_file: ExtantFile):
        msg = f"Failed to push some refs to '{col_file}'\n{HINT}"
        super().__init__(msg)


class TargetExistsError(Exception):
    @beartype
    def __init__(self, target: Path):
        msg = f"fatal: destination path '{target}' already exists and is "
        msg += "not an empty directory."
        super().__init__(msg)


class GitRefNotFoundError(Exception):
    @beartype
    def __init__(self, repo: git.Repo, sha: str):
        msg = f"Repo at '{repo.working_dir}' doesn't contain ref '{sha}'"
        super().__init__(msg)


class CollectionChecksumError(Exception):
    @beartype
    def __init__(self, col_file: ExtantFile):
        msg = f"Checksum mismatch on {col_file}. Was file changed?"
        super().__init__(msg)


class MissingNotetypeError(Exception):
    @beartype
    def __init__(self, model: str):
        msg = f"Notetype '{model}' doesn't exist. "
        msg += "Create it in Anki before adding notes via ki. "
        msg += "This may be caused by a corrupted '{MODELS_FILE}' file. "
        msg += "The models file must contain definitions for all models that appear "
        msg += "in all note files."
        super().__init__(msg)


class MissingFieldOrdinalError(Exception):
    @beartype
    def __init__(self, ord: int, nt: Dict[str, Any]):
        msg = f"Field with ordinal {ord} missing from notetype '{pp.pformat(nt)}'."
        super().__init__(msg)


class MissingNoteIdError(Exception):
    @beartype
    def __init__(self, nid: int):
        msg = f"Failed to locate note with nid '{nid}' in Anki database."
        super().__init__(msg)


class NotetypeMismatchError(Exception):
    @beartype
    def __init__(self, flatnote: FlatNote, new_notetype: Notetype):
        msg = f"Notetype '{flatnote.model}' "
        msg += f"specified in FlatNote with nid '{flatnote.nid}' "
        msg += f"does not match passed notetype '{new_notetype}'. "
        msg += f"This should NEVER happen, "
        msg += f"and indicates a bug in the caller to 'update_note()'."
        super().__init__(msg)


# WARNINGS


# TODO: Make this warning more descriptive. Should given the note id, the path,
# the field(s) which are missing, and the model.
class NoteFieldValidationWarning(Warning):
    pass


class UnhealthyNoteWarning(Warning):
    pass
