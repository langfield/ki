#!/usr/bin/env python3
"""
Python package `ki` is a command-line interface for the version control and
editing of ``.anki2`` collections as git repositories of markdown files.
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
import glob
import json
import shutil
import logging
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

import anki
import click
import prettyprinter as pp
from tqdm import tqdm
from lark import Lark
from loguru import logger

import git
from git.exc import GitCommandError

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
    Callable,
)

from ki.note import KiNote
from ki.transformer import NoteTransformer, FlatNote

logging.basicConfig(level=logging.INFO)

FieldDict = Dict[str, Any]
NotetypeDict = Dict[str, Any]
TemplateDict = Dict[str, Union[str, int, None]]

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


@beartype
@dataclass(frozen=True)
class MaybeRepo:
    """A git.Repo or an error message."""

    value: Union[git.Repo, str]
    type: Type = git.Repo


@beartype
@dataclass
class MaybeNoPath:
    value: Union[Path, NoPath, str]
    type: Type = NoPath

    def __post_init__(self):
        """Validate input."""
        if isinstance(self.value, NoPath):
            msg = f"DANGER: '{NoPath}' values"
            msg += " should not be instantiated directly!"
            self.value = msg
            return
        if not isinstance(self.value, Path):
            return

        # Resolve path.
        path = self.value.resolve()
        if self.value.exists():
            self.value = f"File exists (unexpected): {path}"
        else:
            self.value = NoPath(path)


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

        # Must be an extant directory.
        else:
            self.value = ExtantDir(path)


@beartype
def is_empty(directory: ExtantDir) -> bool:
    """Check if directory is empty, quickly."""
    return not next(os.scandir(directory), None)


@beartype
@dataclass
class MaybeEmptyDir:
    value: Union[Path, EmptyDir, str]
    type: Type = EmptyDir

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
            directory = ExtantDir(path)
            if not is_empty(directory):
                self.value = f"Directory, but not empty: {directory}"
            else:
                self.value = EmptyDir(directory)


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
class MaybeKiRepo:
    """A KiRepo or an error message."""

    value: Union[KiRepo, str]
    type: Type = KiRepo


# DANGER


@beartype
def rmtree(target: ExtantDir) -> NoPath:
    """Call shutil.rmtree()."""
    shutil.rmtree(target)
    return JUST(MaybeNoPath(target))


@beartype
def copytree(source: ExtantDir, target: NoPath) -> ExtantDir:
    """Call shutil.copytree()."""
    shutil.copytree(source, target)
    return JUST(MaybeExtantDir(target))


@beartype
def fcwd() -> ExtantDir:
    """Call Path.cwd()."""
    return JUST(MaybeExtantDir(Path.cwd()))


@beartype
def ftest(
    path: Path,
) -> Union[ExtantFile, ExtantDir, EmptyDir, ExtantStrangePath, NoPath]:
    """
    Test whether ``path`` is a file, a directory, or something else. If
    something else, we consider that, for the sake of simplicity, a
    NoPath.
    """
    if path.is_file():
        return JUST(MaybeExtantFile(path))
    if path.is_dir():
        directory = JUST(MaybeExtantDir(path))
        if is_empty(directory):
            return JUST(MaybeEmptyDir(path))
        return directory
    if path.exists():
        return ExtantStrangePath(path)
    return JUST(MaybeNoPath(path))


@beartype
def ftouch(directory: ExtantDir, name: Singleton) -> ExtantFile:
    """Touch a file."""
    path = directory / name
    path.touch()
    return JUST(MaybeExtantFile(path))


@beartype
def fmkdir(path: NoPath) -> EmptyDir:
    """Make a directory (with parents)."""
    path.mkdir(parents=True)
    return JUST(MaybeEmptyDir(path))


@beartype
def fmksubdir(directory: EmptyDir, suffix: Path) -> EmptyDir:
    """
    Make a subdirectory of an empty directory (with parents).

    Returns
    -------
    EmptyDir
        The created subdirectory.
    """
    subdir = directory / suffix
    subdir.mkdir(parents=True)
    directory.__class__ = ExtantDir
    return JUST(MaybeEmptyDir(subdir))


@beartype
def fmkleaves(directory: EmptyDir, names: List[Singleton]) -> List[ExtantFile]:
    """Touch several files in an empty directory."""
    files = []
    for name in names:
        file = ftouch(directory, name)
        files.append(file)
    return files


@beartype
def fforce_mkdir(path: Path) -> ExtantDir:
    """Make a directory (with parents, ok if it already exists)."""
    path.mkdir(parents=True, exist_ok=True)
    return JUST(MaybeExtantDir(path))


@beartype
def fchdir(directory: ExtantDir) -> ExtantDir:
    """Changes working directory and returns old cwd."""
    old: ExtantDir = fcwd()
    os.chdir(directory)
    return old


@beartype
def fparent(path: Union[ExtantFile, ExtantDir]) -> Union[ExtantFile, ExtantDir]:
    """Get the parent of a path that exists."""
    if path.resolve() == FS_ROOT:
        return JUST(MaybeExtantDir(FS_ROOT))
    if isinstance(path, ExtantFile):
        return JUST(MaybeExtantDir(path.parent))
    return JUST(MaybeExtantDir(path.parent))


@beartype
def fmkdtemp() -> EmptyDir:
    """Make a temporary directory (in /tmp)."""
    return JUST(MaybeEmptyDir(Path(tempfile.mkdtemp())))


# pylint: disable=unused-argument
@beartype
def fcopyfile(source: ExtantFile, target: Path, target_root: ExtantDir) -> ExtantFile:
    """
    Force copy a file (potentially overwrites the target path). May fail due to
    permissions errors.
    """
    shutil.copyfile(source, target)
    return JUST(MaybeExtantFile(target))


@beartype
def working_dir(repo: git.Repo) -> ExtantDir:
    """Get working directory of a repo."""
    return JUST(MaybeExtantDir(Path(repo.working_dir)))


@beartype
def git_dir(repo: git.Repo) -> ExtantDir:
    """Get git directory of a repo."""
    return JUST(MaybeExtantDir(Path(repo.git_dir)))


# SAFE


@beartype
def singleton(name: str) -> Singleton:
    """Removes all forward slashes and returns a Singleton pathlib.Path."""
    return Singleton(name.replace("/", ""))


@beartype
def get_repo(root: ExtantDir) -> MaybeRepo:
    """Read a git repo safely."""
    try:
        return MaybeRepo(git.Repo(root))
    except Exception as err:
        return MaybeRepo(f"{err.__class__} : {err}\n{err.__doc__}")


@beartype
def get_ki_repo(cwd: ExtantDir) -> MaybeKiRepo:
    """Get the containing ki repository of ``path``."""
    current = cwd
    while current != FS_ROOT:
        ki_dir = ftest(current / KI)
        if isinstance(ki_dir, ExtantDir):
            break
        current = fparent(current)

    if current == FS_ROOT:
        msg = "fatal: not a ki repository (or any parent up to mount point /)\n"
        msg += "Stopping at filesystem boundary."
        return MaybeKiRepo(msg)

    maybe_repo: MaybeRepo = get_repo(current)
    if isinstance(maybe_repo.value, git.Repo):
        repo = maybe_repo.value
    else:
        return MaybeKiRepo(maybe_repo.value)

    # Root directory and ki directory of repo now guaranteed to exist.
    root = current

    # Check that config file exists.
    config_file = ftest(ki_dir / CONFIG_FILE)
    if not isinstance(config_file, ExtantFile):
        return MaybeKiRepo(f"Not found or not a file: {config_file}")

    # Parse config file.
    config = configparser.ConfigParser()
    config.read(config_file)
    col_file = Path(config[REMOTE_CONFIG_SECTION][COLLECTION_FILE_PATH_CONFIG_FIELD])

    # Check that collection file exists.
    col_file = ftest(col_file)
    if not isinstance(col_file, ExtantFile):
        return MaybeKiRepo(f"Not found or not a file: {col_file}")

    # TODO: Consider moving file creation out of this function, and handling
    # non-existence like we do for config above.

    # Get path to backups directory (and possible create it).
    backups_dir = ftest(ki_dir / BACKUPS_DIR)
    if isinstance(backups_dir, NoPath):
        backups_dir = fmkdir(backups_dir)
    elif not isinstance(backups_dir, ExtantDir):
        return MaybeKiRepo(f"Not found or not a directory: {backups_dir}")

    # Get path to hashes file (and possible create it).
    hashes_file = ftest(ki_dir / HASHES_FILE)
    if isinstance(hashes_file, NoPath):
        hashes_file = ftouch(ki_dir, singleton(HASHES_FILE))
    elif not isinstance(hashes_file, ExtantFile):
        return MaybeKiRepo(f"Not found or not a file: {hashes_file}")

    # Get path to hashes file (and possible create it).
    last_push_file = ftest(ki_dir / LAST_PUSH_FILE)
    if isinstance(last_push_file, NoPath):
        last_push_file = ftouch(ki_dir, singleton(LAST_PUSH_FILE))
    elif not isinstance(last_push_file, ExtantFile):
        return MaybeKiRepo(f"Not found or not a file: {last_push_file}")

    # Get path to no_submodules_tree directory (and possible create it).
    no_modules_dir = ftest(ki_dir / NO_SM_DIR)
    if isinstance(no_modules_dir, NoPath):
        no_modules_dir = fmkdir(no_modules_dir)
    elif not isinstance(no_modules_dir, ExtantDir):
        return MaybeKiRepo(f"Not found or not a directory: {no_modules_dir}")

    # Get path to models file.
    models_file = ftest(root / MODELS_FILE)
    if not isinstance(models_file, ExtantFile):
        return MaybeKiRepo(f"Not found or not a file: {models_file}")

    # Try to get no_submodules_tree repo.
    maybe_no_modules_repo: MaybeRepo = get_repo(no_modules_dir)
    if isinstance(maybe_no_modules_repo.value, git.Repo):
        no_modules_repo = maybe_no_modules_repo.value
    else:
        # Initialize no_submodules_tree git repository.
        # TODO: This is UNSAFE, creation may fail for strange reasons, this
        # should exist only in ``_clone()``.
        no_modules_repo = git.Repo.init(no_modules_dir, initial_branch=BRANCH_NAME)

    return MaybeKiRepo(
        KiRepo(
            repo,
            root,
            ki_dir,
            col_file,
            backups_dir,
            config_file,
            hashes_file,
            models_file,
            last_push_file,
            no_modules_repo,
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
def unlock(con: sqlite3.Connection) -> None:
    """Unlock a SQLite3 database."""
    con.commit()
    con.close()


@beartype
def md5(path: ExtantFile) -> str:
    """Compute md5sum of file at `path`."""
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


@beartype
@dataclass
class MaybeError:
    value: str
    type: Type = type(None)


@beartype
def updates_rejected_message(col_file: Path) -> MaybeError:
    """Generate need-to-pull message."""
    return MaybeError(f"Failed to push some refs to '{col_file}'\n{HINT}")


@beartype
@dataclass(frozen=True)
class KiRepoRef:
    """
    UNSAFE: A repo-commit pair, where ``sha`` is guaranteed to be an extant
    commit hash of ``repo``.
    """

    kirepo: KiRepo
    sha: str


@beartype
@dataclass
class MaybeKiRepoRef:
    value: Union[KiRepoRef, Tuple[KiRepo, str], str]
    type: Type = KiRepoRef

    def __post_init__(self):
        """Validate input."""
        # Beartype ensures this is Path or str. If str, then it's an error message.
        if isinstance(self.value, KiRepoRef):
            msg = f"DANGER: '{KiRepoRef}' values should not be instantiated directly!"
            self.value = msg
            return
        is_2_tuple = isinstance(self.value, tuple) and len(self.value) == 2
        has_repo = isinstance(self.value[0], KiRepo)
        has_ref = isinstance(self.value[1], str)
        if not (is_2_tuple and has_repo and has_ref):
            return

        kirepo, sha = self.value
        repo = kirepo.repo

        if not ref_exists(repo, sha):
            self.value = f"Repo at '{repo.working_dir}' doesn't contain ref '{sha}'"
        else:
            self.value = KiRepoRef(kirepo, sha)


@beartype
def ref_exists(repo: git.Repo, ref: str) -> bool:
    """Check if git commit reference exists in repository."""
    try:
        repo.git.rev_parse("--verify", ref)
    except GitCommandError:
        return False
    return True


@beartype
@dataclass(frozen=True)
class RepoRef:
    """
    UNSAFE: A repo-commit pair, where ``sha`` is guaranteed to be an extant
    commit hash of ``repo``.
    """

    repo: git.Repo
    sha: str


@beartype
@dataclass
class MaybeRepoRef:
    value: Union[RepoRef, Tuple[git.Repo, str], str]
    type: Type = RepoRef

    def __post_init__(self):
        """Validate input."""
        if isinstance(self.value, RepoRef):
            msg = f"DANGER: '{RepoRef}' values should not be instantiated directly!"
            self.value = msg
            return
        is_2_tuple = isinstance(self.value, tuple) and len(self.value) == 2
        has_repo = isinstance(self.value[0], git.Repo)
        has_ref = isinstance(self.value[1], str)
        if not (is_2_tuple and has_repo and has_ref):
            return

        repo, sha = self.value
        if not ref_exists(repo, sha):
            self.value = f"Repo at '{repo.working_dir}' doesn't contain ref '{sha}'"
        else:
            self.value = RepoRef(repo, sha)


@beartype
@dataclass
class MaybeHeadRepoRef:
    value: Union[RepoRef, git.Repo, str]
    type: Type = RepoRef

    def __post_init__(self):
        """Validate input."""
        # Beartype ensures this is git.Repo or str. If not a Repo, it muust be
        # an error. We should never be instantiating a RepoRef outside of a Maybe
        # __post_init__().
        if isinstance(self.value, RepoRef):
            self.value = (
                f"DANGER: '{RepoRef}' values should not be instantiated directly!"
            )
            return
        if not isinstance(self.value, git.Repo):
            return
        repo = self.value

        # GitPython raises a ValueError when references don't exist.
        try:
            self.value = RepoRef(repo, repo.head.commit.hexsha)
        except ValueError as err:
            self.value = str(err)


@beartype
@dataclass
class MaybeHeadKiRepoRef:
    value: Union[KiRepoRef, KiRepo, str]
    type: Type = KiRepoRef

    def __post_init__(self):
        """Validate input."""
        # Beartype ensures this is git.Repo or str. If not a KiRepo, it muust be
        # an error. We should never be instantiating a KiRepoRef outside of a Maybe
        # __post_init__().
        if isinstance(self.value, self.type):
            self.value = (
                f"DANGER: '{RepoRef}' values should not be instantiated directly!"
            )
            return
        if not isinstance(self.value, KiRepo):
            return
        kirepo = self.value
        repo = kirepo.repo

        # GitPython raises a ValueError when references don't exist.
        try:
            self.value = KiRepoRef(kirepo, repo.head.commit.hexsha)
        except ValueError as err:
            self.value = str(err)


@beartype
def get_ephemeral_repo(suffix: Path, repo_ref: RepoRef, md5sum: str) -> git.Repo:
    """Get a temporary copy of a git repository in /tmp/<suffix>/."""
    tempdir: EmptyDir = fmkdtemp()
    root: EmptyDir = fmksubdir(tempdir, suffix)

    # Git clone `repo` at latest commit in `/tmp/.../<suffix>/<md5sum>`.
    repo: git.Repo = repo_ref.repo
    branch = repo.active_branch
    target: Path = root / md5sum

    # UNSAFE: But only called here, and it should be guaranteed to work because
    # ``repo`` is actually a git repository, presumably there is always an
    # active branch, and ``target`` does not exist.
    ephem = git.Repo.clone_from(repo.working_dir, target, branch=branch, recursive=True)

    # Do a reset --hard to the given SHA.
    ephem.git.reset(repo_ref.sha, hard=True)

    return ephem


@beartype
def get_ephemeral_kirepo(suffix: Path, kirepo_ref: KiRepoRef, md5sum: str) -> KiRepo:
    """
    Given a KiRepoRef, i.e. a pair of the form (kirepo, SHA), we clone
    ``kirepo.repo`` into a temp directory and hard reset to the given commit
    hash.

    Parameters
    ----------
    suffix : pathlib.Path
        /tmp/.../ path suffix, e.g. ``ki/local/``.
    kirepo_ref : KiRepoRef
        The ki repository to clone, and a commit for it.
    md5sum : str
        The md5sum of the associated anki collection.

    Returns
    -------
    KiRepo
        The cloned repository.
    """
    ref: RepoRef = fkirepo_ref_to_repo_ref(kirepo_ref)
    ephem: git.Repo = get_ephemeral_repo(suffix, ref, md5sum)

    # UNSAFE: We copy the .ki directory under the assumption that the ki
    # directory doesn't exist (it shouldn't).
    ephem_ki_dir = JUST(MaybeNoPath(working_dir(ephem) / KI))
    copytree(kirepo_ref.kirepo.ki_dir, ephem_ki_dir)
    kirepo: KiRepo = JUST(get_ki_repo(working_dir(ephem)))

    return kirepo


@beartype
def fkirepo_ref_to_repo_ref(kirepo_ref: KiRepoRef) -> RepoRef:
    """UNSAFE: fold a KiRepo get reference into a repo reference."""
    return JUST(MaybeRepoRef((kirepo_ref.kirepo.repo, kirepo_ref.sha)))


@beartype
def is_anki_note(path: ExtantFile) -> bool:
    """Check if file is an `apy`-style markdown anki note."""
    path = str(path)

    # Ought to have markdown file extension.
    if path[-3:] != ".md":
        return False
    with open(path, "r", encoding="UTF-8") as md_f:
        lines = md_f.readlines()
    if len(lines) < 2:
        return False
    if lines[0] != "## Note\n":
        return False
    if not re.match(r"^nid: [0-9]+$", lines[1]):
        return False
    return True


@beartype
def path_ignore_fn(path: Path, patterns: List[str], root: ExtantDir) -> bool:
    """Lambda to be used as first argument to filter(). Filters out paths-to-ignore."""
    for p in patterns:
        if p == path.name:
            logger.warning(f"Ignoring {path} matching pattern {p}")
            return False

    # Ignore files that match a pattern in ``patterns`` ('*' not supported).
    for ignore_path in [root / p for p in patterns]:
        parents = [path.resolve()] + [p.resolve() for p in path.parents]
        if ignore_path.resolve() in parents:
            logger.warning(f"Ignoring {path} matching pattern {ignore_path}")
            return False

    # If ``path`` is an extant file (not a directory) and NOT a note, ignore it.
    if path.exists() and path.resolve().is_file():
        file = ExtantFile(path)
        if not is_anki_note(file):
            logger.warning(f"Not Anki note {file}")
            return False

    return True


class GitChangeType(Enum):
    """Enum for git file change types."""

    ADDED = "A"
    DELETED = "D"
    RENAMED = "R"
    MODIFIED = "M"
    TYPECHANGED = "T"


@beartype
@dataclass(frozen=True)
class Delta:
    """The git delta for a single file."""

    status: GitChangeType
    path: ExtantFile
    relpath: Path


@beartype
def unsubmodule_repo(repo: git.Repo) -> None:
    """
    Un-submodule all the git submodules (convert to ordinary subdirectories and
    destroy commit history).

    MUTATES REPO in-place!

    UNSAFE: git.rm() calls.
    """
    gitmodules_path: Path = Path(repo.working_dir) / GITMODULES_FILE
    for sm in repo.submodules:

        # Untrack, remove gitmodules file, remove .git file, and add directory back.
        sm.update()

        # Guaranteed to exist by gitpython.
        sm_path: ExtantDir = JUST(MaybeExtantDir(Path(sm.module().working_tree_dir)))
        repo.git.rm(sm_path, cached=True)

        # May not exist.
        repo.git.rm(gitmodules_path)

        # Guaranteed to exist by gitpython.
        ExtantDir(sm_path / GIT).unlink()

        # Should still exist after git.rm().
        repo.git.add(sm_path)
        _ = repo.index.commit(f"Add submodule {sm.name} as ordinary directory.")

    if gitmodules_path.exists():
        repo.git.rm(gitmodules_path)
        _ = repo.index.commit("Remove '.gitmodules' file.")


@beartype
def get_deltas_since_last_push(
    ref: RepoRef,
    md5sum: str,
    ignore_fn: Callable[[Path], bool],
    parser: Lark,
    transformer: NoteTransformer,
) -> List[Delta]:
    """
    Get the list of all deltas (changed files) since the last time that ``ki
    push`` was run successfully.

    In order to treat submodules, we keep a separate version of the entire
    repository in ``.ki/no_submodules_tree``. This is itself a git submodule.
    It is NOT a kirepo, because it does not have a .ki directory of its own.

    We need only commit to it during push calls, so there is no need to make
    any adjustments to ``_clone()``.

    Parameters
    ----------
    """
    # Use a `DiffIndex` to get the changed files.
    deltas = []

    a_repo: git.Repo = get_ephemeral_repo(DELETED_SUFFIX, ref, md5sum)
    b_repo: git.Repo = ref.repo

    a_dir: ExtantDir = working_dir(a_repo)
    b_dir: ExtantDir = working_dir(b_repo)

    diff_index = b_repo.commit(ref.sha).diff(b_repo.head.commit)
    for change_type in GitChangeType:
        for diff in diff_index.iter_change_type(change_type.value):

            if not ignore_fn(a_dir / diff.a_path) or not ignore_fn(a_dir / diff.b_path):
                logger.warning(f"Ignoring:\n{diff.a_path}\n{diff.b_path}")
                continue

            a_path = ftest(a_dir / diff.a_path)
            b_path = ftest(b_dir / diff.b_path)

            a_relpath = Path(diff.a_path)
            b_relpath = Path(diff.b_path)

            if change_type == GitChangeType.DELETED:
                if not isinstance(a_path, ExtantFile):
                    logger.warning(f"Deleted file not found in source commit: {a_path}")
                    continue

                deltas.append(Delta(change_type, a_path, a_relpath))
                continue

            if change_type == GitChangeType.RENAMED:
                a_flatnote: FlatNote = parse_markdown_note(parser, transformer, a_path)
                b_flatnote: FlatNote = parse_markdown_note(parser, transformer, b_path)
                if a_flatnote.nid != b_flatnote.nid:
                    deltas.append(Delta(GitChangeType.DELETED, a_path, a_relpath))
                    deltas.append(Delta(GitChangeType.ADDED, b_path, b_relpath))
                    continue

            if not isinstance(b_path, ExtantFile):
                logger.warning(f"Diff target not found: {b_path}")
                continue

            deltas.append(Delta(change_type, b_path, b_relpath))

    return deltas


@beartype
def get_head(repo: git.Repo) -> Optional[RepoRef]:
    """Get the HEAD ref, or None if it doesn't exist."""
    maybe_ref = MaybeHeadRepoRef(repo)
    return None if isinstance(maybe_ref.value, str) else JUST(maybe_ref)


@beartype
def frglob(root: ExtantDir, pattern: str) -> List[ExtantFile]:
    """Call root.rglob() and returns only files."""
    files = filter(lambda p: isinstance(p, ExtantFile), map(ftest, root.rglob(pattern)))
    return list(files)


@beartype
def get_models_recursively(kirepo: KiRepo) -> Dict[int, NotetypeDict]:
    """Find and merge all ``models.json`` files recursively."""
    all_new_models: Dict[int, NotetypeDict] = {}

    # Load notetypes from json files.
    for models_file in frglob(kirepo.root, MODELS_FILE):
        with open(models_file, "r", encoding="UTF-8") as models_f:
            new_models: Dict[int, NotetypeDict] = json.load(models_f)

        # Add mappings to dictionary.
        all_new_models.update(new_models)

    return all_new_models


@beartype
def display_fields_health_warning(note: anki.notes.Note) -> int:
    """Display warnings when Anki's fields health check fails."""
    health = note.fields_check()
    if health == 1:
        logger.warning(f"Found empty note:\n {note}")
        logger.warning(f"Fields health check code: {health}")
    elif health == 2:
        logger.warning(f"\nFound duplicate note when adding new note w/ nid {note.id}.")
        logger.warning(f"Notetype/fields of note {note.id} match existing note.")
        logger.warning("Note was not added to collection!")
        logger.warning(f"First field: {note.fields[0]}")
        logger.warning(f"Fields health check code: {health}")
    elif health != 0:
        logger.error(f"Failed to process note '{note.id}'.")
        logger.error(f"Note failed fields check with unknown error code: {health}")
    return health


# UNREFACTORED


@beartype
def parse_markdown_note(
    parser: Lark, transformer: NoteTransformer, notes_file: ExtantFile
) -> FlatNote:
    """Parse with lark."""
    tree = parser.parse(notes_file.read_text(encoding="UTF-8"))
    flatnotes: List[FlatNote] = transformer.transform(tree)

    # UNSAFE!
    return flatnotes[0]


# TODO: Refactor into a safe function.
@beartype
def update_kinote(kinote: KiNote, flatnote: FlatNote) -> None:
    """
    Update an `apy` Note in a collection.

    Currently fails if the model has changed. It does not update the model name
    of the KiNote, and it also assumes that the field names have not changed.
    """
    kinote.n.tags = flatnote.tags

    # TODO: Can this raise an error? What if the deck is not in the collection?
    # Should we create it? Probably.
    kinote.set_deck(flatnote.deck)

    # Get new notetype from collection (None if nonexistent).
    notetype: Optional[NotetypeDict] = kinote.n.col.models.by_name(flatnote.model)

    # If notetype doesn't exist, raise an error.
    if notetype is None:
        msg = f"Notetype '{flatnote.model}' doesn't exist. "
        msg += "Create it in Anki before adding notes via ki."
        raise FileNotFoundError(msg)

    # Validate field keys against notetype.
    old_model = kinote.n.note_type()
    validate_flatnote_fields(old_model, flatnote)

    # Set field values.
    for key, field in flatnote.fields.items():
        if flatnote.markdown:
            kinote.n[key] = markdown_to_html(field)
        else:
            kinote.n[key] = plain_to_html(field)

    # Flush note, mark collection as modified, and display any warnings.
    kinote.n.flush()
    kinote.a.modified = True
    display_fields_health_warning(kinote.n)


# TODO: Refactor into a safe function.
@beartype
def validate_flatnote_fields(
    model: NotetypeDict, flatnote: FlatNote
) -> Optional[Exception]:
    """Validate that the fields given in the note match the notetype."""
    # Set current notetype for collection to `model_name`.
    model_field_names = [field["name"] for field in model["flds"]]

    if len(flatnote.fields.keys()) != len(model_field_names):
        return ValueError(f"Not enough fields for model {flatnote.model}!")

    for x, y in zip(model_field_names, flatnote.fields.keys()):
        if x != y:
            return ValueError("Inconsistent field names " f"({x} != {y})")
    return None


@beartype
def set_model(a: Anki, model_name: str) -> NotetypeDict:
    """Set current model based on model name."""
    current = a.col.models.current(for_deck=False)
    if current["name"] == model_name:
        return current

    model = a.get_model(model_name)
    if model is None:
        # UNSAFE!
        echo(f'Model "{model_name}" was not recognized!')
        raise ValueError

    a.col.models.set_current(model)
    return model


# TODO: Refactor into a safe function.
@beartype
def add_note_from_flatnote(a: Anki, flatnote: FlatNote) -> Optional[KiNote]:
    """Add a note given its FlatNote representation, provided it passes health check."""
    # TODO: Does this assume model exists? Kind of. We abort/raise an error if
    # it doesn't.
    model = set_model(a, flatnote.model)
    validate_flatnote_fields(model, flatnote)

    # Note that we call ``a.set_model(flatnote.model)`` above, so the current
    # model is the model given in ``flatnote``.
    notetype = a.col.models.current(for_deck=False)
    note = a.col.new_note(notetype)

    # Create new deck if deck does not exist.
    note.note_type()["did"] = a.col.decks.id(flatnote.deck)

    if flatnote.markdown:
        note.fields = [markdown_to_html(x) for x in flatnote.fields.values()]
    else:
        note.fields = [plain_to_html(x) for x in flatnote.fields.values()]

    for tag in flatnote.tags:
        note.add_tag(tag)

    health = display_fields_health_warning(note)

    result = None
    if health == 0:
        a.col.addNote(note)
        a.modified = True
        result = KiNote(a, note)

    return result


FieldDict = Dict[str, Any]


# TODO: This function is still somewhat unsafe.
@beartype
def get_sort_field_text(
    a: Anki, note: anki.notes.Note, notetype: NotetypeDict
) -> Optional[str]:
    """
    Return the sort field content of a model.

    .field_map() call may raise a KeyError if ``notetype`` doesn't have the
    requisite fields.
    notetype may not have a sortf key
    sort_idx may not be in field_names
    sort_field_name may not be in note
    """

    # Get fieldmap from notetype.
    fieldmap: Dict[str, Tuple[int, FieldDict]]
    try:
        fieldmap = a.col.models.field_map(notetype)
    except KeyError as err:
        logger.warning(err)
        return None

    # Get sort index.
    try:
        sort_idx = a.col.models.sort_idx(notetype)
    except KeyError as err:
        logger.warning(err)
        return None

    # Map field indices to field names.
    sort_field_name: Optional[str] = None
    for field_name, (idx, _) in fieldmap.items():
        if idx == sort_idx:
            sort_field_name = field_name

    if sort_field_name is None:
        return None

    # Get sort field_name.
    try:
        sort_field_text: str = note[sort_field_name]
    except KeyError as err:
        logger.warning(err)
        return None

    return sort_field_text


@beartype
def get_note_path(sort_field_text: str, deck_dir: ExtantDir) -> ExtantFile:
    """Get note path from sort field text."""
    field_text = sort_field_text

    MD = ".md"

    # Construct filename, stripping HTML tags and sanitizing (quickly).
    field_text = plain_to_html(field_text)
    field_text = re.sub("<[^<]+?>", "", field_text)
    name = field_text[:MAX_FIELNAME_LEN]
    name = Path(slugify(name, allow_unicode=True))
    filename: Singleton = singleton(str(name.with_suffix(MD)))
    note_path = ftest(deck_dir / filename)

    i = 1
    while not isinstance(note_path, NoPath):
        filename: Singleton = singleton(str(Path(f"{name}_{i}").with_suffix(MD)))
        note_path = ftest(deck_dir / filename)
        i += 1

    note_path: ExtantFile = ftouch(deck_dir, filename)

    return note_path


# TODO: Refactor into a safe function.
@beartype
def backup(col_file: ExtantFile) -> None:
    """Backup collection to `.ki/backups`."""
    md5sum = md5(col_file)
    backupsdir = Path.cwd() / ".ki" / "backups"
    assert not backupsdir.is_file()
    if not backupsdir.is_dir():
        backupsdir.mkdir()
    backup_path = backupsdir / f"{md5sum}.anki2"
    if backup_path.is_file():
        click.secho("Backup already exists.", bold=True)
        return
    assert not backup_path.is_file()
    echo(f"Writing backup of .anki2 file to '{backupsdir}'")
    shutil.copyfile(col_file, backup_path)
    assert backup_path.is_file()


@beartype
def append_md5sum(
    kidir: ExtantDir, tag: str, md5sum: str, silent: bool = False
) -> None:
    """Append an md5sum hash to the hashes file."""
    hashes_file = kidir / HASHES_FILE
    with open(hashes_file, "a+", encoding="UTF-8") as hashes_f:
        hashes_f.write(f"{md5sum}  {tag}\n")
    echo(f"Wrote md5sum to '{hashes_file}'", silent)


@beartype
def create_deck_dir(deck_name: str, targetdir: ExtantDir) -> ExtantDir:
    """Construct path to deck directory and create it."""
    # Strip leading periods so we don't get hidden folders.
    components = deck_name.split("::")
    components = [re.sub(r"^\.", r"", comp) for comp in components]
    deck_path = Path(targetdir, *components)
    return fforce_mkdir(deck_path)


@beartype
def get_field_note_id(nid: int, fieldname: str) -> str:
    """A str ID that uniquely identifies field-note pairs."""
    return f"{nid}{slugify(fieldname, allow_unicode=True)}"


@beartype
def get_batches(lst: List[ExtantFile], n: int) -> Generator[ExtantFile, None, None]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


@beartype
def write_notes(col_file: ExtantFile, targetdir: ExtantDir, silent: bool):
    """Write notes to appropriate directories in ``targetdir``."""
    # Create temp directory for htmlfield text files.
    tempdir: EmptyDir = fmkdtemp()
    root: EmptyDir = fmksubdir(tempdir, FIELD_HTML_SUFFIX)

    paths: Dict[str, ExtantFile] = {}
    decks: Dict[str, List[KiNote]] = {}

    # Open deck with `apy`, and dump notes and markdown files.
    with Anki(path=col_file) as a:
        all_nids = list(a.col.find_notes(query=""))
        for nid in tqdm(all_nids, ncols=TQDM_NUM_COLS, disable=silent):
            kinote = KiNote(a, a.col.get_note(nid))
            decks[kinote.deck] = decks.get(kinote.deck, []) + [kinote]
            for fieldname, fieldtext in kinote.fields.items():
                if re.search(HTML_REGEX, fieldtext):
                    fid: str = get_field_note_id(nid, fieldname)
                    html_file: ExtantFile = ftouch(root, singleton(fid))
                    html_file.write_text(fieldtext)
                    paths[fid] = html_file

        tidy_html_recursively(root, silent)

        # Write models to disk.
        models_map: Dict[int, NotetypeDict] = {}
        for model in a.col.models.all():
            # UNSAFE!
            model_id: Optional[NotetypeDict] = a.col.models.id_for_name(model["name"])
            models_map[model_id] = model

        with open(targetdir / MODELS_FILE, "w", encoding="UTF-8") as f:
            json.dump(models_map, f, ensure_ascii=False, indent=4)

        deck_model_ids: Set[int] = set()
        for deck_name in sorted(set(decks.keys()), key=len, reverse=True):
            deck_dir: ExtantDir = create_deck_dir(deck_name, targetdir)
            for kinote in decks[deck_name]:

                # Add the notetype id.
                model = kinote.n.note_type()
                if model is None:
                    logger.warning(f"Couldn't find notetype for {kinote}")
                    continue
                model_id = a.col.models.id_for_name(model["name"])
                deck_model_ids.add(model_id)

                sort_field_text: Optional[str] = get_sort_field_text(a, kinote.n, model)
                if sort_field_text is None:
                    logger.warning(f"Couldn't find sort field for {kinote}")
                    continue
                notepath: ExtantFile = get_note_path(sort_field_text, deck_dir)
                payload = get_tidy_payload(kinote, paths)
                notepath.write_text(payload, encoding="UTF-8")

            # Write ``models.json`` for current deck.
            deck_models_map = {mid: models_map[mid] for mid in deck_model_ids}
            with open(deck_dir / MODELS_FILE, "w", encoding="UTF-8") as f:
                json.dump(deck_models_map, f, ensure_ascii=False, indent=4)

    shutil.rmtree(root)


@beartype
def get_tidy_payload(kinote: KiNote, paths: Dict[str, ExtantFile]) -> str:
    """Get the payload for the note (HTML-tidied if necessary)."""
    # Get tidied html if it exists.
    tidyfields = {}
    for fieldname, fieldtext in kinote.fields.items():
        fid = get_field_note_id(kinote.n.id, fieldname)
        if fid in paths:
            tidyfields[fieldname] = paths[fid].read_text()
        else:
            tidyfields[fieldname] = fieldtext

    # Construct note repr from tidyfields map.
    lines = kinote.get_header_lines()
    for fieldname, fieldtext in tidyfields.items():
        lines.append("### " + fieldname)
        lines.append(fieldtext)
        lines.append("")

    return "\n".join(lines)


# TODO: Refactor into a safe function.
@beartype
def git_subprocess_pull(remote: str, branch: str) -> int:
    """Pull remote into branch using a subprocess call."""
    p = subprocess.run(
        [
            "git",
            "pull",
            "-v",
            "--allow-unrelated-histories",
            "--strategy-option",
            "theirs",
            remote,
            branch,
        ],
        check=False,
        capture_output=True,
    )
    pull_stderr = p.stderr.decode()
    logger.debug(f"\n{pull_stderr}")
    logger.debug(f"Return code: {p.returncode}")
    if p.returncode != 0:
        raise ValueError(pull_stderr)
    return p.returncode


@beartype
def echo(string: str, silent: bool = False) -> None:
    """Call `click.secho()` with formatting."""
    if not silent:
        click.secho(string, bold=True)
        # logger.info(string)


# TODO: Refactor into a safe function.
@beartype
def slugify(value: str, allow_unicode: bool = False) -> str:
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


# TODO: Refactor into a safe function.
@beartype
def tidy_html_recursively(root: ExtantDir, silent: bool) -> None:
    """Call html5-tidy on each file in ``root``, editing in-place."""
    # Spin up subprocesses for tidying field HTML in-place.
    batches: List[List[ExtantFile]] = list(get_batches(frglob(root, "*"), BATCH_SIZE))
    for batch in tqdm(batches, ncols=TQDM_NUM_COLS, disable=silent):

        # Fail silently here, so as to not bother user with tidy warnings.
        command = ["tidy", "-q", "-m", "-i", "-omit", "-utf8", "--tidy-mark", "no"]
        command += batch
        subprocess.run(command, check=False, capture_output=True)


@beartype
def get_stage_repo(head: KiRepoRef, md5sum: str) -> KiRepo:
    # Copy current kirepo into a temp directory (the STAGE), hard reset to HEAD.
    stage_kirepo: KiRepo = get_ephemeral_kirepo(STAGE_SUFFIX, head, md5sum)

    # Unsubmodule the stage repo.
    unsubmodule_repo(stage_kirepo.repo)

    # Shutil rmtree the stage repo .git directory.
    stage_git_dir: NoPath = rmtree(git_dir(stage_kirepo.repo))
    stage_root: ExtantDir = stage_kirepo.root
    del stage_kirepo

    # Copy the .git folder from ``no_submodules_tree`` into the stage repo.
    copytree(git_dir(head.kirepo.no_modules_repo), stage_git_dir)

    # Reload stage kirepo.
    stage_kirepo: KiRepo = IO(get_ki_repo(stage_root))

    return stage_kirepo


@beartype
def update_no_submodules_tree(kirepo: KiRepo, stage_root: ExtantDir) -> None:
    no_modules_root: NoPath = rmtree(working_dir(kirepo.no_modules_repo))
    copytree(stage_root, no_modules_root)


Maybe = Union[
    MaybeError,
    MaybeNoPath,
    MaybeEmptyDir,
    MaybeExtantDir,
    MaybeExtantFile,
    MaybeRepo,
    MaybeKiRepo,
    MaybeRepoRef,
    MaybeKiRepoRef,
    MaybeHeadRepoRef,
    MaybeHeadKiRepoRef,
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
def JUST(maybe: Maybe) -> Any:
    """
    UNSAFE: ASSUMES THIS ``maybe`` is not an error!  Unfolds the Maybe type to
    yield an object of type ``maybe.type``.
    """
    if not isinstance(maybe.value, maybe.type):
        msg = "JUST() called on an errored-out 'Maybe' value!\n"
        msg += f"A fundamental assumption (that maybe.value : '{maybe.value}' "
        msg += f"is always of type maybe.type : '{maybe.type}') is wrong!"
        raise TypeError(msg)
    return maybe.value


@click.group()
@click.version_option()
@beartype
def ki() -> None:
    """
    The universal CLI entry point for `ki`.

    Takes no arguments, only has three subcommands (clone, pull, push).
    """
    return


@beartype
def TARGET_EXISTS_MSG(targetdir: ExtantDir) -> str:
    msg = f"fatal: destination path '{targetdir}' already exists "
    msg += "and is not an empty directory."
    return msg


@ki.command()
@click.argument("collection")
@click.argument("directory", required=False, default="")
def clone(collection: str, directory: str = "") -> None:
    """
    Clone an Anki collection into a directory.

    Parameters
    ----------
    collection : str
        The path to a `.anki2` collection file.
    directory : str, default=""
        An optional path to a directory to clone the collection into.
        Note: we check that this directory does not yet exist.
    """
    echo("Cloning.")
    col_file: ExtantFile = IO(MaybeExtantFile(Path(collection)))

    # Create default target directory.
    cwd: ExtantDir = fcwd()
    targetdir = ftest(Path(directory) if directory != "" else cwd / col_file.stem)
    if isinstance(targetdir, NoPath):
        targetdir: EmptyDir = fmkdir(targetdir)
    if not isinstance(targetdir, EmptyDir):
        return echo(TARGET_EXISTS_MSG(targetdir))

    md5sum = _clone(col_file, targetdir, msg="Initial commit", silent=False)

    # Check that we are inside a ki repository, and get the associated collection.
    root: ExtantDir = ftest(targetdir)
    kirepo: KiRepo = IO(get_ki_repo(root))

    # Get reference to HEAD of current repo.
    head: KiRepoRef = IO(MaybeHeadKiRepoRef(kirepo))

    # Get staging repository in temp directory, and copy to ``no_submodules_tree``.
    stage_kirepo: KiRepo = get_stage_repo(head, md5sum)
    stage_kirepo.repo.git.add(all=True)
    stage_kirepo.repo.index.commit(f"Pull changes from ref {head.sha}")
    update_no_submodules_tree(kirepo, stage_kirepo.root)

    # Dump HEAD ref of current repo in ``.ki/last_push``.
    kirepo.last_push_file.write_text(head.sha)

    # TODO: Should figure out how to clean up in case of errors.
    if False:
        echo("Failed: exiting.")
        if targetdir.is_dir():
            shutil.rmtree(targetdir)

    return None


@beartype
def _clone(col_file: ExtantFile, targetdir: EmptyDir, msg: str, silent: bool) -> str:
    """
    Clone an Anki collection into a directory.

    Parameters
    ----------
    col_file : pathlib.Path
        The path to a `.anki2` collection file.
    targetdir : pathlib.Path
        A path to a directory to clone the collection into.
        Note: we check that this directory is empty.
    msg : str
        Message for initial commit.
    silent : bool
        Indicates whether we are calling `_clone()` from `pull()`.

    Returns
    -------
    git.Repo
        The cloned repository.
    """
    echo(f"Found .anki2 file at '{col_file}'", silent=silent)

    # Create .ki subdirectory.
    kidir: EmptyDir = fmksubdir(targetdir, Path(KI))

    # Create config file.
    config_file: ExtantFile = fmkleaves(kidir, [singleton(CONFIG_FILE)])[0]
    config = configparser.ConfigParser()
    config["remote"] = {"path": col_file}
    with open(config_file, "w", encoding="UTF-8") as config_f:
        config.write(config_f)

    # Append to hashes file.
    md5sum = md5(col_file)
    echo(f"Computed md5sum: {md5sum}", silent)
    append_md5sum(kidir, col_file.name, md5sum, silent)
    echo(f"Cloning into '{targetdir}'...", silent=silent)

    # Add `.ki/` to gitignore.
    ignore_path = targetdir / GITIGNORE_FILE
    ignore_path.write_text(".ki/\n")

    # Write notes to disk.
    write_notes(col_file, targetdir, silent)

    # Initialize git repo and commit contents.
    repo = git.Repo.init(targetdir, initial_branch=BRANCH_NAME)
    repo.git.add(all=True)
    _ = repo.index.commit(msg)

    return md5sum


@ki.command()
@beartype
def pull() -> None:
    """
    Pull from a preconfigured remote Anki collection into an existing ki
    repository.
    """

    # Check that we are inside a ki repository, and get the associated collection.
    cwd: ExtantDir = fcwd()
    kirepo: KiRepo = IO(get_ki_repo(cwd))
    con: sqlite3.Connection = lock(kirepo.col_file)

    md5sum: str = md5(kirepo.col_file)
    hashes: List[str] = kirepo.hashes_file.read_text().split("\n")
    hashes = list(filter(lambda l: l != "", hashes))
    logger.debug(f"Hashes:\n{pp.pformat(hashes)}")
    if md5sum in hashes[-1]:
        echo("ki pull: up to date.")
        return unlock(con)

    echo(f"Pulling from '{kirepo.col_file}'")
    echo(f"Computed md5sum: {md5sum}")

    # Git clone `repo` at commit SHA of last successful `push()`.
    sha: str = kirepo.last_push_file.read_text()
    ref: RepoRef = IO(MaybeRepoRef((kirepo.repo, sha)))
    last_push_repo: git.Repo = get_ephemeral_repo(LOCAL_SUFFIX, ref, md5sum)

    # Ki clone collection into an ephemeral ki repository at `anki_remote_dir`.
    msg = f"Fetch changes from DB at '{kirepo.col_file}' with md5sum '{md5sum}'"
    anki_remote_dir: EmptyDir = fmksubdir(fmkdtemp(), REMOTE_SUFFIX / md5sum)
    _clone(kirepo.col_file, anki_remote_dir, msg, silent=True)

    # Create git remote pointing to anki remote repo.
    # TODO: Can this fail? What if the remote name already exists? Does the
    # remote path have to exist?
    # TODO: Should be an fcreate_local_remote() function which returns a Remote
    # object, and only takes a valid git repo.
    anki_remote = last_push_repo.create_remote(REMOTE_NAME, anki_remote_dir / GIT)

    # Pull anki remote repo into ``last_push_repo``.
    last_push_repo_root: ExtantDir = working_dir(last_push_repo)
    fchdir(last_push_repo_root)
    logger.debug(f"Pulling into {last_push_repo_root}")
    last_push_repo.git.config("pull.rebase", "false")
    git_subprocess_pull(REMOTE_NAME, BRANCH_NAME)
    last_push_repo.delete_remote(anki_remote)

    # Create remote pointing to ``last_push`` repo and pull into ``repo``.
    fchdir(cwd)
    last_push_remote = kirepo.repo.create_remote(REMOTE_NAME, git_dir(last_push_repo))
    kirepo.repo.git.config("pull.rebase", "false")
    p = subprocess.run(
        ["git", "pull", "-v", REMOTE_NAME, BRANCH_NAME],
        check=False,
        capture_output=True,
    )
    click.secho(f"{p.stdout.decode()}", bold=True)
    click.secho(f"{p.stderr.decode()}", bold=True)
    kirepo.repo.delete_remote(last_push_remote)

    # Append to hashes file.
    append_md5sum(cwd / ".ki", kirepo.col_file.name, md5sum)

    # Check that md5sum hasn't changed.
    if md5(kirepo.col_file) != md5sum:
        logger.warning(f"Checksum mismatch on {kirepo.col_file}. Was file changed?")
    return unlock(con)


# PUSH


@ki.command()
@beartype
def push() -> None:
    """Push a ki repository into a .anki2 file."""
    pp.install_extras(exclude=["ipython", "django", "ipython_repr_pretty"])

    # Check that we are inside a ki repository, and get the associated collection.
    cwd: ExtantDir = fcwd()
    kirepo: KiRepo = IO(get_ki_repo(cwd))
    con: sqlite3.Connection = lock(kirepo.col_file)

    md5sum: str = md5(kirepo.col_file)
    hashes: List[str] = kirepo.hashes_file.read_text().split("\n")
    hashes = list(filter(lambda l: l != "", hashes))
    logger.debug(f"Hashes:\n{pp.pformat(hashes)}")
    if md5sum not in hashes[-1]:
        IO(updates_rejected_message(kirepo.col_file))

    # Get reference to HEAD of current repo.
    head: KiRepoRef = IO(MaybeHeadKiRepoRef(kirepo))

    # Get staging repository in temp directory.
    # TODO: Consider making this return a KiRepoRef to ``head_1``.
    stage_kirepo: KiRepo = get_stage_repo(head, md5sum)
    head_1: RepoRef = IO(MaybeHeadRepoRef(stage_kirepo.repo))
    stage_kirepo.repo.git.add(all=True)
    stage_kirepo.repo.index.commit(f"Pull changes from ref {head.sha}")

    # Get filter function.
    ignore_fn = functools.partial(path_ignore_fn, patterns=IGNORE, root=kirepo.root)

    # This may error if there is no head commit in the current repository.
    head_kirepo: KiRepo = get_ephemeral_kirepo(LOCAL_SUFFIX, head, md5sum)

    # Read grammar.
    # UNSAFE! Should we assume this always exists? A nice error message should
    # be printed on initialization if the grammar file is missing. No
    # computation should be done, and none of the click commands should work.
    grammar_path = Path(__file__).resolve().parent / "grammar.lark"
    grammar = grammar_path.read_text(encoding="UTF-8")

    # Instantiate parser.
    parser = Lark(grammar, start="file", parser="lalr")
    transformer = NoteTransformer()

    # Get deltas.
    deltas = get_deltas_since_last_push(head_1, md5sum, ignore_fn, parser, transformer)
    new_models: Dict[int, NotetypeDict] = get_models_recursively(head_kirepo)

    # If there are no changes, quit.
    if len(set(deltas)) == 0:
        return echo("ki push: up to date.")

    echo(f"Pushing to '{kirepo.col_file}'")
    echo(f"Computed md5sum: {md5sum}")
    echo(f"Verified md5sum matches latest hash in '{kirepo.hashes_file}'")

    # Copy collection to a temp directory.
    temp_col_dir = fmkdtemp()
    new_col_file = temp_col_dir / kirepo.col_file.name
    new_col_file: ExtantFile = fcopyfile(kirepo.col_file, new_col_file, temp_col_dir)
    head: Optional[RepoRef] = get_head(kirepo.repo)
    if head is None:
        return echo("Failed: no commits in repository. Couldn't find HEAD ref.")
    echo(f"Generating local .anki2 file from latest commit: {head.sha}")
    echo(f"Writing changes to '{new_col_file}'...")

    # Edit the copy with `apy`.
    with Anki(path=new_col_file) as a:

        # Add all new models.
        for new_model in new_models.values():
            # NOTE: Mutates ``new_model`` (appends checksum to name).
            # UNSAFE: Potential KeyError!
            # This can be fixed by writing a strict type for NotetypeDict.
            if a.col.models.id_for_name(new_model["name"]) is None:
                a.col.models.ensure_name_unique(new_model)
                a.col.models.add(new_model)

        # Gather logging statements to display.
        log: List[str] = []
        msg = ""
        num_new_nids = 0

        # Stash both unstaged and staged files (including untracked).
        kirepo.repo.git.stash(include_untracked=True, keep_index=True)
        kirepo.repo.git.reset("HEAD", hard=True)

        is_delete = lambda d: d.status == GitChangeType.DELETED
        deletes: List[Delta] = list(filter(is_delete, deltas))
        logger.debug(f"Deleting {len(deletes)} notes.")

        # TODO: All this logic can be abstracted away from the process of
        # actually parsing notes and constructing Anki-specific objects. This
        # is just a series of filesystem ops. They should be put in a
        # standalone function and tested without anything related to Anki.
        for delta in tqdm(deltas, ncols=TQDM_NUM_COLS):

            if is_delete(delta):
                a.col.remove_notes(
                    [parse_markdown_note(parser, transformer, delta.path).nid]
                )
                a.modified = True
                continue

            # Get flatnote from parser, and add/edit/delete in collection.
            flatnote = parse_markdown_note(parser, transformer, delta.path)

            # If a kinote with this nid exists in DB, update it.
            # TODO: If relevant prefix of sort field has changed, we regenerate
            # the file. Recall that the sort field is used to determine the
            # filename. If the content of the sort field has changed, then we
            # may need to update the filename.
            try:
                kinote: KiNote = KiNote(a, a.col.get_note(flatnote.nid))
                update_kinote(kinote, flatnote)

            # Otherwise, we generate/reassign an nid for it.
            except anki.errors.NotFoundError:
                kinote: Optional[KiNote] = add_note_from_flatnote(a, flatnote)

                if kinote is not None:
                    log.append(f"Reassigned nid: '{flatnote.nid}' -> '{kinote.n.id}'")

                    # Get paths to note in local repo, as distinct from staging repo.
                    repo_note_path: Path = kirepo.root / delta.relpath

                    # If this is not an entirely new file, remove it.
                    if repo_note_path.is_file():
                        repo_note_path.unlink()

                    # Construct markdown file contents and write.
                    # UNSAFE: Can't pass None to 'get_sort_field_text()'. Need
                    # a ``MaybeNotetype``.
                    model: Optional[NotetypeDict] = kinote.n.note_type()
                    if model is None:
                        logger.warning(f"Couldn't find notetype for {kinote}")
                        continue
                    sort_field_text: Optional[str] = get_sort_field_text(
                        a, kinote.n, model
                    )
                    if sort_field_text is None:
                        logger.warning(f"Couldn't find sort field for {kinote}")
                        continue
                    parent: ExtantDir = fforce_mkdir(repo_note_path.parent)
                    new_note_path: ExtantFile = get_note_path(sort_field_text, parent)
                    new_note_path.write_text(str(kinote), encoding="UTF-8")

                    new_note_relpath = os.path.relpath(new_note_path, kirepo.root)
                    num_new_nids += 1
                    msg += f"Wrote note '{kinote.n.id}' in file {new_note_relpath}\n"

        # Commit nid reassignments.
        logger.warning(f"Reassigned {num_new_nids} nids.")
        if num_new_nids > 0:
            msg = "Generated new nid(s).\n\n" + msg

            # Commit in all submodules (doesn't support recursing yet).
            for sm in kirepo.repo.submodules:
                subrepo: git.Repo = sm.update().module()
                subrepo.git.add(all=True)
                subrepo.index.commit(msg)

            # Commit in main repository.
            kirepo.repo.git.add(all=True)
            _ = kirepo.repo.index.commit(msg)

    # Backup collection file and overwrite collection.
    backup(kirepo.col_file)
    new_col_file = fcopyfile(new_col_file, kirepo.col_file, fparent(kirepo.col_file))
    echo(f"Overwrote '{kirepo.col_file}'")

    # Append to hashes file.
    new_md5sum = md5(new_col_file)
    append_md5sum(kirepo.ki_dir, new_col_file.name, new_md5sum, silent=False)

    # Dump HEAD ref of current repo in ``.ki/last_push``.
    update_no_submodules_tree(kirepo, stage_kirepo.root)
    kirepo.last_push_file.write_text(head.sha)

    # Unlock Anki SQLite DB.
    unlock(con)

    return None
