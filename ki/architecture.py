#!/usr/bin/env python3
"""
Push architecture redesign.
"""

# pylint: disable=invalid-name, unused-import, missing-class-docstring, broad-except
# pylint: disable=too-many-return-statements, too-many-lines

import os
import re
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
from pyinstrument import Profiler

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
IGNORE = [".git", ".ki", ".gitignore", ".gitmodules", "models.json"]
LOCAL_SUFFIX = Path("ki/local")
NO_SM_SUFFIX = Path("ki/no_submodules")
DELETED_SUFFIX = Path("ki/deleted")
NO_SM_GIT_SUFFIX = Path("ki/no_submodules_git")

REMOTE_CONFIG_SECTION = "remote"
COLLECTION_FILE_PATH_CONFIG_FIELD = "path"


class ExtantFile(type(Path())):
    """UNSAFE: Indicates that file *was* extant when it was resolved."""


class ExtantDir(type(Path())):
    """UNSAFE: Indicates that dir *was* extant when it was resolved."""


class NoPath(type(Path())):
    """UNSAFE: Indicates that path *was not* extant when it was resolved."""


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
    models_file: ExtantFile
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
def ftest(path: Path) -> Union[ExtantFile, ExtantDir, ExtantStrangePath, NoPath]:
    """
    Test whether ``path`` is a file, a directory, or something else. If
    something else, we consider that, for the sake of simplicity, a
    NoPath.
    """
    if path.is_file():
        return JUST(MaybeExtantFile(path))
    if path.is_dir():
        return JUST(MaybeExtantDir(path))
    if path.exists():
        return ExtantStrangePath(path)
    return JUST(MaybeNoPath(path))


@beartype
def ftouch(path: NoPath) -> ExtantFile:
    """Touch a file."""
    path.touch()
    return JUST(MaybeExtantFile(path))


@beartype
def fmkdir(path: NoPath) -> ExtantDir:
    """Make a directory (with parents)."""
    path.mkdir(parents=True)
    return JUST(MaybeExtantDir(path))


@beartype
def fforce_mkdir(path: Path) -> ExtantDir:
    """Make a directory (with parents, ok if it already exists)."""
    path.mkdir(parents=True, exist_ok=True)
    return JUST(MaybeExtantDir(path))


@beartype
def fparent(path: Union[ExtantFile, ExtantDir]) -> Union[ExtantFile, ExtantDir]:
    """Get the parent of a path that exists."""
    if path.resolve() == FS_ROOT:
        return JUST(MaybeExtantDir(FS_ROOT))
    if isinstance(path, ExtantFile):
        return JUST(MaybeExtantDir(path.parent))
    return JUST(MaybeExtantDir(path.parent))


@beartype
def fmkdtemp() -> ExtantDir:
    """Make a temporary directory (in /tmp)."""
    return JUST(MaybeExtantDir(Path(tempfile.mkdtemp())))


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
        hashes_file = ftouch(hashes_file)
    elif not isinstance(hashes_file, ExtantFile):
        return MaybeKiRepo(f"Not found or not a file: {hashes_file}")

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
def updates_rejected_message(col_file: Path) -> str:
    """Generate need-to-pull message."""
    return f"Failed to push some refs to '{col_file}'\n{HINT}"


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
    tempdir = fmkdtemp()
    root = fforce_mkdir(Path(tempdir, suffix))

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
def get_ephemeral_kirepo(
    suffix: Path, kirepo_ref: KiRepoRef, md5sum: str
) -> KiRepo:
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
        sm_path: ExtantDir = JUST(MaybeExtantDir(sm.module().working_tree_dir))
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
        ref: RepoRef, md5sum: str, ignore_fn: Callable[[Path], bool]
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

    logger.debug(f"Old ref: {ref.sha}")
    logger.debug(f"New ref: {b_repo.head.commit.hexsha}")

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

            logger.debug(f"{a_path = }")
            logger.debug(f"{b_path = }")

            if change_type == GitChangeType.DELETED:
                if not isinstance(a_path, ExtantFile):
                    logger.warning(f"Deleted file not found in source commit: {a_path}")
                    continue

                deltas.append(Delta(change_type, a_path, a_relpath))
                continue

            if change_type == GitChangeType.RENAMED:
                a_flatnote: FlatNote = parse_markdown_note(a_path)
                b_flatnote: FlatNote = parse_markdown_note(b_path)
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
def get_all_nids(root: ExtantDir) -> List[int]:
    """Get all nids from all notes in current directory."""
    files: List[ExtantFile] = frglob(root, "*")
    files = filter(ignore_fn, files)
    files = filter(is_anki_note, files)


@beartype
def get_note_file_git_deltas(
    root: ExtantDir,
    repo_ref: Optional[RepoRef],
    md5sum: str,
    ignore_fn: Callable[[Path], bool],
) -> List[Delta]:
    """Gets a list of paths to modified/new/deleted note md files since commit."""
    deltas: List[Delta] = []

    # Treat case where there is no last push.
    # TODO: Should this ever be the case?
    if repo_ref is None:
        logger.warning("HEAD~1 is None, globbing and adding everything.")
        files: List[ExtantFile] = frglob(root, "*")
        files = filter(ignore_fn, files)
        for file in files:
            # UNSAFE: relative_to() can raise a ValueError!
            deltas.append(Delta(GitChangeType.ADDED, file, file.relative_to(root)))
    else:
        logger.debug("Calculating diff via gitpython.")
        deltas = get_deltas_since_last_push(repo_ref, md5sum, ignore_fn)

    return deltas


@beartype
def get_head(repo: git.Repo) -> Optional[RepoRef]:
    """Get the HEAD ref, or None if it doesn't exist."""
    maybe_ref = MaybeHeadRepoRef(repo)
    return None if isinstance(maybe_ref.value, str) else JUST(maybe_ref)


def frglob(root: ExtantDir, pattern: str) -> List[ExtantFile]:
    """Call root.rglob() and returns only files."""
    return filter(lambda p: isinstance(p, ExtantFile), map(ftest, root.rglob(pattern)))


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
def parse_markdown_note(notes_file: ExtantFile) -> FlatNote:
    """Parse with lark."""
    # Read grammar.
    # UNSAFE! Should we assume this always exists? A nice error message should
    # be printed on initialization if the grammar file is missing. No
    # computation should be done, and none of the click commands should work.
    grammar_path = Path(__file__).resolve().parent / "grammar.lark"
    grammar = grammar_path.read_text(encoding="UTF-8")

    # Instantiate parser.
    parser = Lark(grammar, start="file", parser="lalr")
    transformer = NoteTransformer()
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
def validate_flatnote_fields(model: NotetypeDict, flatnote: FlatNote) -> None:
    """Validate that the fields given in the note match the notetype."""
    # Set current notetype for collection to `model_name`.
    model_field_names = [field["name"] for field in model["flds"]]

    if len(flatnote.fields.keys()) != len(model_field_names):
        logger.error(f"Not enough fields for model {flatnote.model}!")
        raise ValueError

    for x, y in zip(model_field_names, flatnote.fields.keys()):
        if x != y:
            logger.error("Inconsistent field names " f"({x} != {y})")
            raise ValueError


# TODO: Refactor into a safe function.
@beartype
def add_note_from_flatnote(a: Anki, flatnote: FlatNote) -> Optional[KiNote]:
    """Add a note given its FlatNote representation, provided it passes health check."""
    # TODO: Does this assume model exists?
    model = a.set_model(flatnote.model)
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


# TODO: Refactor into a safe function.
@beartype
def get_sort_field_name(a: Anki, notetype: Dict[str, Any]) -> str:
    """Return the sort field name of a model."""
    assert notetype is not None

    # Get fieldmap from notetype.
    fieldmap: Dict[str, Tuple[int, Dict[str, Any]]]
    fieldmap = a.col.models.field_map(notetype)

    # Map field indices to field names.
    fieldnames: Dict[int, str] = {}
    for fieldname, (idx, _) in fieldmap.items():
        assert idx not in fieldnames
        fieldnames[idx] = fieldname

    # Get sort fieldname.
    sort_idx = a.col.models.sort_idx(notetype)
    return fieldnames[sort_idx]


# TODO: Refactor into a safe function.
@beartype
def get_note_path(kinote: KiNote, sort_fieldname: str, deck_path: Path) -> Path:
    """Get note_path from sort field name."""
    # Get the filename for this note.
    assert sort_fieldname in kinote.n
    field_text = kinote.n[sort_fieldname]

    # Construct filename, stripping HTML tags and sanitizing (quickly).
    field_text = plain_to_html(field_text)
    field_text = re.sub("<[^<]+?>", "", field_text)
    filename = field_text[:MAX_FIELNAME_LEN]
    filename = slugify(filename, allow_unicode=True)
    basepath = deck_path / f"{filename}"
    note_path = basepath.with_suffix(".md")

    i = 1
    while note_path.exists():
        note_path = Path(f"{basepath}_{i}").with_suffix(".md")
        i += 1

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


# TODO: Refactor into a safe function.
@beartype
def append_md5sum(
    kidir: Path, colpath: Path, md5sum: str, silent: bool = False
) -> None:
    """Append an md5sum hash to the hashes file."""
    hashes_path = kidir / "hashes"
    with open(hashes_path, "a", encoding="UTF-8") as hashes_file:
        hashes_file.write(f"{md5sum}  {colpath.name}")
    echo(f"Wrote md5sum to '{hashes_path}'", silent)


@beartype
def echo(string: str, silent: bool = False) -> None:
    """Call `click.secho()` with formatting."""
    if not silent:
        # click.secho(string, bold=True)
        logger.info(string)


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


@beartype
def update_no_submodules_tree(head: KiRepoRef, md5sum: str) -> KiRepo:
    """
    Update the contents of ``kirepo`` in ``.ki/no_submodules_tree``.
    """

    kirepo = head.kirepo
    root = kirepo.root

    # Copy .git folder out of existing .ki/no_submodules_tree into a temp directory.
    no_sm_git_path = JUST(MaybeNoPath(fmkdtemp() / NO_SM_GIT_SUFFIX))
    no_sm_git_dir: ExtantDir = copytree(git_dir(kirepo.no_modules_repo), no_sm_git_path)

    # Shutil rmtree the entire thing.
    # ``kirepo`` now has a NoPath attribute where it should be extant,
    # so we delete the reference to avoid using an object with invalid type.
    kirepo_no_modules_dir: NoPath = rmtree(working_dir(kirepo.no_modules_repo))
    del kirepo

    # Copy current kirepo into a temp directory, hard reset to HEAD.
    no_sm_kirepo: KiRepo = get_ephemeral_kirepo(NO_SM_SUFFIX, head, md5sum)

    # Unsubmodule the temp repo.
    unsubmodule_repo(no_sm_kirepo.repo)

    # Shutil rmtree the temp repo .git directory.
    no_sm_kirepo_git_dir: NoPath = rmtree(git_dir(no_sm_kirepo.repo))
    no_sm_kirepo_root: ExtantDir = no_sm_kirepo.root
    del no_sm_kirepo

    # Copy the other .git folder back in.
    copytree(no_sm_git_dir, no_sm_kirepo_git_dir)

    # Copy the temporary repo into .ki/no_submodules_tree.
    kirepo_no_modules_dir = copytree(no_sm_kirepo_root, kirepo_no_modules_dir)

    # Reload top-level repo.
    kirepo: KiRepo = IO(get_ki_repo(root))

    return kirepo


Maybe = Union[
    MaybeNoPath,
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


# PUSH


@beartype
def _push() -> None:
    """Push a ki repository into a .anki2 file."""
    pp.install_extras()
    profiler = Profiler()
    profiler.start()

    # Check that we are inside a ki repository, and get the associated collection.
    cwd: ExtantDir = fcwd()
    kirepo: KiRepo = IO(get_ki_repo(cwd))
    con: sqlite3.Connection = lock(kirepo.col_file)

    md5sum: str = md5(kirepo.col_file)
    hashes: str = kirepo.hashes_file.read_text().split("\n")[-1]
    if md5sum not in hashes:
        IO(updates_rejected_message(kirepo.col_file))

    # Get reference to HEAD of current repo.
    head: KiRepoRef = IO(MaybeHeadKiRepoRef(kirepo))

    # Commit in no_submodules_tree.
    kirepo: KiRepo = update_no_submodules_tree(head, md5sum)
    head_1: Optional[RepoRef] = get_head(kirepo.no_modules_repo)
    kirepo.no_modules_repo.git.add(all=True)
    kirepo.no_modules_repo.index.commit("Add updated submodule-less copy of repo.")

    # Get filter function.
    ignore_fn = functools.partial(path_ignore_fn, patterns=IGNORE, root=kirepo.root)

    # This may error if there is no head commit in the current repository.
    head_kirepo: KiRepo = get_ephemeral_kirepo(LOCAL_SUFFIX, head, md5sum)

    # Get deltas.
    deltas = get_note_file_git_deltas(head_kirepo.root, head_1, md5sum, ignore_fn)
    new_models: Dict[int, NotetypeDict] = get_models_recursively(head_kirepo)

    logger.debug(f"Deltas:\n{pp.pformat(deltas)}")

    echo(f"Pushing to '{kirepo.col_file}'")
    echo(f"Computed md5sum: {md5sum}")
    echo(f"Verified md5sum matches latest hash in '{kirepo.hashes_file}'")

    # Copy collection to a temp directory.
    temp_col_dir = fmkdtemp()
    new_col_file = temp_col_dir / kirepo.col_file.name
    new_col_file: ExtantFile = fcopyfile(kirepo.col_file, new_col_file, temp_col_dir)
    head: Optional[RepoRef] = get_head(kirepo.repo)
    sha: Optional[str] = None if head is None else head.sha
    echo(f"Generating local .anki2 file from latest commit: {sha}")
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
        new_nid_path_map = {}

        p = subprocess.run(
            ["git", "rev-parse", "-q", "--verify", "refs/stash"],
            check=False,
            capture_output=True,
        )
        _stash_ref = p.stdout.decode()

        # Stash both unstaged and staged files (including untracked).
        kirepo.repo.git.stash(include_untracked=True, keep_index=True)
        kirepo.repo.git.reset("HEAD", hard=True)

        # TODO: All this logic can be abstracted away from the process of
        # actually parsing notes and constructing Anki-specific objects. This
        # is just a series of filesystem ops. They should be put in a
        # standalone function and tested without anything related to Anki.
        for delta in tqdm(deltas, ncols=TQDM_NUM_COLS):

            # If the file doesn't exist, parse its `nid` and then remove.
            if delta.status == GitChangeType.DELETED:
                flatnote = parse_markdown_note(delta.path)
                a.col.remove_notes([flatnote.nid])
                a.modified = True
                continue

            # Get flatnote from parser, and add/edit/delete in collection.
            flatnote = parse_markdown_note(delta.path)

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
                    sort_field_name = get_sort_field_name(a, kinote.n.note_type())
                    new_note_path = get_note_path(
                        kinote, sort_field_name, repo_note_path.parent
                    )
                    new_note_path.write_text(str(kinote), encoding="UTF-8")

                    new_note_relpath = os.path.relpath(new_note_path, kirepo.root)
                    new_nid_path_map[kinote.n.id] = new_note_relpath

        if len(new_nid_path_map) > 0:
            # Consider moving this into a function.
            msg = "Generated new nid(s).\n\n"
            for new_nid, path in new_nid_path_map.items():
                msg += f"Wrote note '{new_nid}' in file {path}\n"
            kirepo.repo.git.add(all=True)
            _ = kirepo.repo.index.commit(msg)

        p = subprocess.run(
            ["git", "rev-parse", "-q", "--verify", "refs/stash"],
            check=False,
            capture_output=True,
        )
        _new_stash_ref = p.stdout.decode()

        # Display warnings.
        for line in log:
            logger.warning(line)

    # Backup collection file and overwrite collection.
    backup(kirepo.col_file)
    new_col_file = fcopyfile(new_col_file, kirepo.col_file, fparent(kirepo.col_file))
    echo(f"Overwrote '{kirepo.col_file}'")

    # Append to hashes file.
    new_md5sum = md5(new_col_file)
    append_md5sum(cwd / KI, new_col_file, new_md5sum, silent=True)

    # Unlock Anki SQLite DB.
    unlock(con)

    profiler.stop()
    profiler.output_html()
