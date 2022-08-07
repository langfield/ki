#!/usr/bin/env python3
"""Monadic factory functions for safely handling errors in type construction."""

# pylint: disable=invalid-name, missing-class-docstring, broad-except
# pylint: disable=too-many-return-statements, too-many-lines, import-self

import configparser
from pathlib import Path

import git
from beartype import beartype

import anki
from anki.collection import Collection

import ki.maybes as M
import ki.functional as F
from ki.types import (
    MODELS_FILE,
    ExtantFile,
    ExtantDir,
    EmptyDir,
    NoPath,
    NoFile,
    KiRepo,
    KiRepoRef,
    RepoRef,
    MissingFileError,
    MissingDirectoryError,
    ExpectedFileButGotDirectoryError,
    ExpectedDirectoryButGotFileError,
    ExpectedEmptyDirectoryButGotNonEmptyDirectoryError,
    ExpectedNonexistentPathError,
    StrangeExtantPathError,
    NotKiRepoError,
    GitRefNotFoundError,
    GitHeadRefNotFoundError,
    AnkiAlreadyOpenError,
)
from ki.functional import FS_ROOT

KI = ".ki"
GIT = ".git"
GITIGNORE_FILE = ".gitignore"
GITMODULES_FILE = ".gitmodules"

CONFIG_FILE = "config"
HASHES_FILE = "hashes"
BACKUPS_DIR = "backups"
LAST_PUSH_FILE = "last_push"

REMOTE_CONFIG_SECTION = "remote"
COLLECTION_FILE_PATH_CONFIG_FIELD = "path"


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

MODELS_FILE_INFO = f"""
This is the top-level '{MODELS_FILE}' file, which contains serialized notetypes
for all notes in the current repository. Ki should always create this during
cloning. If it has been manually deleted, try reverting to an earlier commit.
Otherwise, it may indicate that the repository has become corrupted.
"""

LAST_PUSH_FILE_INFO = """
This is the '.ki/last_push' file, used internally by ki to keep diffs and
eliminate unnecessary merge conflicts during pull operations. It should never
be missing, and if it is, the repository may have become corrupted.
"""

COL_FILE_INFO = """
This is the '.anki2' database file that contains all the data for a user's
collection. This path was contained in the '.ki/config' file, indicating that
the collection this repository previously referred to has been moved or
deleted. The path can be manually fixed by editing the '.ki/config' file.
"""


# MAYBES


@beartype
def nopath(path: Path) -> NoPath:
    """
    Maybe convert a path to a NoPath, i.e. a path that did not exist at
    resolve-time, which is when this function was called.
    """
    path = path.resolve()
    if path.exists():
        raise ExpectedNonexistentPathError(path)
    return NoPath(path)


@beartype
def nofile(path: Path) -> NoFile:
    """
    Maybe convert a path to a NoPath, i.e. a path that did not exist at
    resolve-time, which is when this function was called.
    """
    path = path.resolve()
    path = M.nopath(path)
    M.xdir(path.parent)
    return NoFile(path)


@beartype
def xfile(path: Path, info: str = "") -> ExtantFile:
    """
    Attempt to instantiate an ExtantFile.
    """
    # Resolve path.
    path = path.resolve()

    # Check that path exists and is a file.
    if not path.exists():
        raise MissingFileError(path, info)
    if path.is_dir():
        raise ExpectedFileButGotDirectoryError(path, info)
    if not path.is_file():
        raise StrangeExtantPathError(path, info)

    # Must be an extant file.
    return ExtantFile(path)


@beartype
def xdir(path: Path, info: str = "") -> ExtantDir:
    """
    Attempt to instantiate an ExtantDir.
    """
    # Resolve path.
    path = path.resolve()

    # Check that path exists and is a directory.
    if not path.exists():
        raise MissingDirectoryError(path, info)
    if path.is_dir():
        return ExtantDir(path)
    if path.is_file():
        raise ExpectedDirectoryButGotFileError(path, info)
    raise StrangeExtantPathError(path, info)


@beartype
def emptydir(path: Path) -> ExtantDir:
    """
    Attempt to instantiate an ExtantDir.
    """
    # Check if it's an extant directory.
    directory: ExtantDir = M.xdir(path)
    if F.is_empty(directory):
        return EmptyDir(Path(directory).resolve())
    raise ExpectedEmptyDirectoryButGotNonEmptyDirectoryError(directory)


@beartype
def repo(root: ExtantDir) -> git.Repo:
    """Read a git repo safely."""
    try:
        repository = git.Repo(root)
    except git.InvalidGitRepositoryError as err:
        # TODO: Make this error more descriptive. It currently sucks. A test
        # should be written for 'M.kirepo()' in which we return this error.
        raise err
    return repository


@beartype
def kirepo(cwd: ExtantDir) -> KiRepo:
    """Get the containing ki repository of `path`."""
    current = cwd

    # Note that `current` is an `ExtantDir` but `FS_ROOT` is a `Path`.
    # We can make this comparison because both are instances of `Path` and
    # `Path` implements the `==` operator nicely.
    while current != FS_ROOT:
        ki_dir = F.test(current / KI)
        if isinstance(ki_dir, ExtantDir):
            break
        current = F.parent(current)

    if current == FS_ROOT:
        raise NotKiRepoError()

    # Root directory and ki directory of repo now guaranteed to exist.
    root = current
    repository: git.Repo = M.repo(root)

    # Check that relevant files in .ki/ subdirectory exist.
    backups_dir = M.xdir(ki_dir / BACKUPS_DIR, info=BACKUPS_DIR_INFO)
    config_file = M.xfile(ki_dir / CONFIG_FILE, info=CONFIG_FILE_INFO)
    hashes_file = M.xfile(ki_dir / HASHES_FILE, info=HASHES_FILE_INFO)
    models_file = M.xfile(root / MODELS_FILE, info=MODELS_FILE_INFO)
    last_push_file = M.xfile(ki_dir / LAST_PUSH_FILE, info=LAST_PUSH_FILE_INFO)

    # Check that collection file exists.
    config = configparser.ConfigParser()
    config.read(config_file)
    col_file = Path(config[REMOTE_CONFIG_SECTION][COLLECTION_FILE_PATH_CONFIG_FIELD])
    col_file = M.xfile(col_file, info=COL_FILE_INFO)

    return KiRepo(
        repository,
        root,
        ki_dir,
        col_file,
        backups_dir,
        config_file,
        hashes_file,
        models_file,
        last_push_file,
    )


@beartype
def repo_ref(repository: git.Repo, sha: str) -> RepoRef:
    """Validate a commit SHA against a repository and return a `RepoRef`."""
    if not F.ref_exists(repository, sha):
        raise GitRefNotFoundError(repository, sha)
    return RepoRef(repository, sha)


@beartype
def head_repo_ref(repository: git.Repo) -> RepoRef:
    """Return a `RepoRef` for HEAD of current branch."""
    # GitPython raises a ValueError when references don't exist.
    try:
        ref = RepoRef(repository, repository.head.commit.hexsha)
    except ValueError as err:
        raise GitHeadRefNotFoundError(repository, err) from err
    return ref


@beartype
def head_kirepo_ref(kirepository: KiRepo) -> KiRepoRef:
    """Return a `KiRepoRef` for HEAD of current branch."""
    # GitPython raises a ValueError when references don't exist.
    try:
        ref = KiRepoRef(kirepository, kirepository.repo.head.commit.hexsha)
    except ValueError as err:
        raise GitHeadRefNotFoundError(kirepository.repo, err) from err
    return ref


@beartype
def collection(col_file: ExtantFile) -> Collection:
    """Open a collection or raise a pretty exception."""
    try:
        col = Collection(col_file)
    except anki.errors.DBError as err:
        raise AnkiAlreadyOpenError(str(err)) from err
    return col
