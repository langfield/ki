#!/usr/bin/env python3
"""Monadic factory functions for safely handling errors in type construction."""

# pylint: disable=invalid-name, missing-class-docstring, broad-except
# pylint: disable=too-many-return-statements, too-many-lines

import re
import configparser
from pathlib import Path

import git
from result import Result, Err, Ok, OkErr

from anki import notetypes_pb2

from beartype import beartype
from beartype.typing import (
    List,
    Dict,
    Any
)

from ki.safe import safe
from ki.types import (
    MODELS_FILE,
    ExtantFile,
    ExtantDir,
    EmptyDir,
    NoPath,
    KiRepo,
    KiRepoRef,
    RepoRef,
    MissingFileError,
    MissingDirectoryError,
    ExpectedFileButGotDirectoryError,
    ExpectedDirectoryButGotFileError,
    ExpectedEmptyDirectoryButGotNonEmptyDirectoryError,
    StrangeExtantPathError,
    NotKiRepoError,
    GitRefNotFoundError,
)

FS_ROOT = Path("/")

KI = ".ki"
GIT = ".git"
GITIGNORE_FILE = ".gitignore"
GITMODULES_FILE = ".gitmodules"

CONFIG_FILE = "config"
HASHES_FILE = "hashes"
BACKUPS_DIR = "backups"
LAST_PUSH_FILE = "last_push"
NO_SM_DIR = "no_submodules_tree"

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


# MAYBES


@safe
@beartype
def M_nopath(path: Path) -> Result[NoPath, Exception]:
    """
    Maybe convert a path to a NoPath, i.e. a path that did not exist at
    resolve-time, which is when this function was called.
    """
    path = path.resolve()
    if path.exists():
        return Err(FileExistsError(str(path)))
    return Ok(NoPath(path))


@safe
@beartype
def M_xfile(path: Path, info: str = "") -> Result[ExtantFile, Exception]:
    """
    Attempt to instantiate an ExtantFile.
    """
    # Resolve path.
    path = path.resolve()

    # Check that path exists and is a file.
    if not path.exists():
        return Err(MissingFileError(path, info))
    if path.is_dir():
        return Err(ExpectedFileButGotDirectoryError(path, info))
    if not path.is_file():
        return Err(StrangeExtantPathError(path, info))

    # Must be an extant file.
    return Ok(ExtantFile(path))


@safe
@beartype
def M_xdir(path: Path, info: str = "") -> Result[ExtantDir, Exception]:
    """
    Attempt to instantiate an ExtantDir.
    """
    # Resolve path.
    path = path.resolve()

    # Check that path exists and is a directory.
    if not path.exists():
        return Err(MissingDirectoryError(path, info))
    if path.is_dir():
        return Ok(ExtantDir(path))
    if path.is_file():
        return Err(ExpectedDirectoryButGotFileError(path, info))
    return Err(StrangeExtantPathError(path, info))


@safe
@beartype
def M_emptydir(path: Path) -> Result[ExtantDir, Exception]:
    """
    Attempt to instantiate an ExtantDir.
    """
    # Check if it's an extant directory.
    res: OkErr = M_xdir(path)

    # Return the Err if not.
    if res.is_err():
        return res

    # Unwrap the value otherwise.
    directory: ExtantDir = res.unwrap()
    if is_empty(directory):
        return Ok(EmptyDir(Path(directory).resolve()))
    return ExpectedEmptyDirectoryButGotNonEmptyDirectoryError(str(directory))


@safe
@beartype
def M_repo(root: ExtantDir) -> Result[git.Repo, Exception]:
    """Read a git repo safely."""
    try:
        repo = git.Repo(root)
    except git.InvalidGitRepositoryError as err:
        return Err(err)
    return Ok(repo)


@safe
@beartype
def M_kirepo(cwd: ExtantDir) -> Result[KiRepo, Exception]:
    """Get the containing ki repository of `path`."""
    current = cwd

    # Note that `current` is an `ExtantDir` but `FS_ROOT` is a `Path`.
    # We can make this comparison because both are instances of `Path` and
    # `Path` implements the `==` operator nicely.
    while current != FS_ROOT:
        ki_dir = fftest(current / KI)
        if isinstance(ki_dir, ExtantDir):
            break
        current = ffparent(current)

    if current == FS_ROOT:
        return Err(NotKiRepoError())

    # Root directory and ki directory of repo now guaranteed to exist.
    root = current
    repo: OkErr = M_repo(root)

    # Check that relevant files in .ki/ subdirectory exist.
    backups_dir = M_xdir(ki_dir / BACKUPS_DIR, info=BACKUPS_DIR_INFO)
    config_file = M_xfile(ki_dir / CONFIG_FILE, info=CONFIG_FILE_INFO)
    hashes_file = M_xfile(ki_dir / HASHES_FILE, info=HASHES_FILE_INFO)
    models_file = M_xfile(root / MODELS_FILE, info=MODELS_FILE_INFO)
    last_push_file = M_xfile(ki_dir / LAST_PUSH_FILE, info=LAST_PUSH_FILE_INFO)

    # Load the no_submodules_tree.
    no_modules_dir = M_xdir(ki_dir / NO_SM_DIR, info=NO_MODULES_DIR_INFO)
    no_modules_repo = M_repo(no_modules_dir)

    # Check that collection file exists.
    if config_file.is_err():
        return config_file
    config_file: ExtantFile = config_file.unwrap()
    config = configparser.ConfigParser()
    config.read(config_file)
    col_file = Path(config[REMOTE_CONFIG_SECTION][COLLECTION_FILE_PATH_CONFIG_FIELD])
    col_file = M_xfile(col_file, info=COL_FILE_INFO)

    @safe
    @beartype
    def constructor(
        repo: git.Repo,
        root: ExtantDir,
        ki_dir: ExtantDir,
        col_file: ExtantFile,
        backups_dir: ExtantDir,
        config_file: ExtantFile,
        hashes_file: ExtantFile,
        models_file: ExtantFile,
        last_push_file: ExtantFile,
        no_modules_repo: git.Repo,
    ) -> Result[KiRepo, Exception]:
        return Ok(
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

    return constructor(
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


@safe
@beartype
def M_kirepo_ref(kirepo: KiRepo, sha: str) -> Result[KiRepoRef, Exception]:
    if not ref_exists(kirepo.repo, sha):
        return Err(GitRefNotFoundError(kirepo.repo, sha))
    return Ok(KiRepoRef(kirepo, sha))


@safe
@beartype
def M_repo_ref(repo: git.Repo, sha: str) -> Result[RepoRef, Exception]:
    if not ref_exists(repo, sha):
        return Err(GitRefNotFoundError(repo, sha))
    return Ok(RepoRef(repo, sha))


@safe
@beartype
def M_head_repo_ref(repo: git.Repo) -> Result[RepoRef, Exception]:
    # GitPython raises a ValueError when references don't exist.
    try:
        ref = RepoRef(repo, repo.head.commit.hexsha)
    except ValueError as err:
        return Err(err)
    return Ok(ref)


@safe
@beartype
def M_head_kirepo_ref(kirepo: KiRepo) -> Result[KiRepoRef, Exception]:
    # GitPython raises a ValueError when references don't exist.
    try:
        ref = KiRepoRef(kirepo, kirepo.repo.head.commit.hexsha)
    except ValueError as err:
        return Err(err)
    return Ok(ref)
