#!/usr/bin/env python3
"""Type-safe, non Anki-specific functions."""

# pylint: disable=import-self

import os
import re
import shutil
import hashlib
import tempfile
import unicodedata
from pathlib import Path

import git
from result import Result, Err, Ok

from beartype import beartype
from beartype.typing import (
    Set,
    List,
    Dict,
    Optional,
    Union,
    Generator,
)

import ki.functional as F
from ki.monadic import monadic
from ki.types import (
    ExtantFile,
    ExtantDir,
    EmptyDir,
    NoPath,
    Singleton,
    ExtantStrangePath,
    Leaves,
    KiRepoRef,
    RepoRef,
    PathCreationCollisionError,
)

FS_ROOT = Path("/")

# Emoji regex character classes.
EMOJIS = "\U0001F600-\U0001F64F"
PICTOGRAPHS = "\U0001F300-\U0001F5FF"
TRANSPORTS = "\U0001F680-\U0001F6FF"
FLAGS = "\U0001F1E0-\U0001F1FF"

# Regex to filter out bad stuff from filenames.
SLUG_REGEX = re.compile(r"[^\w\s\-" + EMOJIS + PICTOGRAPHS + TRANSPORTS + FLAGS + "]")


# DANGER


@beartype
def rmtree(target: ExtantDir) -> NoPath:
    """Call shutil.rmtree()."""
    shutil.rmtree(target)
    return NoPath(target)


@beartype
def copytree(source: ExtantDir, target: NoPath) -> ExtantDir:
    """Call shutil.copytree()."""
    shutil.copytree(source, target)
    return ExtantDir(target.resolve())


@beartype
def cwd() -> ExtantDir:
    """Call Path.cwd()."""
    return ExtantDir(Path.cwd().resolve())


@beartype
def test(
    path: Path,
) -> Union[ExtantFile, ExtantDir, EmptyDir, ExtantStrangePath, NoPath]:
    """Test whether `path` is a file, a directory, or something else."""
    path = path.resolve()
    if path.is_file():
        return ExtantFile(path)
    if path.is_dir():
        if is_empty(ExtantDir(path)):
            return EmptyDir(path)
        return ExtantDir(path)
    if path.exists():
        return ExtantStrangePath(path)
    return NoPath(path)


@beartype
def touch(directory: ExtantDir, name: str) -> ExtantFile:
    """Touch a file."""
    path = directory / singleton(name)
    path.touch()
    return ExtantFile(path.resolve())


@beartype
def mkdir(path: NoPath) -> EmptyDir:
    """Make a directory (with parents)."""
    path.mkdir(parents=True)
    return EmptyDir(path)


@beartype
def mksubdir(directory: EmptyDir, suffix: Path) -> EmptyDir:
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
    return EmptyDir(subdir.resolve())


@beartype
def force_mkdir(path: Path) -> ExtantDir:
    """Make a directory (with parents, ok if it already exists)."""
    path.mkdir(parents=True, exist_ok=True)
    return ExtantDir(path.resolve())


@beartype
def chdir(directory: ExtantDir) -> ExtantDir:
    """Changes working directory and returns old cwd."""
    old: ExtantDir = F.cwd()
    os.chdir(directory)
    return old


@beartype
def parent(path: Union[ExtantFile, ExtantDir]) -> ExtantDir:
    """
    Get the parent of a path that exists.  If the path points to the filesystem
    root, we return itself.
    """
    if path.resolve() == FS_ROOT:
        return ExtantDir(FS_ROOT.resolve())
    return ExtantDir(path.parent)


@beartype
def mkdtemp() -> EmptyDir:
    """Make a temporary directory (in /tmp)."""
    return EmptyDir(tempfile.mkdtemp()).resolve()


@beartype
def copyfile(source: ExtantFile, target_root: ExtantDir, name: str) -> ExtantFile:
    """Force copy a file (potentially overwrites the target path)."""
    name = singleton(name)
    target = target_root / name
    shutil.copyfile(source, target)
    return ExtantFile(target.resolve())


@beartype
def rglob(root: ExtantDir, pattern: str) -> List[ExtantFile]:
    """Call root.rglob() and returns only files."""
    files = filter(
        lambda p: isinstance(p, ExtantFile), map(F.test, root.rglob(pattern))
    )
    return list(files)


@beartype
def is_empty(directory: ExtantDir) -> bool:
    """Check if directory is empty, quickly."""
    return not next(os.scandir(directory), None)


@beartype
def working_dir(repo: git.Repo) -> ExtantDir:
    """Get working directory of a repo."""
    return ExtantDir(repo.working_dir).resolve()


@beartype
def git_dir(repo: git.Repo) -> ExtantDir:
    """Get git directory of a repo."""
    return ExtantDir(repo.git_dir).resolve()


@beartype
def singleton(name: str) -> Singleton:
    """Removes all forward slashes and returns a Singleton pathlib.Path."""
    return Singleton(name.replace("/", ""))


@beartype
def md5(path: ExtantFile) -> str:
    """Compute md5sum of file at `path`."""
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


@beartype
def ref_exists(repo: git.Repo, ref: str) -> bool:
    """Check if git commit reference exists in repository."""
    try:
        repo.git.rev_parse("--verify", ref)
    except git.GitCommandError:
        return False
    return True


@beartype
def get_batches(lst: List[ExtantFile], n: int) -> Generator[ExtantFile, None, None]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


@beartype
def slugify(value: str, allow_unicode: bool = False) -> str:
    """
    Taken from [1]. Convert to ASCII if 'allow_unicode' is False. Convert
    spaces or repeated dashes to single dashes. Remove characters that aren't
    alphanumerics, underscores, or hyphens. Convert to lowercase. Also strip
    leading and trailing whitespace, dashes, and underscores.

    [1] https://github.com/django/django/blob/master/django/utils/text.py
    """
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )

    value = re.sub(SLUG_REGEX, "", value.lower())
    result: str = re.sub(r"[-\s]+", "-", value).strip("-_")

    return result


# Only monadic function in here.


@monadic
@beartype
def fmkleaves(
    root: EmptyDir,
    *,
    files: Optional[Dict[str, str]] = None,
    dirs: Optional[Dict[str, str]] = None,
) -> Result[Leaves, Exception]:
    """Safely populate an empty directory with empty files and empty subdirectories."""
    # Check that there are no collisions, returning an appropriate error if one
    # is found.
    leaves: Set[str] = set()
    if files is not None:
        for key, token in files.items():
            if str(token) in leaves:
                return Err(PathCreationCollisionError(root, str(token)))
            leaves.add(str(token))
    if dirs is not None:
        for key, token in dirs.items():
            # We lie to the `F.mksubdir` call and tell it the root is empty
            # on every iteration.
            if str(token) in leaves:
                return Err(PathCreationCollisionError(root, str(token)))
            leaves.add(str(token))

    # Actually populate the directory.
    new_files: Dict[str, ExtantFile] = {}
    new_dirs: Dict[str, EmptyDir] = {}
    if files is not None:
        for key, token in files.items():
            new_files[key] = F.touch(root, token)
    if dirs is not None:
        for key, token in dirs.items():
            new_dirs[key] = F.mksubdir(EmptyDir(root), singleton(token))

    return Ok(Leaves(root, new_files, new_dirs))


@beartype
def kirepo_ref_to_repo_ref(kirepo_ref: KiRepoRef) -> RepoRef:
    """Convert a ki repository commit ref to a git repository commit ref."""
    return RepoRef(kirepo_ref.kirepo.repo, kirepo_ref.sha)
