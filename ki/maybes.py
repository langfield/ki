#!/usr/bin/env python3
"""Factory functions for safely handling errors in type construction."""

# pylint: disable=invalid-name, missing-class-docstring, broad-except
# pylint: disable=too-many-return-statements, too-many-lines, import-self

import re
import traceback
import configparser
from pathlib import Path
from functools import partial

import git
from loguru import logger
from beartype import beartype
from beartype.typing import Union, Dict, Any, Optional, List

import anki
from anki.decks import DeckTreeNode
from anki.errors import NotFoundError
from anki.collection import Collection

import ki.maybes as M
import ki.functional as F
from ki.types import (
    MODELS_FILE,
    File,
    Dir,
    EmptyDir,
    NoPath,
    NoFile,
    Link,
    PseudoFile,
    WindowsLink,
    KiRepo,
    KiRev,
    Rev,
    Template,
    Field,
    ColNote,
    Deck,
    PlannedLink,
    Notetype,
    NotetypeKeyError,
    UnnamedNotetypeError,
    MissingFieldOrdinalError,
    MissingNoteIdError,
    NoteFieldKeyError,
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
    MaximumWindowsLinkChainingDepthExceededError,
    GitFileModeParseError,
)

KI = ".ki"
GIT = F.GIT
MEDIA = "_media"
GITIGNORE_FILE = ".gitignore"
GITMODULES_FILE = F.GITMODULES_FILE

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
def xfile(path: Path, info: str = "") -> File:
    """
    Attempt to instantiate an File.
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
    return File(path)


@beartype
def xdir(path: Path, info: str = "") -> Dir:
    """
    Attempt to instantiate an Dir.
    """
    # Resolve path.
    path = path.resolve()

    # Check that path exists and is a directory.
    if not path.exists():
        raise MissingDirectoryError(path, info)
    if path.is_dir():
        return Dir(path)
    if path.is_file():
        raise ExpectedDirectoryButGotFileError(path, info)
    raise StrangeExtantPathError(path, info)


@beartype
def emptydir(path: Path) -> Dir:
    """
    Attempt to instantiate an Dir.
    """
    # Check if it's an extant directory.
    directory: Dir = M.xdir(path)
    if F.is_empty(directory):
        return EmptyDir(Path(directory).resolve())
    raise ExpectedEmptyDirectoryButGotNonEmptyDirectoryError(directory)


@beartype
def repo(root: Dir) -> git.Repo:
    """Read a git repo safely."""
    try:
        repository = git.Repo(root)
    except git.InvalidGitRepositoryError as err:
        # TODO: Make this error more descriptive. It currently sucks. A test
        # should be written for 'M.kirepo()' in which we return this error.
        raise err
    return repository


@beartype
def kirepo(cwd: Dir) -> KiRepo:
    """Get the containing ki repository of `path`."""
    current = cwd

    while not F.is_root(current):
        kid = F.chk(current / KI)
        if isinstance(kid, Dir):
            break
        current = F.parent(current)

    if F.is_root(current):
        raise NotKiRepoError()

    # Root directory and ki directory of repo now guaranteed to exist.
    root = current
    repository: git.Repo = M.repo(root)

    # Check that relevant files in .ki/ subdirectory exist.
    backups_dir = M.xdir(kid / BACKUPS_DIR, info=BACKUPS_DIR_INFO)
    config_file = M.xfile(kid / CONFIG_FILE, info=CONFIG_FILE_INFO)
    hashes_file = M.xfile(kid / HASHES_FILE, info=HASHES_FILE_INFO)
    models_file = M.xfile(root / MODELS_FILE, info=MODELS_FILE_INFO)
    lca_file = M.xfile(kid / LAST_PUSH_FILE, info=LAST_PUSH_FILE_INFO)

    # Check that collection file exists.
    config = configparser.ConfigParser()
    config.read(config_file)
    col_file = Path(config[REMOTE_CONFIG_SECTION][COLLECTION_FILE_PATH_CONFIG_FIELD])
    col_file = M.xfile(col_file, info=COL_FILE_INFO)

    return KiRepo(
        repository,
        root,
        kid,
        col_file,
        backups_dir,
        config_file,
        hashes_file,
        models_file,
        lca_file,
    )


@beartype
def rev(repository: git.Repo, sha: str) -> Rev:
    """Validate a commit SHA against a repository and return a `Rev`."""
    if not F.rev_exists(repository, sha):
        raise GitRefNotFoundError(repository, sha)
    return Rev(repository, sha)


@beartype
def head(repository: git.Repo) -> Rev:
    """Return a `Rev` for HEAD of current branch."""
    # GitPython raises a ValueError when references don't exist.
    try:
        rev = Rev(repository, repository.head.commit.hexsha)
    except ValueError as err:
        raise GitHeadRefNotFoundError(repository, err) from err
    return rev


@beartype
def head_ki(kirepository: KiRepo) -> KiRev:
    """Return a `KiRev` for HEAD of current branch."""
    # GitPython raises a ValueError when references don't exist.
    try:
        rev = KiRev(kirepository, kirepository.repo.head.commit.hexsha)
    except ValueError as err:
        raise GitHeadRefNotFoundError(kirepository.repo, err) from err
    return rev


@beartype
def collection(col_file: File) -> Collection:
    """Open a collection or raise a pretty exception."""
    cwd: Dir = F.cwd()
    try:
        col = Collection(col_file)
    except anki.errors.DBError as err:
        raise AnkiAlreadyOpenError(str(err)) from err
    finally:
        F.chdir(cwd)
    return col


@beartype
def linktarget(orig: File) -> File:
    """Follow a windows symlink inside a git repo, or return regular file unchanged."""
    # Check file mode, and follow symlink if applicable.
    depth = 0
    file = orig
    while M.filemode(file) == 120000:
        target: str = file.read_text(encoding="UTF-8")
        parent: Dir = F.parent(file)
        file = M.xfile(parent / target)
        depth += 1
        if depth > 999:
            raise MaximumWindowsLinkChainingDepthExceededError(orig, depth)
    return file


@beartype
def hardlink(link: Union[File, Link]) -> File:
    """Replace a possibly windows symlink with its target."""
    # Treat true POSIX symlink case.
    if isinstance(link, Link):
        tgt = F.chk(link.resolve())
        return F.copyfile(tgt, link)

    # Treat windows symlink case.
    tgt = M.linktarget(link)
    if tgt != link:
        link: NoFile = F.unlink(link)
        link: File = F.copyfile(tgt, link)
    return link


@beartype
def filemode(file: Union[File, Dir, PseudoFile, Link, WindowsLink]) -> int:
    """Get git file mode."""
    try:
        # We must search from file upwards in case inside submodule.
        root_repo = git.Repo(file, search_parent_directories=True)
        out = root_repo.git.ls_files(["-s", str(file)])

        # Treat case where file is untracked.
        if out == "":
            return -1

        mode: int = int(out.split()[0])
    except Exception as err:
        raise GitFileModeParseError(file, out) from err
    return mode


@beartype
def template(t: Dict[str, Any]) -> Template:
    """Construct a template."""
    name, qfmt, afmt, ord = t["name"], t["qfmt"], t["afmt"], t["ord"]
    return Template(name=name, qfmt=qfmt, afmt=afmt, ord=ord)


@beartype
def field(fld: Dict[str, Any]) -> Field:
    """Construct a field."""
    return Field(name=fld["name"], ord=fld["ord"])


@beartype
def notetype(nt: Dict[str, Any]) -> Notetype:
    """
    Convert an Anki NotetypeDict into a Notetype dataclass.

    Anki returns objects of type `NotetypeDict` (see pylib/anki/models.py)
    when you call a method like `col.models.all()`. This is a dictionary
    mapping strings to various stuff, and we read all its data into a python
    dataclass here so that we can access it safely. Since we don't expect Anki
    to ever give us 'invalid' notetypes (since we define 'valid' as being
    processable by Anki), we return an exception if the parse fails.

    Note on naming convention: Below, abbreviated variable names represent
    dicts coming from Anki, like `nt: NotetypeDict` or `fld: FieldDict`.
    Full words like `field: Field` represent ki dataclasses. The parameters
    of the dataclasses, however, use abbreviations for consistency with Anki
    map keys.
    """
    # If we can't even read the name of the notetype, then we can't print out a
    # nice error message in the event of a `KeyError`. So we have to print out
    # a different error message saying that the notetype doesn't have a name
    # field.
    try:
        nt["name"]
    except KeyError as err:
        raise UnnamedNotetypeError(nt) from err
    try:
        fields: Dict[int, Field] = {fld["ord"]: M.field(fld) for fld in nt["flds"]}
        if nt["sortf"] not in fields:
            raise MissingFieldOrdinalError(ord=nt["sortf"], model=nt["name"])
        return Notetype(
            id=nt["id"],
            name=nt["name"],
            type=nt["type"],
            flds=list(fields.values()),
            tmpls=list(map(M.template, nt["tmpls"])),
            sortf=fields[nt["sortf"]],
            dict=nt,
        )
    except KeyError as err:
        raise NotetypeKeyError(key=str(err), name=str(nt["name"])) from err


@beartype
def colnote(col: Collection, nid: int) -> ColNote:
    """Get a dataclass representation of an Anki note."""
    try:
        note = col.get_note(nid)
    except NotFoundError as err:
        raise MissingNoteIdError(nid) from err
    notetype: Notetype = M.notetype(note.note_type())

    # Get sort field content. See comment where we subscript in the same way in
    # `push_note()`.
    try:
        sortf_text: str = note[notetype.sortf.name]
    except KeyError as err:
        raise NoteFieldKeyError(str(err), nid) from err

    # TODO: Remove implicit assumption that all cards are in the same deck, and
    # work with cards instead of notes.
    try:
        deck = col.decks.name(note.cards()[0].did)
    except IndexError as err:
        logger.error(f"{note.cards() = }")
        logger.error(f"{note.guid = }")
        logger.error(f"{note.id = }")
        raise err
    colnote = ColNote(
        n=note,
        new=False,
        deck=deck,
        title="",
        markdown=False,
        notetype=notetype,
        sortf_text=sortf_text,
    )
    return colnote


@beartype
def deckd(deck_name: str, targetdir: Dir) -> Dir:
    """
    Construct path to deck directory and create it, allowing the case in which
    the directory already exists because we already created one of its
    children, in which case this function is a no-op.
    """
    # Strip leading periods so we don't get hidden folders.
    components = deck_name.split("::")
    components = [re.sub(r"^\.", r"", comp) for comp in components]
    components = [re.sub(r"/", r"-", comp) for comp in components]
    deck_path = Path(targetdir, *components)
    return F.force_mkdir(deck_path)


@beartype
def tree(col: Collection, targetd: Dir, root: DeckTreeNode) -> Deck:
    """Get the deck directory and did for a decknode."""
    did = root.deck_id
    name = col.decks.name(did)
    deckd = M.deckd(name, targetd)
    mediad: Dir = F.force_mkdir(deckd / MEDIA)
    children: List[Deck] = list(map(partial(M.tree, col, targetd), root.children))
    return Deck(
        did=did,
        node=root,
        deckd=deckd,
        mediad=mediad,
        fullname=name,
        children=children,
    )


@beartype
def winlink(targetd: Dir, l: PlannedLink) -> Optional[WindowsLink]:
    """Create the symlink `l` and return any windows links."""
    distance = len(l.link.parent.relative_to(targetd).parts)
    target: Path = Path("../" * distance) / l.tgt.relative_to(targetd)
    try:
        link: Union[Link, WindowsLink] = F.symlink(l.link, target)
    except OSError as _:
        trace = traceback.format_exc(limit=3)
        logger.warning(f"Failed to create symlink '{l.link}' -> '{target}'\n{trace}")
    return link if isinstance(link, WindowsLink) else None
