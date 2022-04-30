#!/usr/bin/env python3
"""Types for ki."""
import sqlite3
import textwrap
from enum import Enum
from pathlib import Path
from dataclasses import dataclass

import git
import prettyprinter as pp
from anki.collection import Note

from beartype import beartype
from beartype.typing import List, Dict, Any, Optional

from ki.transformer import FlatNote

NotetypeDict = Dict[str, Any]
MODELS_FILE = "models.json"
HINT = (
    "hint: Updates were rejected because the tip of your current branch is behind\n"
    + "hint: the Anki remote collection. Integrate the remote changes (e.g.\n"
    + "hint: 'ki pull ...') before pushing again."
)
ERROR_MESSAGE_WIDTH = 69
DATABASE_LOCKED_MSG = "database is locked"


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


@beartype
@dataclass(frozen=True)
class WrittenNoteFile:
    """Store a written file and its primary deck id."""

    did: int
    file: ExtantFile


@beartype
def errwrap(msg: str) -> str:
    out: str = textwrap.fill(textwrap.dedent(msg), width=ERROR_MESSAGE_WIDTH)
    out = out.lstrip()
    out = out.rstrip()
    return out


# EXCEPTIONS


class MissingFileError(FileNotFoundError):
    @beartype
    def __init__(self, path: Path, info: str = ""):
        msg = f"File not found: '{path}'{info.rstrip()}"
        super().__init__(textwrap.fill(textwrap.dedent(msg), width=ERROR_MESSAGE_WIDTH))


class MissingDirectoryError(Exception):
    @beartype
    def __init__(self, path: Path, info: str = ""):
        msg = f"Directory not found: '{path}'{info.rstrip()}"
        super().__init__(textwrap.fill(textwrap.dedent(msg), width=ERROR_MESSAGE_WIDTH))


class ExpectedFileButGotDirectoryError(FileNotFoundError):
    @beartype
    def __init__(self, path: Path, info: str = ""):
        msg = "A file was expected at this location, but got a directory: "
        msg += f"'{path}'{info.rstrip()}"
        super().__init__(textwrap.fill(textwrap.dedent(msg), width=ERROR_MESSAGE_WIDTH))


class ExpectedDirectoryButGotFileError(Exception):
    @beartype
    def __init__(self, path: Path, info: str = ""):
        msg = "A directory was expected at this location, but got a file: "
        msg += f"'{path}'{info.rstrip()}"
        super().__init__(textwrap.fill(textwrap.dedent(msg), width=ERROR_MESSAGE_WIDTH))


class ExpectedEmptyDirectoryButGotNonEmptyDirectoryError(Exception):
    @beartype
    def __init__(self, path: Path, info: str = ""):
        msg = "An empty directory was expected at this location, but it is nonempty: "
        msg += f"'{path}'{info.rstrip()}"
        super().__init__(textwrap.fill(textwrap.dedent(msg), width=ERROR_MESSAGE_WIDTH))


class StrangeExtantPathError(Exception):
    @beartype
    def __init__(self, path: Path, info: str = ""):
        msg = "A normal file or directory was expected, but got a weird pseudofile "
        msg += "(e.g. a socket, or a device): "
        msg += f"'{path}'{info.rstrip()}"
        super().__init__(textwrap.fill(textwrap.dedent(msg), width=ERROR_MESSAGE_WIDTH))


class ExpectedNonexistentPathError(FileExistsError):
    @beartype
    def __init__(self, path: Path, info: str = ""):
        msg = f"""
        Expected this path not to exist, but it does: '{path}'{info.rstrip()}
        """
        super().__init__(textwrap.fill(textwrap.dedent(msg), width=ERROR_MESSAGE_WIDTH))


class NotKiRepoError(Exception):
    @beartype
    def __init__(self):
        msg = "fatal: not a ki repository (or any parent up to mount point /)\n"
        msg += "Stopping at filesystem boundary."
        super().__init__(textwrap.fill(textwrap.dedent(msg), width=ERROR_MESSAGE_WIDTH))


class UpdatesRejectedError(Exception):
    @beartype
    def __init__(self, col_file: ExtantFile):
        msg = f"Failed to push some refs to '{col_file}'\n{HINT}"
        super().__init__(textwrap.fill(textwrap.dedent(msg), width=ERROR_MESSAGE_WIDTH))


class TargetExistsError(Exception):
    @beartype
    def __init__(self, target: Path):
        msg = f"fatal: destination path '{target}' already exists and is "
        msg += "not an empty directory."
        super().__init__(textwrap.fill(textwrap.dedent(msg), width=ERROR_MESSAGE_WIDTH))


class GitRefNotFoundError(Exception):
    @beartype
    def __init__(self, repo: git.Repo, sha: str):
        msg = f"Repo at '{repo.working_dir}' doesn't contain ref '{sha}'"
        super().__init__(textwrap.fill(textwrap.dedent(msg), width=ERROR_MESSAGE_WIDTH))


class GitHeadRefNotFoundError(Exception):
    @beartype
    def __init__(self, repo: git.Repo, error: Exception):
        msg = f"""
        ValueError raised while trying to get ref 'HEAD' from repo at
        '{repo.working_dir}': '{error}'. This may have occurred because there
        are no commits in the current repository. However, this should never be
        the case, because ki repositories must be instantiated with a 'ki clone
        <collection>' command, and this command creates an initial commit.
        """
        super().__init__(textwrap.fill(textwrap.dedent(msg), width=ERROR_MESSAGE_WIDTH))


class CollectionChecksumError(Exception):
    @beartype
    def __init__(self, col_file: ExtantFile):
        msg = f"Checksum mismatch on {col_file}. Was file changed?"
        super().__init__(textwrap.fill(textwrap.dedent(msg), width=ERROR_MESSAGE_WIDTH))


class MissingNotetypeError(Exception):
    @beartype
    def __init__(self, model: str):
        msg = f"""
        Notetype '{model}' doesn't exist. Create it in Anki before adding notes
        via ki. This may be caused by a corrupted '{MODELS_FILE}' file. The
        models file must contain definitions for all models that appear in all
        note files.
        """
        super().__init__(errwrap(msg))


# TODO: Should we also print which field ordinals *are* valid?
class MissingFieldOrdinalError(Exception):
    @beartype
    def __init__(self, ord: int, model: str):
        msg = f"Field with ordinal {ord} missing from notetype '{model}'."
        super().__init__(textwrap.fill(textwrap.dedent(msg), width=ERROR_MESSAGE_WIDTH))


class MissingNoteIdError(Exception):
    @beartype
    def __init__(self, nid: int):
        msg = f"Failed to locate note with nid '{nid}' in Anki database."
        super().__init__(textwrap.fill(textwrap.dedent(msg), width=ERROR_MESSAGE_WIDTH))


class NotetypeMismatchError(Exception):
    @beartype
    def __init__(self, flatnote: FlatNote, new_notetype: Notetype):
        msg = f"Notetype '{flatnote.model}' "
        msg += f"specified in FlatNote with nid '{flatnote.nid}' "
        msg += f"does not match passed notetype '{new_notetype}'. "
        msg += "This should NEVER happen, "
        msg += "and indicates a bug in the caller to 'update_note()'."
        super().__init__(textwrap.fill(textwrap.dedent(msg), width=ERROR_MESSAGE_WIDTH))


class NotetypeKeyError(Exception):
    @beartype
    def __init__(self, key: str, name: str):
        msg = f"""
        Expected key {key} not found in notetype '{name}' parsed from a
        '{MODELS_FILE}' file in the current repository (may be contained in a
        subdirectory).
        """
        super().__init__(textwrap.fill(textwrap.dedent(msg), width=ERROR_MESSAGE_WIDTH))


class NoteFieldKeyError(Exception):
    @beartype
    def __init__(self, key: str, nid: int):
        msg = f"""
        Expected field {key} not found in note '{nid}'. This should *never*
        happen, and indicates a serious failure, since we only ever index
        `anki.notes.Note` objects on names pulled from their own notetype
        dictionary.
        """
        super().__init__(textwrap.fill(textwrap.dedent(msg), width=ERROR_MESSAGE_WIDTH))


class UnnamedNotetypeError(Exception):
    @beartype
    def __init__(self, nt: NotetypeDict):
        msg = f"""
        Failed to find 'name' field for a notetype while parsing
        a '{MODELS_FILE}' file in the current repository (may be
        contained in a subdirectory):
        """
        msg = textwrap.fill(textwrap.dedent(msg), width=ERROR_MESSAGE_WIDTH)
        super().__init__(msg + "\n" + pp.pformat(nt))


class SQLiteLockError(Exception):
    @beartype
    def __init__(self, col_file: ExtantFile, err: sqlite3.DatabaseError):
        if str(err) == DATABASE_LOCKED_MSG:
            header = f"fatal: {DATABASE_LOCKED_MSG} (Anki must not be running)."
            super().__init__(header)
            return
        header = "Unexpected SQLite3 error while attempting to acquire lock on file: "
        header += f"'{col_file}':"
        msg = f"""
        A 'sqlite3.DatabaseError' was raised with error message: '{str(err)}'.
        This may indicate that either the database file at the location
        specified above is corrupted, or the config file at '.ki/config' is
        pointing to the wrong location. (The latter may occur in the unlikely
        event that the collection file in the Anki data directory has been
        accidentally overwritten.)
        """
        super().__init__(
            header
            + "\n"
            + textwrap.fill(textwrap.dedent(msg), width=ERROR_MESSAGE_WIDTH)
        )


class PathCreationCollisionError(Exception):
    @beartype
    def __init__(self, root: ExtantDir, token: str):
        header = "Collision in children names for population of empty directory "
        header += f"'{root}':"
        msg = f"""
        Attempted to create two children (files or directories) of the empty
        directory specified above with the same name ('{token}'). This should
        *never* happen, as population of empty directories only happens in
        calls to 'F.fmkleaves()', and this is only used to populate the '.ki/'
        directory, whose contents all ought to be distinct.
        """
        super().__init__(
            header
            + "\n"
            + textwrap.fill(textwrap.dedent(msg), width=ERROR_MESSAGE_WIDTH)
        )


class MissingMediaDirectoryError(Exception):
    @beartype
    def __init__(self, col_path: str, media_dir: Path):
        top = f"Missing or bad Anki collection media directory '{media_dir}' "
        top += f"while processing collection '{col_path}':"
        msg = """
        This should *never* happen, as Anki generates a media directory at the
        relevant location whenever a `Collection` object is instantiated.  It
        is possible that the collection's containing directory was manually
        tampered with, or an old version of Anki incompatible with ki is
        installed.
        """
        super().__init__(
            top + "\n" + textwrap.fill(textwrap.dedent(msg), width=ERROR_MESSAGE_WIDTH)
        )


class AnkiAlreadyOpenError(Exception):
    @beartype
    def __init__(self, msg: str):
        super().__init__(f"fatal: {msg}")


# WARNINGS


# TODO: Make this warning more descriptive. Should given the note id, the path,
# the field(s) which are missing, and the model.
class NoteFieldValidationWarning(Warning):
    pass


class UnhealthyNoteWarning(Warning):
    pass


class UnPushedPathWarning(Warning):
    @beartype
    def __init__(self, path: Path, pattern: str):
        msg = f"Warning: ignoring '{path}' matching ignore pattern '{pattern}'"
        super().__init__(msg)


class NotAnkiNoteWarning(Warning):
    @beartype
    def __init__(self, file: ExtantFile):
        msg = f"Warning: not Anki note '{file}'"
        super().__init__(msg)


class DeletedFileNotFoundWarning(Warning):
    @beartype
    def __init__(self, path: Path):
        top = f"Deleted file not found in source commit: '{path}'"
        msg = """
        Unexpected: this may indicate a bug in ki. The source commit is what we
        are diffing against, and so we expect all files whose change type is
        'DELETED' to appear in a checkout of that reference. However, we return
        a 'Warning' instead of an 'Exception' in order to avoid interrupting
        the execution of a 'push()' call where it is not strictly necessary.
        """
        super().__init__(
            top + "\n" + textwrap.fill(textwrap.dedent(msg), width=ERROR_MESSAGE_WIDTH)
        )


class DiffTargetFileNotFoundWarning(Warning):
    @beartype
    def __init__(self, path: Path):
        top = f"Diff target file not found: '{path}'"
        msg = """
        Unexpected: this may indicate a bug in ki. The caller prevents this
        warning from being instantiated unless the git change type is one of
        'ADDED', 'MODIFIED', or 'RENAMED'. In all cases, the file being diffed
        should be extant in the target commit of the repository.  However, we
        return a 'Warning' instead of an 'Exception' in order to avoid
        interrupting the execution of a 'push()' call where it is not strictly
        necessary.
        """
        super().__init__(
            top + "\n" + textwrap.fill(textwrap.dedent(msg), width=ERROR_MESSAGE_WIDTH)
        )


class MissingMediaFileWarning(Warning):
    @beartype
    def __init__(self, col_path: str, media_file: Path):
        top = f"Missing or bad media file '{media_file}' "
        top += f"while processing collection '{col_path}':"
        msg = f"""
        Expected an extant file at the location specified above, but got a
        '{type(media_file)}'. This may indicate a corrupted Anki collection, as
        all media filenames present in note fields should correspond to extant
        files within the media directory (usually called 'collection.media/'
        within the relevant Anki user profile directory).
        """
        super().__init__(
            top + "\n" + textwrap.fill(textwrap.dedent(msg), width=ERROR_MESSAGE_WIDTH)
        )
