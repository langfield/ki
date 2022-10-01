#!/usr/bin/env python3
"""Types for ki."""
import json
import sqlite3
import textwrap
import dataclasses
from enum import Enum
from pathlib import Path
from dataclasses import dataclass

import git
import whatthepatch
from anki.decks import DeckTreeNode
from anki.collection import Note, Card

from beartype import beartype
from beartype.typing import List, Dict, Any, Optional, Union

# pylint: disable=too-many-lines, missing-class-docstring, too-many-instance-attributes

NotetypeDict = Dict[str, Any]
MODELS_FILE = "models.json"
HINT = (
    "hint: Updates were rejected because the tip of your current branch is behind\n"
    + "hint: the Anki remote collection. Integrate the remote changes (e.g.\n"
    + "hint: 'ki pull ...') before pushing again."
)
ERROR_MESSAGE_WIDTH = 69
DATABASE_LOCKED_MSG = "database is locked"
DeckId = int


# TYPES


class File(type(Path())):
    """UNSAFE: Indicates that file *was* extant when it was resolved."""


class Dir(type(Path())):
    """UNSAFE: Indicates that dir *was* extant when it was resolved."""


class EmptyDir(Dir):
    """UNSAFE: Indicates that dir *was* empty (and extant) when it was resolved."""


class NoPath(type(Path())):
    """UNSAFE: Indicates that path *was not* extant when it was resolved."""


class Singleton(type(Path())):
    """UNSAFE: A path consisting of a single component (e.g. `file`, not `dir/file`)."""


class PseudoFile(type(Path())):
    """
    UNSAFE: Indicates that path was extant but weird (e.g. a device or socket)
    when it was resolved.
    """


class Link(type(Path())):
    """UNSAFE: Indicates that this path was a symlink when tested."""


class WindowsLink(type(Path())):
    """UNSAFE: A POSIX-style symlink created on Windows with mode 100644."""


class NoFile(NoPath):
    """A nonexistent file in an extant directory."""

    @property
    def parent(self):
        return Dir(super().parent)


# ENUMS


class GitChangeType(Enum):
    """Enum for git file change types."""

    ADDED = "A"
    DELETED = "D"
    RENAMED = "R"
    MODIFIED = "M"
    TYPECHANGED = "T"


class PushResult(Enum):
    """Enum for `push()` return codes."""

    NONTRIVIAL = "NONTRIVIAL"
    UP_TO_DATE = "UP_TO_DATE"


# DATACLASSES


@beartype
@dataclass(frozen=True)
class Patch:
    """Relative paths and a Diff object."""

    a: Path
    b: Path
    diff: whatthepatch.patch.diffobj


@beartype
@dataclass(frozen=True)
class DeckNote:
    """Flat (as possible) representation of a note, but with deck."""

    title: str
    guid: str
    deck: str
    model: str
    tags: List[str]
    fields: Dict[str, str]


@beartype
@dataclass(frozen=True)
class NoteMetadata:
    """The nid, mod, and mid of a note."""

    nid: int
    mod: int
    mid: int


@beartype
@dataclass(frozen=True)
class Delta:
    """
    The git delta for a single file.

    We don't instead store a root and a relative path, because we need the
    `File` object to avoid making unnecessary syscalls to check that stuff
    exists.
    """

    status: GitChangeType
    path: File
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

    # pylint: disable=invalid-name

    repo: git.Repo
    root: Dir
    ki: Dir
    col_file: File
    backups_dir: Dir
    config_file: File
    hashes_file: File
    models_file: File
    lca_file: File


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

    # pylint: disable=invalid-name

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
    markdown: bool
    notetype: Notetype
    sortf_text: str


@beartype
@dataclass(frozen=True)
class KiRev:
    """
    UNSAFE: A repo-commit pair, where `sha` is guaranteed to be an extant
    commit hash of `repo`.
    """

    kirepo: KiRepo
    sha: str


@beartype
@dataclass(frozen=True)
class Rev:
    """
    UNSAFE: A repo-commit pair, where `sha` is guaranteed to be an extant
    commit hash of `repo`.
    """

    repo: git.Repo
    sha: str


@beartype
@dataclass(frozen=True)
class CardFile:
    """A card written to disk, either as a link or a file."""

    card: Card
    link: Optional[WindowsLink]
    file: File


@beartype
@dataclass(frozen=True)
class Deck:
    did: DeckId
    node: DeckTreeNode
    deckd: Dir
    mediad: Dir
    children: List["Deck"]
    fullname: str


@beartype
@dataclass(frozen=True)
class Root:
    did: DeckId
    node: DeckTreeNode
    deckd: None
    mediad: None
    children: List[Deck]
    fullname: str


@beartype
@dataclass(frozen=True)
class PlannedLink:
    """A not-yet-created symlink path and its extant target."""

    link: NoFile
    tgt: Union[File, Link]


@beartype
@dataclass(frozen=True)
class DotKi:
    config: File
    last_push: File
    backups: EmptyDir


@beartype
@dataclass(frozen=True)
class Submodule:
    sm: git.Submodule
    sm_repo: git.Repo
    rel_root: Path
    branch: str


@beartype
@dataclass(frozen=True)
class MediaBytes:
    """A media file, its old bytes (from collection) and new bytes (from file)."""

    file: File
    old: bytes
    new: bytes


@beartype
@dataclass(frozen=True)
class AddedMedia:
    """An added media file and its (possibly changed) filename."""

    file: File
    new_name: str


@beartype
@dataclass(frozen=True)
class NoteDBRow:
    nid: int
    guid: str
    mid: int
    mod: int
    usn: int
    tags: str
    flds: str
    sfld: Union[str, int]
    csum: int
    flags: int
    data: str


@beartype
def notetype_json(notetype: Notetype) -> str:
    """Return the JSON for a notetype as a string."""
    dictionary: Dict[str, Any] = dataclasses.asdict(notetype)
    dictionary.pop("id")
    inner = dictionary["dict"]
    inner.pop("id")
    inner.pop("mod")
    dictionary["dict"] = inner
    return json.dumps(dictionary, sort_keys=True, indent=4)


@beartype
def nt_str(notetype: Notetype) -> str:
    """Display a notetype and its JSON."""
    # pylint: disable=invalid-name
    s = notetype_json(notetype)
    return f"JSON for '{notetype.id}':\n{s}"


# EXCEPTIONS


@beartype
def errwrap(msg: str) -> str:
    """Wrap an error message to a fixed width."""
    out: str = textwrap.fill(textwrap.dedent(msg), width=ERROR_MESSAGE_WIDTH)
    out = out.lstrip()
    out = out.rstrip()
    return out


class MissingFileError(FileNotFoundError):
    @beartype
    def __init__(self, path: Path, info: str = ""):
        header = f"File not found: '{path}'"
        msg = f"{info.rstrip()}"
        super().__init__(f"{header}\n\n{errwrap(msg)}")


class MissingDirectoryError(RuntimeError):
    @beartype
    def __init__(self, path: Path, info: str = ""):
        msg = f"Directory not found: '{path}'{info.rstrip()}"
        super().__init__(errwrap(msg))


class ExpectedFileButGotDirectoryError(FileNotFoundError):
    @beartype
    def __init__(self, path: Path, info: str = ""):
        msg = "A file was expected at this location, but got a directory: "
        msg += f"'{path}'{info.rstrip()}"
        super().__init__(errwrap(msg))


class ExpectedDirectoryButGotFileError(RuntimeError):
    @beartype
    def __init__(self, path: Path, info: str = ""):
        msg = "A directory was expected at this location, but got a file: "
        msg += f"'{path}'{info.rstrip()}"
        super().__init__(errwrap(msg))


class ExpectedEmptyDirectoryButGotNonEmptyDirectoryError(RuntimeError):
    @beartype
    def __init__(self, path: Path, info: str = ""):
        msg = "An empty directory was expected at this location, but it is nonempty: "
        msg += f"'{path}'{info.rstrip()}"
        super().__init__(errwrap(msg))


class StrangeExtantPathError(RuntimeError):
    @beartype
    def __init__(self, path: Path, info: str = ""):
        msg = "A normal file or directory was expected, but got a weird pseudofile "
        msg += "(e.g. a socket, or a device): "
        msg += f"'{path}'{info.rstrip()}"
        super().__init__(errwrap(msg))


class ExpectedNonexistentPathError(FileExistsError):
    @beartype
    def __init__(self, path: Path, info: str = ""):
        top = f"""
        Expected this path not to exist, but it does: '{path}'{info.rstrip()}
        """
        msg = """
        If the path is to the `.ki/` metadata directory, this error may have
        been caused by a `.gitignore` file that does not include `.ki/` (this
        metadata should not be tracked by git). Check if this pattern is
        included in the `.gitignore` file, and if it is not included, try
        adding it.
        """
        super().__init__(f"{top}\n\n{errwrap(msg)}")


class NotKiRepoError(RuntimeError):
    @beartype
    def __init__(self):
        msg = "fatal: not a ki repository (or any parent up to mount point /)\n"
        msg += "Stopping at filesystem boundary."
        super().__init__(errwrap(msg))


class UpdatesRejectedError(RuntimeError):
    @beartype
    def __init__(self, col_file: File):
        msg = f"Failed to push some commits to '{col_file}'\n{HINT}"
        super().__init__(errwrap(msg))


class TargetExistsError(RuntimeError):
    @beartype
    def __init__(self, target: Path):
        msg = f"fatal: destination path '{target}' already exists and is "
        msg += "not an empty directory."
        super().__init__(errwrap(msg))


class GitRefNotFoundError(RuntimeError):
    @beartype
    def __init__(self, repo: git.Repo, sha: str):
        msg = f"Repo at '{repo.working_dir}' doesn't contain rev '{sha}'"
        super().__init__(errwrap(msg))


class GitHeadRefNotFoundError(RuntimeError):
    @beartype
    def __init__(self, repo: git.Repo, error: Exception):
        msg = f"""
        ValueError raised while trying to get rev 'HEAD' from repo at
        '{repo.working_dir}': '{error}'. This may have occurred because there
        are no commits in the current repository. However, this should never be
        the case, because ki repositories must be instantiated with a 'ki clone
        <collection>' command, and this command creates an initial commit.
        """
        super().__init__(errwrap(msg))


class CollectionChecksumError(RuntimeError):
    @beartype
    def __init__(self, col_file: File):
        msg = f"Checksum mismatch on {col_file}. Was file changed?"
        super().__init__(errwrap(msg))


class MissingNotetypeError(RuntimeError):
    @beartype
    def __init__(self, model: str):
        msg = f"""
        Notetype '{model}' doesn't exist. Create it in Anki before adding notes
        via ki. This may be caused by a corrupted '{MODELS_FILE}' file. The
        models file must contain definitions for all models that appear in all
        note files.
        """
        super().__init__(errwrap(msg))


# TODO: We should also print which field ordinals *are* valid.
class MissingFieldOrdinalError(RuntimeError):

    # pylint: disable=redefined-builtin

    @beartype
    def __init__(self, ord: int, model: str):
        msg = f"Field with ordinal {ord} missing from notetype '{model}'."
        super().__init__(errwrap(msg))


class MissingNoteIdError(RuntimeError):
    @beartype
    def __init__(self, nid: int):
        msg = f"Failed to locate note with nid '{nid}' in Anki database."
        super().__init__(errwrap(msg))


class NotetypeMismatchError(RuntimeError):
    @beartype
    def __init__(self, decknote: DeckNote, new_notetype: Notetype):
        msg = f"Notetype '{decknote.model}' "
        msg += f"specified in DeckNote with GUID '{decknote.guid}' "
        msg += f"does not match passed notetype '{new_notetype}'. "
        msg += "This should NEVER happen, "
        msg += "and indicates a bug in the caller to 'update_note()'."
        super().__init__(errwrap(msg))


class NotetypeKeyError(RuntimeError):
    @beartype
    def __init__(self, key: str, name: str):
        msg = f"""
        Expected key {key} not found in notetype '{name}' parsed from a
        '{MODELS_FILE}' file in the current repository (may be contained in a
        subdirectory).
        """
        super().__init__(errwrap(msg))


class NoteFieldKeyError(RuntimeError):
    @beartype
    def __init__(self, key: str, nid: int):
        msg = f"""
        Expected field {key} not found in note '{nid}'. This should *never*
        happen, and indicates a serious failure, since we only ever index
        `anki.notes.Note` objects on names pulled from their own notetype
        dictionary.
        """
        super().__init__(errwrap(msg))


class UnnamedNotetypeError(RuntimeError):
    @beartype
    def __init__(self, nt: NotetypeDict):
        msg = f"""
        Failed to find 'name' field for a notetype while parsing
        a '{MODELS_FILE}' file in the current repository (may be
        contained in a subdirectory):
        """
        super().__init__(errwrap(msg) + "\n" + str(nt))


class SQLiteLockError(RuntimeError):
    @beartype
    def __init__(self, col_file: File, err: sqlite3.DatabaseError):
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
        super().__init__(f"{header}\n{errwrap(msg)}")


class MissingMediaDirectoryError(RuntimeError):
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
        super().__init__(f"{top}\n{errwrap(msg)}")


class AnkiAlreadyOpenError(RuntimeError):
    @beartype
    def __init__(self, msg: str):
        super().__init__(f"fatal: {msg}")


class MissingTidyExecutableError(FileNotFoundError):
    @beartype
    def __init__(self, err: FileNotFoundError):
        top = "Command not found: 'tidy' (Is 'html5-tidy' installed?)"
        msg = f"Original exception: {err}"
        super().__init__(f"{top}\n{errwrap(msg)}")


class AnkiDBNoteMissingFieldsError(RuntimeError):
    @beartype
    def __init__(self, decknote: DeckNote, nid: int, key: str):
        top = f"fatal: Note with GUID '{decknote.guid}' missing DB field '{key}'"
        msg = f"""
        This is strange, should only happen if the `add_db_note()` call fails
        or behaves strangely. This may indicate a bug in ki. Please report this
        on GitHub at https://github.com/langfield/ki/issues. Note ID: '{nid}'.
        """
        super().__init__(f"{top}\n\n{errwrap(msg)}")


class GitFileModeParseError(RuntimeError):
    @beartype
    def __init__(self, file: Path, out: str):
        top = f"fatal: Failed to parse git file mode for media file '{file}'"
        msg = """
        A 'git ls-files' call is used to figure out the git file mode for
        cloned media files. This is done in order to detect symlinks on
        Windows, and follow them manually. This error is raised when we are
        unable to parse the output of 'git ls-files' for some reason or
        another, which for a symlink called 'filename', should look like this:
        """
        example = "120000 a35bd1f49b7b9225a76d052e9a35fb711a8646a6 0       filename"
        msg2 = f"Actual unparsed git command output:\n{out}"
        super().__init__(f"{top}\n\n{errwrap(msg)}\n\n{example}\n\n{msg2}")


class NonEmptyWorkingTreeError(RuntimeError):
    @beartype
    def __init__(self, repo: git.Repo):
        top = "fatal: Non-empty working tree in freshly cloned repo at "
        top += f"'{repo.working_dir}'"

        msg = """
        The working tree in a fresh clone should always be empty, and so if it
        isn't, this means that some files were either errantly generated during
        the clone process, or were not committed when they should have been.
        This may indicate a bug in ki. Please report this on GitHub at
        https://github.com/langfield/ki/issues.
        """
        details = "\nUntracked files:\n"
        for untracked in repo.untracked_files:
            details += f"  * {untracked}\n"
        details += "\nChanged files:\n"
        for item in repo.index.diff(None):
            details += f"  * {item.b_path}\n"
        super().__init__(f"{top}\n\n{errwrap(msg)}\n{details}")


class MaximumWindowsLinkChainingDepthExceededError(RuntimeError):
    @beartype
    def __init__(self, orig: File, depth: int):
        top = f"Maximum windows symlink depth exceeded while resolving '{orig}'"
        msg = f"""
        Latent symlinks are regular files whose only contents are a path to
        some other file, (possibly another windows symlink). They are used to
        implement POSIX-compatible symlinks on Win32 systems where Developer
        Mode is not enabled. This error means that we tried to resolve this
        link, and recursively followed subsequent windows symlinks to a
        ludicrous depth ('{depth}'). There is possibly a cycle within the link
        graph, which indicates that the links were written incorrectly. In
        practice, this should never happen. Please report this bug on GitHub at
        https://github.com/langfield/ki/issues.
        """
        super().__init__(f"{top}\n{errwrap(msg)}")


# WARNINGS


class NoteFieldValidationWarning(Warning):
    @beartype
    def __init__(self, nid: int, field: str, notetype: Notetype):
        top = f"Warning: Bad field '{field}' for notetype '{notetype}' in note '{nid}'"
        msg = "Try correcting the field name or changing the notetype."
        msg += f"The fields for the notetype '{notetype}' are:"
        fields: List[str] = [field.name for field in notetype.flds]
        listing: str = "  " + "\n  ".join(fields)
        super().__init__(f"{top}\n{errwrap(msg)}\n{listing}")


class WrongFieldCountWarning(Warning):
    @beartype
    def __init__(self, decknote: DeckNote, names: List[str]):
        top = f"Warning: Wrong number of fields for model '{decknote.model}'"
        msg = f"""
        The notetype '{decknote.model}' takes '{len(names)}' fields, but got
        '{len(decknote.fields.keys())}' for note with GUID '{decknote.guid}'.
        """
        super().__init__(f"{top}\n{errwrap(msg)}")


class InconsistentFieldNamesWarning(Warning):
    @beartype
    def __init__(self, x: str, y: str, decknote: DeckNote):
        top = f"Warning: Inconsistent field names ('{x}' != '{y}')"
        msg = f"""
        Expected a field '{x}' for notetype '{decknote.model}', but got a field
        '{y}' in note with GUID '{decknote.guid}'.
        """
        super().__init__(f"{top}\n{errwrap(msg)}")


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
        super().__init__(f"{top}\n{errwrap(msg)}")


class DiffTargetFileNotFoundWarning(Warning):
    @beartype
    def __init__(self, path: Path):
        top = f"Diff target file not found: '{path}'"
        msg1 = """
        Unexpected: this sometimes happens when a git repository is copied into
        a subdirectory of a ki repository, and then added with 'git add'
        instead of being added as a git submodule with 'git submodule add'. If
        git displayed a warning on a recent 'git add' command, refer to the
        hints within that warning.
        """
        msg2 = """
        Otherwise, this may indicate a bug in ki.  The caller prevents this
        warning from being instantiated unless the git change type is one of
        'ADDED', 'MODIFIED', or 'RENAMED'. In all cases, the file being diffed
        should be extant in the target commit of the repository.  However, we
        return a 'Warning' instead of an 'Exception' in order to avoid
        interrupting the execution of a 'push()' call where it is not strictly
        necessary.
        """
        super().__init__(f"{top}\n\n{errwrap(msg1)}\n\n{errwrap(msg2)}")


class RenamedMediaFileWarning(Warning):
    @beartype
    def __init__(self, src: str, dst: str):
        top = f"Media file '{src}' renamed to '{dst}'"
        msg = """
        This happens when we push a media file to a collection that already
        contains another media file with the same name. In this case, Anki does
        some deduplication by renaming the new one.
        """
        super().__init__(f"{top}\n{errwrap(msg)}")


class MissingWindowsLinkTarget(Warning):
    @beartype
    def __init__(self, link: File, tgt: str):
        top = f"Failed to locate target '{tgt}' of windows symlink '{link}'"
        msg = """
        Latent symlinks are regular files whose only contents are a path to
        some other file, (possibly another windows symlink). They are used to
        implement POSIX-compatible symlinks on Win32 systems where Developer
        Mode is not enabled. This warning means that the target of a windows
        symlink is not extant, and thus the link could not be resolved.
        """
        super().__init__(f"{top}\n{errwrap(msg)}")


class NotetypeCollisionWarning(Warning):
    @beartype
    def __init__(self, model: Notetype, existing: Notetype):
        msg = """
        Collision: new notetype '{model.name}' has same name as existing
        notetype with mid '{existing.id}', but hashes differ.
        """
        super().__init__(f"{errwrap(msg)}\n\n{nt_str(model)}\n\n{nt_str(existing)}")


class EmptyNoteWarning(Warning):
    @beartype
    def __init__(self, note: Note, health: int):
        top = f"Found empty note with nid '{note.id}'"
        msg = f"""
        Anki fields health check code: '{health}'
        """
        super().__init__(f"{top}\n{errwrap(msg)}")


class DuplicateNoteWarning(Warning):
    @beartype
    def __init__(self, note: Note, health: int, rep: str):
        top = "Failed to add duplicate note to collection"
        msg = f"""
        Notetype/fields of note with nid '{note.id}' are duplicate of existing note.
        """
        field = f"First field\n-----------\n{rep}"
        code = f"Anki fields health check code: {health}"
        super().__init__(f"{top}\n{errwrap(msg)}\n\n{field}\n\n{code}")


class UnhealthyNoteWarning(Warning):
    @beartype
    def __init__(self, note: Note, health: int):
        top = f"Note with nid '{note.id}' failed fields check with unknown error code"
        msg = f"""
        Anki fields health check code: '{health}'
        """
        super().__init__(f"{top}\n{errwrap(msg)}")


class MediaDirectoryDeckNameCollisionWarning(Warning):
    @beartype
    def __init__(self):
        top = "Decks with name '_media' skipped as name is reserved"
        super().__init__(f"{top}")
