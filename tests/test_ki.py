#!/usr/bin/env python3
"""Tests for ki command line interface (CLI)."""
import os
import io
import gc
import sys
import json
import time
import random
import shutil
import sqlite3
import tempfile
import itertools
import subprocess
import contextlib
from pathlib import Path
from functools import partial
from dataclasses import dataclass

import git
import pytest
import bitstring
import checksumdir
from lark import Lark
from lark.exceptions import UnexpectedToken
from pytest_mock import MockerFixture
from click.testing import CliRunner

import anki.collection
from anki.collection import Collection, Note

from beartype import beartype
from beartype.typing import (
    List,
    Union,
    Set,
    Iterator,
    Tuple,
    Callable,
    Iterable,
    Dict,
    Any,
    Optional,
)

import ki
import ki.maybes as M
import ki.functional as F
from ki.functional import curried
from ki import (
    BRANCH_NAME,
    KI,
    LCA,
    UTF8,
    MEDIA,
    HEAD_SUFFIX,
    REMOTE_SUFFIX,
    HASHES_FILE,
    MODELS_FILE,
    GITIGNORE_FILE,
    BACKUPS_DIR,
    NotetypeDict,
    DeckNote,
    GitChangeType,
    Notetype,
    ColNote,
    Delta,
    Dir,
    File,
    EmptyDir,
    NotetypeMismatchError,
    UnhealthyNoteWarning,
    KiRepo,
    DotKi,
    is_ignorable,
    cp_ki,
    get_note_payload,
    get_note_path,
    backup,
    update_note,
    check_fields_health,
    is_anki_note,
    parse_note,
    write_repository,
    get_target,
    push_note,
    get_models_recursively,
    append_md5sum,
    copy_media_files,
    diff2,
    _clone2,
    add_db_note,
    media_filenames_in_field,
    write_decks,
    write_collection,
    _clone1,
    _pull1,
    _push,
    get_guid,
    do,
    stardo,
)
from ki.maybes import CONFIG_FILE
from ki.types import (
    NoFile,
    PseudoFile,
    ExpectedEmptyDirectoryButGotNonEmptyDirectoryError,
    StrangeExtantPathError,
    MissingNotetypeError,
    MissingFieldOrdinalError,
    MissingNoteIdError,
    DiffTargetFileNotFoundWarning,
    NotetypeKeyError,
    UnnamedNotetypeError,
    NoteFieldKeyError,
    ExpectedDirectoryButGotFileError,
    GitHeadRefNotFoundError,
    MissingMediaDirectoryError,
    TargetExistsError,
    WrongFieldCountWarning,
    InconsistentFieldNamesWarning,
    UpdatesRejectedError,
    MediaDirectoryDeckNameCollisionWarning,
    EmptyNoteWarning,
    AnkiAlreadyOpenError,
)
from ki.transformer import NoteTransformer


# pylint: disable=unnecessary-pass, too-many-lines, invalid-name
# pylint: disable=missing-function-docstring, too-many-instance-attributes
# pylint: disable=too-many-statements, redefined-outer-name, unused-argument
# pylint: disable=no-value-for-parameter


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_PATH = os.path.join(TESTS_DIR, "data/")
COLLECTIONS_PATH = os.path.join(TEST_DATA_PATH, "collections/")


GITREPO_PATH = os.path.abspath(os.path.join(TEST_DATA_PATH, "repos/", "original/"))
MULTI_GITREPO_PATH = os.path.join(TEST_DATA_PATH, "repos/", "multideck/")
JAPANESE_GITREPO_PATH = os.path.join(TEST_DATA_PATH, "repos/", "japanese-core-2000/")

MULTI_NOTE_PATH = "aa/bb/cc/cc.md"

NOTES_PATH = os.path.abspath(os.path.join(TEST_DATA_PATH, "notes/"))
MEDIA_PATH = os.path.abspath(os.path.join(TEST_DATA_PATH, "media/"))

MEDIA_FILENAME = "bullhorn-lg.png"
MEDIA_FILE_PATH = os.path.join(MEDIA_PATH, MEDIA_FILENAME)

NOTE_2 = "note123412341234.md"
NOTE_6 = "no_guid.md"
NO_GUID_NOTE_2 = "no_guid2.md"

NOTE_2_PATH = os.path.join(NOTES_PATH, NOTE_2)
NOTE_6_PATH = os.path.join(NOTES_PATH, NOTE_6)
NO_GUID_2_NOTE_PATH = os.path.join(NOTES_PATH, NO_GUID_NOTE_2)

# A models dictionary mapping model ids to notetype dictionaries.
# Note that the `flds` field has been removed.
MODELS_DICT = {
    "1645010146011": {
        "id": 1645010146011,
        "name": "Basic",
        "type": 0,
        "mod": 0,
        "usn": 0,
        "sortf": 0,
        "did": None,
        "tmpls": [
            {
                "name": "Card 1",
                "ord": 0,
                "qfmt": "{{Front}}",
                "afmt": "{{FrontSide}}\n\n<hr id=answer>\n\n{{Back}}",
                "bqfmt": "",
                "bafmt": "",
                "did": None,
                "bfont": "",
                "bsize": 0,
            }
        ],
        "req": [[0, "any", [0]]],
    }
}
NAMELESS_MODELS_DICT = {
    "1645010146011": {
        "id": 1645010146011,
    }
}
NT = {
    "id": 1645010146011,
    "name": "Basic",
    "type": 0,
    "mod": 0,
    "usn": 0,
    "sortf": 0,
    "did": None,
    "tmpls": [
        {
            "name": "Card 1",
            "ord": 0,
            "qfmt": "{{Front}}",
            "afmt": "{{FrontSide}}\n\n<hr id=answer>\n\n{{Back}}",
            "bqfmt": "",
            "bafmt": "",
            "did": None,
            "bfont": "",
            "bsize": 0,
        }
    ],
    "flds": [
        {
            "name": "Front",
            "ord": 0,
            "sticky": False,
            "rtl": False,
            "font": "Arial",
            "size": 20,
        },
        {
            "name": "Back",
            "ord": 1,
            "sticky": False,
            "rtl": False,
            "font": "Arial",
            "size": 20,
        },
    ],
    "req": [[0, "any", [0]]],
}


# HELPER FUNCTIONS AND CLASSES


@beartype
@dataclass(frozen=True)
class SampleCollection:
    """A test collection with all names and paths constructed."""

    col_file: File
    path: Path
    stem: str
    suffix: str
    repodir: str
    filename: str
    media_db_path: Path
    media_db_filename: str
    media_directory_path: Path
    media_directory_name: str


@beartype
@dataclass(frozen=True)
class DiffReposArgs:
    """Arguments for `diff2()`."""

    repo: git.Repo
    parser: Lark
    transformer: NoteTransformer


@beartype
def get_test_collection(stem: str) -> SampleCollection:
    collections_path = Path(__file__).parent / "data" / "collections"

    # Handle restricted valid path character set on Win32.
    if stem in ("multideck", "html") and sys.platform == "win32":
        stem = f"win32_{stem}"

    repodir = stem
    suffix = ".anki2"
    filename = stem + suffix
    path = (collections_path / filename).resolve()
    media_db_filename = stem + ".media.db2"
    media_db_path = collections_path / media_db_filename
    media_directory_name = stem + ".media"
    media_directory_path = collections_path / media_directory_name

    # Get col file.
    tempdir = Path(tempfile.mkdtemp())
    col_file = tempdir / filename
    shutil.copyfile(path, col_file)

    if media_db_path.exists():
        shutil.copyfile(media_db_path, tempdir / media_db_filename)
    if media_directory_path.exists():
        shutil.copytree(media_directory_path, tempdir / media_directory_name)

    return SampleCollection(
        F.chk(Path(col_file)),
        path,
        stem,
        suffix,
        repodir,
        filename,
        media_db_path,
        media_db_filename,
        media_directory_path,
        media_directory_name,
    )


def invoke(*args, **kwargs):
    """Wrap click CliRunner invoke()."""
    return CliRunner().invoke(*args, **kwargs)


@beartype
def capture(f: Callable[[Any, ...], Any], *args, **kwargs) -> Tuple[Any, str]:
    a, b = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(a) as out, contextlib.redirect_stderr(b) as err:
        ret = f(*args, **kwargs)
    return ret, out.getvalue() + "\n" + err.getvalue()


@beartype
def clone(
    collection: File,
    directory: str = "",
) -> Tuple[git.Repo, str]:
    """Make a test `ki clone` call."""
    d = F.mkdtemp()
    os.chdir(d)
    repo, out = capture(_clone1, str(collection), directory)
    os.chdir(repo.working_dir)
    return repo, out


@beartype
def pull() -> str:
    """Make a test `ki pull` call."""
    _, out = capture(_pull1)
    return out


@beartype
def push() -> str:
    """Make a test `ki push` call."""
    _, out = capture(_push)
    return out


@beartype
def is_git_repo(path: str) -> bool:
    """Check if path is git repository."""
    if not os.path.isdir(path):
        return False
    try:
        _ = git.Repo(path).git_dir
        return True
    except git.InvalidGitRepositoryError:
        return False


@beartype
def randomly_swap_1_bit(path: File) -> None:
    """Randomly swap a bit in a file."""
    # Read in bytes.
    with open(path, "rb") as file:
        data: bytes = file.read()

    # Construct BitArray and swap bit.
    bits = bitstring.BitArray(bytes=data)
    i = random.randrange(len(bits))
    bits.invert(i)

    # Write out bytes.
    with open(path, "wb") as file:
        file.write(bits.bytes)


@beartype
def checksum_git_repository(path: str) -> str:
    """Compute a checksum of git repository without .git folder."""
    assert is_git_repo(path)
    tempdir = tempfile.mkdtemp()
    repodir = os.path.join(tempdir, "REPO")
    shutil.copytree(path, repodir)
    F.rmtree(F.chk(Path(os.path.join(repodir, ".git/"))))
    checksum = checksumdir.dirhash(repodir)
    F.rmtree(F.chk(Path(tempdir)))
    return checksum


@beartype
def get_notes(collection: File) -> List[ColNote]:
    """Get a list of notes from a path."""
    cwd: Dir = F.cwd()
    col = Collection(collection)
    F.chdir(cwd)

    notes: List[ColNote] = []
    for nid in set(col.find_notes("")):
        colnote: ColNote = M.colnote(col, nid)
        notes.append(colnote)

    return notes


@pytest.fixture
def tmpfs() -> None:
    """Run test in a subdirectory of /tmp."""
    d = F.mkdtemp()
    os.chdir(d)
    yield None
    os.chdir(Path(TESTS_DIR).parent)


# DSL


DATA = Path(__file__).parent / "data"


Deck = str
Nid = int
Field = str
Model = str
NoteSpec = Tuple[Model, List[Deck], Nid, List[Field]]


@beartype
def read(path: Union[str, Path]) -> str:
    return Path(path).read_text(encoding="UTF-8")


@beartype
def write(path: Union[str, Path], s: str) -> None:
    Path(path).write_text(s, encoding="UTF-8")


@beartype
def append(path: Union[str, Path], s: str) -> None:
    with Path(path).open("a", encoding="UTF-8") as f:
        f.write(s)


@curried
@beartype
def addnote(col: Collection, spec: NoteSpec) -> None:
    model, fullnames, nid, fields = spec
    dids = list(map(col.decks.id, fullnames))
    guid = get_guid(list(fields))
    mid = col.models.id_for_name(model)
    timestamp_ns: int = time.time_ns()
    note: Note = add_db_note(
        col=col,
        nid=nid,
        guid=guid,
        mid=mid,
        mod=int(timestamp_ns // 1e9),
        usn=-1,
        tags=[],
        fields=list(fields),
        sfld="",
        csum=0,
        flags=0,
        data="",
    )
    cids = [c.id for c in note.cards()]
    assert len(dids) == len(cids)
    stardo(lambda cid, did: note.col.set_deck([cid], did), zip(cids, dids))


@beartype
def opencol(f: File) -> Collection:
    cwd: Dir = F.cwd()
    try:
        col = Collection(f)
    except anki.errors.DBError as err:
        raise AnkiAlreadyOpenError(str(err)) from err
    F.chdir(cwd)
    return col


@beartype
def mkcol(ns: List[NoteSpec]) -> File:
    file = F.touch(F.mkdtemp(), "a.anki2")
    col = opencol(file)
    do(addnote(col), ns)
    col.close(save=True)
    return F.chk(file)


@beartype
def rm(f: File, nid: int) -> File:
    """Remove note with given `nid`."""
    col = opencol(f)
    col.remove_notes([nid])
    col.close(save=True)
    return f


@curried
@beartype
def editnote(col: Collection, spec: NoteSpec) -> None:
    _, fullnames, nid, fields = spec
    note = col.get_note(nid)
    assert len(note.keys()) == len(fields)
    stardo(note.__setitem__, zip(note.keys(), fields))
    dids = list(map(col.decks.id, fullnames))
    cids = [c.id for c in note.cards()]
    assert len(dids) == len(cids)
    stardo(lambda cid, did: note.col.set_deck([cid], did), zip(cids, dids))
    note.flush()


@beartype
def edit(f: File, spec: NoteSpec) -> File:
    """Edit a note with specified nid."""
    col = opencol(f)
    editnote(col, spec)
    col.close(save=True)
    return f


@beartype
def editcol(
    f: File,
    adds: Optional[List[NoteSpec]] = None,
    edits: Optional[List[NoteSpec]] = None,
    deletes: Optional[List[int]] = None,
) -> File:
    """Edit an existing collection file."""
    col = opencol(f)
    adds, edits, deletes = adds or [], edits or [], deletes or []
    do(addnote(col), adds)
    do(editnote(col), edits)
    col.remove_notes(deletes)
    col.close(save=True)
    return f


def mkbasic(guid: str, fields: Tuple[str, str]) -> str:
    front, back = fields
    return f"""# Note
```
guid: {guid}
notetype: Basic
```

### Tags
```
```

## Front
{front}

## Back
{back}
"""


@beartype
def write_basic(deck: str, fields: Tuple[str, str]) -> File:
    """Write a markdown note to a deck from the root of a ki repository."""
    front = fields[0]
    path = Path("./" + "/".join(deck.split("::") + [f"{front}.md"]))
    write(path, mkbasic(get_guid(list(fields)), fields))
    return F.chk(path)


@beartype
def runcmd(c: str) -> str:
    out = subprocess.run(
        c,
        text=True,
        check=True,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout
    return f"\n>>> {c}\n{out}"


@beartype
def runcmds(cs: List[str]) -> Tuple[str, str]:
    return "\n".join(cs), "".join(map(runcmd, cs))


# UTILS


def test_parse_note(tmpfs: None):
    """Does ki raise an error when it fails to parse nid?"""
    # Read grammar.
    # UNSAFE! Should we assume this always exists? A nice error message should
    # be printed on initialization if the grammar file is missing. No
    # computation should be done, and none of the click commands should work.
    grammar_path = Path(ki.__file__).resolve().parent / "grammar.lark"
    grammar = grammar_path.read_text(encoding="UTF-8")

    # Instantiate parser.
    parser = Lark(grammar, start="note", parser="lalr")
    transformer = NoteTransformer()

    with pytest.raises(UnexpectedToken):
        delta = Delta(GitChangeType.ADDED, F.chk(Path(NOTE_6_PATH)), Path("a/b"))
        parse_note(parser, transformer, delta)


def test_parse_note_handles_empty_guids(tmpfs: None):
    """Does ki generate new guids when the user does not provide one?"""
    grammar_path = Path(ki.__file__).resolve().parent / "grammar.lark"
    grammar = grammar_path.read_text(encoding="UTF-8")
    parser = Lark(grammar, start="note", parser="lalr")
    transformer = NoteTransformer()
    delta = Delta(
        status=GitChangeType.ADDED,
        path=File(NO_GUID_2_NOTE_PATH),
        relpath=Path("a/b/c/d.md"),
    )
    parse_note(parser, transformer, delta)


def test_get_batches(tmpfs: None):
    """Does it get batches from a list of strings?"""
    root = F.cwd()
    one = F.touch(root, "note1.md")
    two = F.touch(root, "note2.md")
    three = F.touch(root, "note3.md")
    four = F.touch(root, "note4.md")
    batches = list(F.get_batches([one, two, three, four], n=2))
    assert batches == [[one, two], [three, four]]


def test_is_anki_note(tmpfs: None):
    """Do the checks in ``is_anki_note()`` actually do anything?"""
    root = F.cwd()
    mda = F.touch(root, "note.mda")
    amd = F.touch(root, "note.amd")
    mdtxt = F.touch(root, "note.mdtxt")
    nd = F.touch(root, "note.nd")

    assert is_anki_note(mda) is False
    assert is_anki_note(amd) is False
    assert is_anki_note(mdtxt) is False
    assert is_anki_note(nd) is False

    note_file: File = F.touch(root, "note.md")

    note_file.write_text("", encoding="UTF-8")
    assert is_anki_note(note_file) is False

    note_file.write_text("one line", encoding="UTF-8")
    assert is_anki_note(note_file) is False

    note_file.write_text("## Note\n# Note\n\n\n\n\n\n\n\n", encoding="UTF-8")
    assert is_anki_note(note_file) is False

    note_file.write_text("# Note\n```\nguid: 00a\n\n\n\n\n\n\n", encoding="UTF-8")
    assert is_anki_note(note_file) is True

    note_file.write_text("# Note\n```\nguid: 00\n\n\n\n\n\n\n\n", encoding="UTF-8")
    assert is_anki_note(note_file) is True


@beartype
def open_collection(col_file: File) -> Collection:
    cwd: Dir = F.cwd()
    col = Collection(col_file)
    F.chdir(cwd)
    return col


def test_update_note_raises_error_on_too_few_fields(tmpfs: None):
    """Do we raise an error when the field names don't match up?"""
    ORIGINAL: SampleCollection = get_test_collection("original")
    col = open_collection(ORIGINAL.col_file)
    note = col.get_note(set(col.find_notes("")).pop())
    field = "data"

    # Note that "Back" field is missing.
    decknote = DeckNote("title", "0", "Default", "Basic", [], {"Front": field})
    notetype: Notetype = M.notetype(note.note_type())
    warnings = update_note(note, decknote, notetype, notetype)
    warning: Warning = warnings.pop()
    assert isinstance(warning, Warning)
    assert isinstance(warning, WrongFieldCountWarning)
    assert "Wrong number of fields for model 'Basic'" in str(warning)


def test_update_note_raises_error_on_too_many_fields(tmpfs: None):
    """Do we raise an error when the field names don't match up?"""
    ORIGINAL: SampleCollection = get_test_collection("original")
    col = open_collection(ORIGINAL.col_file)
    note = col.get_note(set(col.find_notes("")).pop())
    field = "data"

    # Note that "Left" field is extra.
    fields = {"Front": field, "Back": field, "Left": field}
    decknote = DeckNote("title", "0", "Default", "Basic", [], fields)

    notetype: Notetype = M.notetype(note.note_type())
    warnings = update_note(note, decknote, notetype, notetype)
    warning: Warning = warnings.pop()
    assert isinstance(warning, Warning)
    assert isinstance(warning, WrongFieldCountWarning)
    assert "Wrong number of fields for model 'Basic'" in str(warning)


def test_update_note_raises_error_wrong_field_name(tmpfs: None):
    """Do we raise an error when the field names don't match up?"""
    ORIGINAL: SampleCollection = get_test_collection("original")
    col = open_collection(ORIGINAL.col_file)
    note = col.get_note(set(col.find_notes("")).pop())
    field = "data"

    # Field `Backus` has wrong name, should be `Back`.
    fields = {"Front": field, "Backus": field}
    decknote = DeckNote("title", "0", "Default", "Basic", [], fields)

    notetype: Notetype = M.notetype(note.note_type())
    warnings = update_note(note, decknote, notetype, notetype)
    warning: Warning = warnings.pop()
    assert isinstance(warning, Warning)
    assert isinstance(warning, InconsistentFieldNamesWarning)
    assert "Inconsistent field names" in str(warning)
    assert "Backus" in str(warning)
    assert "Back" in str(warning)


def test_update_note_sets_tags(tmpfs: None):
    """Do we update tags of anki note?"""
    ORIGINAL: SampleCollection = get_test_collection("original")
    col = open_collection(ORIGINAL.col_file)
    note = col.get_note(set(col.find_notes("")).pop())
    field = "data"

    fields = {"Front": field, "Back": field}
    decknote = DeckNote("", "0", "Default", "Basic", ["tag"], fields)

    assert note.tags == []
    notetype: Notetype = M.notetype(note.note_type())
    update_note(note, decknote, notetype, notetype)
    assert note.tags == ["tag"]


def test_update_note_sets_deck(tmpfs: None):
    ORIGINAL: SampleCollection = get_test_collection("original")
    col = open_collection(ORIGINAL.col_file)
    note = col.get_note(set(col.find_notes("")).pop())
    field = "data"

    fields = {"Front": field, "Back": field}
    decknote = DeckNote("title", "0", "deck", "Basic", [], fields)

    # TODO: Remove implicit assumption that all cards are in the same deck, and
    # work with cards instead of notes.
    deck = col.decks.name(note.cards()[0].did)
    assert deck == "Default"
    notetype: Notetype = M.notetype(note.note_type())
    update_note(note, decknote, notetype, notetype)
    deck = col.decks.name(note.cards()[0].did)
    assert deck == "deck"


def test_update_note_sets_field_contents(tmpfs: None):
    ORIGINAL: SampleCollection = get_test_collection("original")
    col = open_collection(ORIGINAL.col_file)
    note = col.get_note(set(col.find_notes("")).pop())

    field = "TITLE\ndata"
    fields = {"Front": field, "Back": field}
    decknote = DeckNote("title", "0", "Default", "Basic", [], fields)

    assert "TITLE" not in note.fields[0]

    notetype: Notetype = M.notetype(note.note_type())
    update_note(note, decknote, notetype, notetype)

    assert note.fields[0] == "TITLE<br>data"


def test_update_note_removes_field_contents(tmpfs: None):
    ORIGINAL: SampleCollection = get_test_collection("original")
    col = open_collection(ORIGINAL.col_file)
    note = col.get_note(set(col.find_notes("")).pop())

    field = "y"
    fields = {"Front": field, "Back": field}
    decknote = DeckNote("title", "0", "Default", "Basic", [], fields)

    assert "a" in note.fields[0]
    notetype: Notetype = M.notetype(note.note_type())
    update_note(note, decknote, notetype, notetype)
    assert "a" not in note.fields[0]


def test_update_note_raises_error_on_nonexistent_notetype_name(tmpfs: None):
    ORIGINAL: SampleCollection = get_test_collection("original")
    col = open_collection(ORIGINAL.col_file)
    note = col.get_note(set(col.find_notes("")).pop())

    field = "data"
    fields = {"Front": field, "Back": field}
    decknote = DeckNote("title", "0", "Nonexistent", "Default", [], fields)

    notetype: Notetype = M.notetype(note.note_type())

    with pytest.raises(NotetypeMismatchError):
        update_note(note, decknote, notetype, notetype)


def test_check_fields_health_catches_missing_clozes(tmpfs: None):
    ORIGINAL: SampleCollection = get_test_collection("original")
    col = open_collection(ORIGINAL.col_file)
    note = col.get_note(set(col.find_notes("")).pop())

    field = "data"
    fields = {"Text": field, "Back Extra": ""}
    decknote = DeckNote("title", "0", "Default", "Cloze", [], fields)

    clz: NotetypeDict = col.models.by_name("Cloze")
    cloze: Notetype = M.notetype(clz)
    notetype: Notetype = M.notetype(note.note_type())
    warnings = update_note(note, decknote, notetype, cloze)
    warning = list(warnings).pop()
    assert isinstance(warning, Exception)
    assert isinstance(warning, UnhealthyNoteWarning)

    assert "'1645010162168'" in str(warning)
    assert "Anki fields health check code: '3'" in str(warning)


def test_update_note_changes_notetype(tmpfs: None):
    ORIGINAL: SampleCollection = get_test_collection("original")
    col = open_collection(ORIGINAL.col_file)
    note = col.get_note(set(col.find_notes("")).pop())

    field = "data"
    fields = {"Front": field, "Back": field}
    decknote = DeckNote(
        "title", "0", "Default", "Basic (and reversed card)", [], fields
    )

    rev: NotetypeDict = col.models.by_name("Basic (and reversed card)")
    reverse: Notetype = M.notetype(rev)
    notetype: Notetype = M.notetype(note.note_type())
    update_note(note, decknote, notetype, reverse)


def test_check_fields_health_catches_empty_notes(tmpfs: None):
    ORIGINAL: SampleCollection = get_test_collection("original")
    col = open_collection(ORIGINAL.col_file)
    note = col.get_note(set(col.find_notes("")).pop())

    note.fields = []
    w = check_fields_health(note)[0]
    assert isinstance(w, EmptyNoteWarning)


def test_slugify_handles_unicode(tmpfs: None):
    """Test that slugify handles unicode alphanumerics."""
    # Hiragana should be okay.
    text = "ゅ"
    result = F.slugify(text)
    assert result == text

    # Emojis as well.
    text = "😶"
    result = F.slugify(text)
    assert result == text


def test_slugify_handles_html_tags(tmpfs: None):
    text = '<img src="card11front.jpg" />'
    result = F.slugify(text)

    assert result == "img-srccard11frontjpg"


@beartype
def get_colnote_with_sfld(sfld: str) -> Tuple[Collection, ColNote]:
    ORIGINAL: SampleCollection = get_test_collection("original")
    col = open_collection(ORIGINAL.col_file)
    note = col.get_note(set(col.find_notes("")).pop())

    notetype: Notetype = M.notetype(note.note_type())
    deck = col.decks.name(note.cards()[0].did)
    return col, ColNote(
        n=note,
        new=False,
        deck=deck,
        title="",
        markdown=False,
        notetype=notetype,
        sfld=sfld,
    )


@beartype
def get_basic_colnote_with_fields(front: str, back: str) -> Tuple[Collection, ColNote]:
    ORIGINAL: SampleCollection = get_test_collection("original")
    col = open_collection(ORIGINAL.col_file)
    note = col.get_note(set(col.find_notes("")).pop())
    note["Front"] = front
    note["Back"] = back
    note.flush()

    notetype: Notetype = M.notetype(note.note_type())
    deck = col.decks.name(note.cards()[0].did)
    return col, ColNote(
        n=note,
        new=False,
        deck=deck,
        title="",
        markdown=False,
        notetype=notetype,
        sfld=note[notetype.sortf.name],
    )


def test_get_note_path_produces_nonempty_filenames(tmpfs: None):
    field_text = '<img src="card11front.jpg" />'
    _, colnote = get_colnote_with_sfld(field_text)

    deck_dir: Dir = F.force_mkdir(Path("a"))

    path: File = get_note_path(colnote, deck_dir)
    assert path.name == "img-srccard11frontjpg.md"

    # Check that it even works if the field is empty.
    _, empty_colnote = get_colnote_with_sfld("")
    path: File = get_note_path(empty_colnote, deck_dir)
    assert ".md" in str(path)
    assert f"{os.sep}a{os.sep}" in str(path)

    _, empty = get_basic_colnote_with_fields("", "")
    path: File = get_note_path(empty, deck_dir)


@beartype
def get_diff2_args() -> Tuple[git.Repo, Callable[[Delta], DeckNote]]:
    """
    A test 'fixture' (not really a pytest fixture, but a setup function) to be
    called when we need to test `diff2()`.

    Basically a section of the code from `push()`, but without any error
    handling, since we expect things to work out nicely, and for the
    repositories operated upon during tests to be valid.

    Returns the values needed to pass as arguments to `diff2()`.

    This makes ephemeral repositories, so we should make any changes we expect
    to see the results of in `deltas: List[Delta]` *before* calling this
    function. For example, if we wanted to add a note, and then expected to see
    a `GitChangeType.ADDED`, then we should do that in `ORIGINAL.repodir`
    before calling this function.
    """
    kirepo: KiRepo = M.kirepo(F.cwd())
    md5sum: str = F.md5(kirepo.col_file)
    hashes: List[str] = kirepo.hashes_file.read_text(encoding=UTF8).split("\n")
    hashes = list(filter(lambda l: l != "", hashes))
    if md5sum not in hashes[-1]:
        raise UpdatesRejectedError(kirepo.col_file)
    head_kirepo: KiRepo = cp_ki(M.head_ki(kirepo), f"{HEAD_SUFFIX}-{md5sum}")
    remote_root: EmptyDir = F.mksubdir(F.mkdtemp(), REMOTE_SUFFIX / md5sum)
    msg = f"Fetch changes from collection '{kirepo.col_file}' with md5sum '{md5sum}'"
    col = M.collection(kirepo.col_file)
    remote_repo, _ = _clone2(col, remote_root, msg, silent=True)
    remote_repo = M.gitcopy(remote_repo, head_kirepo.root, unsub=True)
    F.commitall(remote_repo, f"Pull changes from repository at `{kirepo.root}`")
    parser, transformer = M.parser_and_transformer()
    parse: Callable[[Delta], DeckNote] = partial(parse_note, parser, transformer)
    return remote_repo, parse


def test_diff2_shows_no_changes_when_no_changes_have_been_made(tmpfs: None):
    ORIGINAL = get_test_collection("original")
    clone(ORIGINAL.col_file)
    deltas: List[Delta] = diff2(*get_diff2_args())
    changed = [str(delta.path) for delta in deltas]
    assert changed == []


def test_diff2_yields_a_warning_when_a_file_cannot_be_found(tmpfs: None):
    ORIGINAL = get_test_collection("original")
    clone(ORIGINAL.col_file)
    shutil.copyfile(NOTE_2_PATH, NOTE_2)
    repo = git.Repo(".")
    repo.git.add(all=True)
    repo.index.commit("CommitMessage")

    args = get_diff2_args()
    remote_repo = args[0]

    os.remove(Path(remote_repo.working_dir) / NOTE_2)

    deltas: List[Union[Delta, Warning]] = diff2(*args)
    del args
    gc.collect()
    warnings = [d for d in deltas if isinstance(d, DiffTargetFileNotFoundWarning)]
    assert len(warnings) == 1
    warning = warnings.pop()
    assert "note123412341234.md" in str(warning)


def test_backup_is_no_op_when_backup_already_exists(mocker: MockerFixture):
    """
    Does the second subsequent backup call register as redundant (we assume a
    nice error message is displayed if so)?
    """
    ORIGINAL = get_test_collection("original")

    clone(ORIGINAL.col_file)
    kirepo: KiRepo = M.kirepo(F.cwd())

    mocker.patch("ki.echo")

    returncode: int = backup(kirepo)
    assert returncode == 0

    returncode = backup(kirepo)
    assert returncode == 1


def test_get_note_path(tmpfs: None):
    """Do we add ordinals to generated filenames if there are duplicates?"""
    _, colnote = get_colnote_with_sfld("a")
    deck_dir = F.cwd()
    dupe_path = deck_dir / "a.md"
    dupe_path.write_text("ay")
    note_path = get_note_path(colnote, deck_dir)
    assert str(note_path.name) == "a_1.md"


def test_create_deck_dir(tmpfs: None):
    deckname = "aa::bb::cc"
    root = F.cwd()
    path = M.deckd(deckname, root)
    assert path.is_dir()
    assert os.path.isdir("aa/bb/cc")


def test_create_deck_dir_strips_leading_periods(tmpfs: None):
    deckname = ".aa::bb::.cc"
    root = F.cwd()
    path = M.deckd(deckname, root)
    assert path.is_dir()
    assert os.path.isdir("aa/bb/cc")


def test_get_note_payload(tmpfs: None):
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    col = opencol(mkcol([n1]))
    note = col.get_note(set(col.find_notes("")).pop())
    colnote: ColNote = M.colnote(col, note.id)
    result = get_note_payload(colnote)
    # Check that the dumped `front` field content is there.
    assert "a" in result
    # The `back` field content.
    assert "\nb\n" in result


def test_write_repository_handles_html(tmpfs: None):
    """Does generated repo handle html okay?"""
    HTML: SampleCollection = get_test_collection("html")

    targetdir = F.mkdir(F.chk(Path(HTML.repodir)))

    kidir, mediadir = M.empty_kirepo(targetdir)
    dotki: DotKi = M.dotki(kidir)
    col = M.collection(HTML.col_file)
    _ = write_repository(col, targetdir, dotki, mediadir)

    note_file = targetdir / "Default" / "あだ名.md"
    contents: str = note_file.read_text(encoding="UTF-8")

    assert '<div class="word-card"> <table class="kanji-match">' in contents


@beartype
def test_write_repository_propagates_errors_from_colnote(
    tmpfs: None, mocker: MockerFixture
):
    """Do errors get forwarded nicdely?"""
    HTML: SampleCollection = get_test_collection("html")

    targetdir = F.mkdir(F.chk(Path(HTML.repodir)))
    kidir, mediadir = M.empty_kirepo(targetdir)
    dotki: DotKi = M.dotki(kidir)

    mocker.patch("ki.M.colnote", side_effect=NoteFieldKeyError("'bad_field_key'", 0))
    with pytest.raises(NoteFieldKeyError) as error:
        col = M.collection(HTML.col_file)
        _ = write_repository(col, targetdir, dotki, mediadir)
    assert "'bad_field_key'" in str(error.exconly())


def test_maybe_kirepo_displays_nice_errors(tmpfs: None):
    """Does a nice error get printed when kirepo metadata is missing?"""
    ORIGINAL = get_test_collection("original")

    # Case where `.ki/` directory doesn't exist.
    clone(ORIGINAL.col_file)
    os.chdir("../")
    targetdir: Dir = F.chk(Path(ORIGINAL.repodir))
    F.rmtree(targetdir / KI)
    with pytest.raises(Exception) as error:
        M.kirepo(targetdir)
    assert "fatal: not a ki repository" in str(error.exconly())
    F.rmtree(targetdir)

    # Case where `.ki/` is a file instead of a directory.
    clone(ORIGINAL.col_file)
    os.chdir("../")
    targetdir: Dir = F.chk(Path(ORIGINAL.repodir))
    F.rmtree(targetdir / KI)
    (targetdir / KI).touch()
    with pytest.raises(Exception) as error:
        M.kirepo(targetdir)
    assert "fatal: not a ki repository" in str(error.exconly())
    F.rmtree(targetdir)

    # Case where `.ki/backups` directory doesn't exist.
    clone(ORIGINAL.col_file)
    os.chdir("../")
    targetdir: Dir = F.chk(Path(ORIGINAL.repodir))
    F.rmtree(targetdir / KI / BACKUPS_DIR)
    with pytest.raises(Exception) as error:
        M.kirepo(targetdir)
    assert "Directory not found" in str(error.exconly())
    assert "'.ki/backups'" in str(error.exconly())
    F.rmtree(targetdir)

    # Case where `.ki/backups` is a file instead of a directory.
    clone(ORIGINAL.col_file)
    os.chdir("../")
    targetdir: Dir = F.chk(Path(ORIGINAL.repodir))
    F.rmtree(targetdir / KI / BACKUPS_DIR)
    (targetdir / KI / BACKUPS_DIR).touch()
    with pytest.raises(Exception) as error:
        M.kirepo(targetdir)
    assert "A directory was expected" in str(error.exconly())
    assert "'.ki/backups'" in str(error.exconly())
    gc.collect()
    F.rmtree(targetdir)

    # Case where `.ki/config` file doesn't exist.
    clone(ORIGINAL.col_file)
    os.chdir("../")
    targetdir: Dir = F.chk(Path(ORIGINAL.repodir))
    os.remove(targetdir / KI / CONFIG_FILE)
    with pytest.raises(Exception) as error:
        M.kirepo(targetdir)
    assert "File not found" in str(error.exconly())
    assert "'.ki/config'" in str(error.exconly())
    F.rmtree(targetdir)

    # Case where `.ki/config` is a directory instead of a file.
    clone(ORIGINAL.col_file)
    os.chdir("../")
    targetdir: Dir = F.chk(Path(ORIGINAL.repodir))
    os.remove(targetdir / KI / CONFIG_FILE)
    os.mkdir(targetdir / KI / CONFIG_FILE)
    with pytest.raises(Exception) as error:
        M.kirepo(targetdir)
    assert "A file was expected" in str(error.exconly())
    assert "'.ki/config'" in str(error.exconly())
    F.rmtree(targetdir)

    # Case where `.ki/hashes` file doesn't exist.
    clone(ORIGINAL.col_file)
    os.chdir("../")
    targetdir: Dir = F.chk(Path(ORIGINAL.repodir))
    os.remove(targetdir / KI / HASHES_FILE)
    with pytest.raises(Exception) as error:
        M.kirepo(targetdir)
    assert "File not found" in str(error.exconly())
    assert "'.ki/hashes'" in str(error.exconly())
    F.rmtree(targetdir)

    # Case where `.ki/models` file doesn't exist.
    clone(ORIGINAL.col_file)
    os.chdir("../")
    targetdir: Dir = F.chk(Path(ORIGINAL.repodir))
    os.remove(targetdir / MODELS_FILE)
    with pytest.raises(Exception) as error:
        M.kirepo(targetdir)
    assert "File not found" in str(error.exconly())
    assert f"'{MODELS_FILE}'" in str(error.exconly())
    F.rmtree(targetdir)

    # Case where collection file doesn't exist.
    clone(ORIGINAL.col_file)
    os.chdir("../")
    targetdir: Dir = F.chk(Path(ORIGINAL.repodir))
    os.remove(ORIGINAL.col_file)
    with pytest.raises(Exception) as error:
        M.kirepo(targetdir)
    assert "File not found" in str(error.exconly())
    assert "'.anki2'" in str(error.exconly())
    assert "database" in str(error.exconly())
    assert ORIGINAL.filename in str(error.exconly())
    F.rmtree(targetdir)


def test_get_target(tmpfs: None):
    """Do we print a nice error when the targetdir is nonempty?"""

    os.mkdir("file")
    Path("file/subfile").touch()
    col_file: File = F.touch(F.cwd(), "file.anki2")
    with pytest.raises(TargetExistsError) as error:
        get_target(F.cwd(), col_file, "")
    assert "fatal: destination path" in str(error.exconly())
    assert "file" in str(error.exconly())


def test_maybe_emptydir(tmpfs: None):
    """Do we print a nice error when the directory is unexpectedly nonempty?"""

    Path("file").touch()
    with pytest.raises(ExpectedEmptyDirectoryButGotNonEmptyDirectoryError) as error:
        M.emptydir(F.cwd())
    assert "but it is nonempty" in str(error.exconly())
    assert str(Path.cwd()) in str(error.exconly()).replace("\n", "")


def test_maybe_emptydir_handles_non_directories(tmpfs: None):
    """Do we print a nice error when the path is not a directory?"""
    file = Path("file")
    file.touch()
    with pytest.raises(ExpectedDirectoryButGotFileError) as error:
        M.emptydir(file)
    assert str(file) in str(error.exconly()).replace("\n", "")


@pytest.mark.skipif(
    sys.platform == "win32", reason="Windows does not have `os.mkfifo()`."
)
def test_maybe_xdir(tmpfs: None):
    """Do we print a nice error when there is a non-file non-directory thing?"""
    os.mkfifo("pipe")
    with pytest.raises(StrangeExtantPathError) as error:
        M.xdir(Path("pipe"))
    assert "pseudofile" in str(error.exconly())
    assert "pipe" in str(error.exconly())


@pytest.mark.skipif(
    sys.platform == "win32", reason="Windows does not have `os.mkfifo()`."
)
def test_maybe_xfile(tmpfs: None):
    """Do we print a nice error when there is a non-file non-directory thing?"""
    os.mkfifo("pipe")
    with pytest.raises(StrangeExtantPathError) as error:
        M.xfile(Path("pipe"))
    assert "pseudofile" in str(error.exconly())
    assert "pipe" in str(error.exconly())


def test_push_note(tmpfs: None):
    """Do we print a nice error when a notetype is missing?"""
    ORIGINAL: SampleCollection = get_test_collection("original")
    col = open_collection(ORIGINAL.col_file)

    field = "data"
    fields = {"Front": field, "Back": field}
    decknote = DeckNote("title", "0", "Default", "NonexistentModel", [], fields)
    new_nids: Iterator[int] = itertools.count(int(time.time_ns() / 1e6))
    with pytest.raises(MissingNotetypeError) as error:
        push_note(col, int(time.time_ns()), {}, new_nids, decknote)
    assert "NonexistentModel" in str(error.exconly())


def test_maybe_head(tmpfs: None):
    repo = git.Repo.init("repo")
    with pytest.raises(GitHeadRefNotFoundError) as error:
        M.head(repo)
    err_snippet = "ValueError raised while trying to get rev 'HEAD' from repo"
    assert err_snippet in str(error.exconly())


def test_maybe_head_ki(tmpfs: None):
    ORIGINAL: SampleCollection = get_test_collection("original")

    # Initialize the kirepo *without* committing.
    os.mkdir("original")
    targetdir = F.chk(Path("original"))

    kidir, mediadir = M.empty_kirepo(targetdir)
    dotki: DotKi = M.dotki(kidir)
    md5sum = F.md5(ORIGINAL.col_file)
    (targetdir / GITIGNORE_FILE).write_text(KI + "\n")

    # Write notes to disk.
    col = M.collection(ORIGINAL.col_file)
    _ = write_repository(col, targetdir, dotki, mediadir)

    repo = git.Repo.init(targetdir, initial_branch=BRANCH_NAME)
    append_md5sum(kidir, ORIGINAL.col_file.name, md5sum)

    # Since we didn't commit, there will be no HEAD.
    kirepo = M.kirepo(F.chk(Path(repo.working_dir)))
    with pytest.raises(GitHeadRefNotFoundError) as error:
        M.head_ki(kirepo)
    err_snippet = "ValueError raised while trying to get rev 'HEAD' from repo"
    assert err_snippet in str(error.exconly())


def test_push_note_handles_note_field_name_mismatches(tmpfs: None):
    """Do we return a nice warning when note fields are missing?"""
    ORIGINAL: SampleCollection = get_test_collection("original")
    col = open_collection(ORIGINAL.col_file)

    field = "data"
    fields = {"Front": field, "Backk": field}
    decknote = DeckNote("title", "0", "Default", "Basic", [], fields)
    new_nids: Iterator[int] = itertools.count(int(time.time_ns() / 1e6))
    warnings = push_note(col, int(time.time_ns()), {}, new_nids, decknote)
    assert len(warnings) == 1
    warning = warnings.pop()
    assert isinstance(warning, InconsistentFieldNamesWarning)
    assert "Backk" in str(warning)
    assert "Expected a field" in str(warning)


def test_notetype(tmpfs: None):
    nt = NT

    # This field ordinal doesn't exist.
    nt["sortf"] = 3
    with pytest.raises(MissingFieldOrdinalError) as error:
        M.notetype(nt)
    assert "3" in str(error.exconly())


def test_colnote_prints_nice_error_when_nid_doesnt_exist(tmpfs: None):
    ORIGINAL: SampleCollection = get_test_collection("original")
    col = open_collection(ORIGINAL.col_file)
    nid = 44444444444444444
    with pytest.raises(MissingNoteIdError) as error:
        M.colnote(col, nid)
    assert str(nid) in str(error.exconly())


@beartype
def test_colnote_propagates_errors_from_notetype(mocker: MockerFixture):
    ORIGINAL: SampleCollection = get_test_collection("original")
    col = open_collection(ORIGINAL.col_file)
    note = col.get_note(set(col.find_notes("")).pop())

    mocker.patch("ki.M.notetype", side_effect=UnnamedNotetypeError(NT))
    with pytest.raises(UnnamedNotetypeError) as error:
        M.colnote(col, note.id)
    assert "Failed to find 'name' field" in str(error.exconly())


@beartype
def test_colnote_propagates_errors_key_errors_from_sort_field(
    mocker: MockerFixture,
):
    mocker.patch("anki.notes.Note.__getitem__", side_effect=KeyError("bad_field_key"))
    ORIGINAL: SampleCollection = get_test_collection("original")
    col = open_collection(ORIGINAL.col_file)
    note = col.get_note(set(col.find_notes("")).pop())
    with pytest.raises(NoteFieldKeyError) as error:
        M.colnote(col, note.id)
    assert "Expected field" in str(error.exconly())
    assert "'bad_field_key'" in str(error.exconly())


def test_nopath(tmpfs: None):
    file = Path("file")
    file.touch()
    with pytest.raises(FileExistsError):
        M.nopath(file)


def test_filter_note_path(tmpfs: None):
    directory = Path("directory")
    directory.mkdir()
    path = directory / "file"
    path.touch()
    root: Dir = F.cwd()
    assert is_ignorable(root=root, path=path) is True


def test_get_models_recursively(tmpfs: None):
    ORIGINAL = get_test_collection("original")

    clone(ORIGINAL.col_file)
    os.remove(MODELS_FILE)
    with open(Path(MODELS_FILE), "w", encoding="UTF-8") as models_f:
        json.dump(MODELS_DICT, models_f, ensure_ascii=False, indent=4)
    kirepo: KiRepo = M.kirepo(F.cwd())
    with pytest.raises(NotetypeKeyError) as error:
        get_models_recursively(kirepo)
    assert "not found in notetype" in str(error.exconly())
    assert "flds" in str(error.exconly())
    assert "Basic" in str(error.exconly())


def test_get_models_recursively_prints_a_nice_error_when_models_dont_have_a_name(
    tmpfs: None,
):
    ORIGINAL = get_test_collection("original")

    clone(ORIGINAL.col_file)
    os.remove(MODELS_FILE)
    with open(Path(MODELS_FILE), "w", encoding="UTF-8") as models_f:
        json.dump(NAMELESS_MODELS_DICT, models_f, ensure_ascii=False, indent=4)
    kirepo: KiRepo = M.kirepo(F.cwd())
    with pytest.raises(UnnamedNotetypeError) as error:
        get_models_recursively(kirepo)
    assert "Failed to find 'name' field" in str(error.exconly())
    assert "1645010146011" in str(error.exconly())


@pytest.mark.skipif(
    sys.platform == "win32", reason="Windows does not have `os.mkfifo()`."
)
def test_ftest_handles_strange_paths(tmpfs: None):
    """Do we print a nice error when there is a non-file non-directory thing?"""
    os.mkfifo("pipe")
    pipe = F.chk(Path("pipe"))
    assert isinstance(pipe, PseudoFile)


def test_fparent_handles_fs_root(tmpfs: None):
    root: str = os.path.abspath(os.sep)
    parent = F.parent(F.chk(Path(root)))
    assert isinstance(parent, Dir)
    assert str(parent) == root


def test_copy_media_files_returns_nice_errors(tmpfs: None):
    """Does `copy_media_files()` handle case where media directory doesn't exist?"""
    MEDIACOL: SampleCollection = get_test_collection("media")
    col: Collection = open_collection(MEDIACOL.col_file)

    # Remove the media directory.
    media_dir = F.chk(MEDIACOL.col_file.parent / MEDIACOL.media_directory_name)
    F.rmtree(media_dir)

    with pytest.raises(MissingMediaDirectoryError) as error:
        copy_media_files(col, F.mkdtemp())
    assert "media.media" in str(error.exconly())
    assert "bad Anki collection media directory" in str(error.exconly())


def test_copy_media_files_finds_notetype_media(tmpfs: None):
    """Does `copy_media_files()` get files like `collection.media/_vonNeumann.jpg`?"""
    MEDIACOL: SampleCollection = get_test_collection("media")
    col: Collection = open_collection(MEDIACOL.col_file)

    media = copy_media_files(col, F.mkdtemp())
    media_files: Set[File] = set()
    for media_set in media.values():
        media_files = media_files | media_set
    assert len(media_files) == 2
    media_files = sorted(list(media_files))
    von_neumann: File = media_files[1]
    assert "_vonNeumann" in str(von_neumann)


def test_shallow_walk_returns_extant_paths(tmpfs: None):
    """Shallow walk should only return paths that actually exist."""
    tmpdir = F.mkdtemp()
    (tmpdir / "file").touch()
    (tmpdir / "directory").mkdir()
    root, dirs, files = F.shallow_walk(tmpdir)
    assert os.path.isdir(root)
    for d in dirs:
        assert os.path.isdir(d)
    for file in files:
        assert os.path.isfile(file)


def test_get_test_collection_copies_media(tmpfs: None):
    """Do media files get copied into the temp directory?"""
    MEDIACOL: SampleCollection = get_test_collection("media")
    assert MEDIACOL.col_file.parent / (MEDIACOL.stem + ".media") / "1sec.mp3"


def test_cp_repo_preserves_git_symlink_file_modes(tmpfs: None):
    """Especially on Windows, are copied symlinks still mode 120000?"""
    MEDIACOL: SampleCollection = get_test_collection("media")

    # Clone.
    repo, _ = clone(MEDIACOL.col_file)

    # Check that filemode is initially 120000.
    onesec_file = F.root(repo) / "Default" / MEDIA / "1sec.mp3"
    mode = M.filemode(onesec_file)
    assert mode == 120000

    # Check that `cp_repo()` keeps it as 120000.
    rev = M.head(repo)
    ephem = ki.cp_repo(rev, "filemode-test")
    onesec_file = F.root(ephem) / "Default" / MEDIA / "1sec.mp3"
    mode = M.filemode(onesec_file)
    assert mode == 120000


def test_write_deck_node_cards_does_not_fail_due_to_special_characters_in_paths_on_windows(
    tmpfs: None,
):
    """If there are e.g. asterisks in a `guid`, do we still write the note file successfully?"""
    ORIGINAL: SampleCollection = get_test_collection("original")
    col = Collection(ORIGINAL.col_file)
    timestamp_ns: int = time.time_ns()
    new_nids: Iterator[int] = itertools.count(int(timestamp_ns / 1e6))

    nid = next(new_nids)
    mid = 1645010146011
    guid = r"QM6kTt.Nt*"
    decknote = DeckNote(
        title="title",
        guid=guid,
        deck="Default",
        model="Basic",
        tags=[],
        fields={"Front": "", "Back": "b"},
    )
    note: Note = add_db_note(
        col,
        nid,
        guid,
        mid,
        mod=int(timestamp_ns // 1e9),
        usn=-1,
        tags=decknote.tags,
        fields=list(decknote.fields.values()),
        sfld="",
        csum=0,
        flags=0,
        data="",
    )
    notetype: Notetype = M.notetype(note.note_type())

    targetdir: Dir = F.mkdir(F.chk(Path("collection")))
    colnote = ColNote(
        n=note,
        new=True,
        deck="Default",
        title="None",
        markdown=False,
        notetype=notetype,
        sfld="",
    )
    payload: str = "payload"
    note_path: NoFile = get_note_path(colnote, targetdir)
    note_path: File = F.write(note_path, payload)


@pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning")
@pytest.mark.skipif(
    sys.platform == "win32", reason="Windows does not support colons in paths."
)
def test_diff2_handles_paths_containing_colons(tmp_path: Path):
    """Check that gitpython doesn't crash when a path contains a `:`."""
    # Create parser and transformer.
    grammar_path = Path(ki.__file__).resolve().parent / "grammar.lark"
    grammar = grammar_path.read_text(encoding="UTF-8")
    parser = Lark(grammar, start="note", parser="lalr")
    transformer = NoteTransformer()

    # Move to tmp directory.
    tgt = F.chk(tmp_path / "mwe")
    repodir = F.mkdir(tgt)
    repo = git.Repo.init(repodir, initial_branch=BRANCH_NAME)
    _ = F.mkdir(F.chk(repodir / "a:b"))
    file = F.chk(tgt / "a:b" / "file")
    file.write_text("a", encoding="UTF-8")
    repo.git.add(all=True)
    repo.index.commit("Initial commit.")
    file.write_text("b", encoding="UTF-8")
    repo.git.add(all=True)
    repo.index.commit("Edit file.")
    parse = partial(parse_note, parser, transformer)
    diff2(repo, parse)


def test_media_filenames_in_field_strips_newlines(mocker: MockerFixture):
    """Are newlines stripped from all found media filenames?"""
    s = (
        "Wie sieht die Linkstruktur von einem Hub in einem Web-Graphen aus?"
        "\x1fEin guter Hub zeigt auf viele Authorities:"
        '\n<div><img src=\n"paste-64c7a314b90f3e9ef1b2d94edb396e07a121afdf.jpg"></div>'
    )
    mock = mocker.MagicMock()
    mock.__class__ = Collection
    mock.media.regexps = [
        "(?i)(\\[sound:(?P<fname>[^]]+)\\])",
        "(?i)(<(?:img|audio)\\b[^>]* src=(?P<str>[\\\"'])(?P<fname>[^>]+?)(?P=str)[^>]*>)",
        "(?i)(<(?:img|audio)\\b[^>]* src=(?!['\\\"])(?P<fname>[^ >]+)[^>]*?>)",
        "(?i)(<object\\b[^>]* data=(?P<str>[\\\"'])(?P<fname>[^>]+?)(?P=str)[^>]*>)",
        "(?i)(<object\\b[^>]* data=(?!['\\\"])(?P<fname>[^ >]+)[^>]*?>)",
    ]
    fnames: Iterable[str] = media_filenames_in_field(mock, s)
    assert not any(map(lambda f: "\n" in f, fnames))
    assert not any(map(lambda f: "\\n" in f, fnames))


def test_write_decks_skips_root_deck(tmp_path: Path):
    """Is `[no deck]` skipped (as it should be)?"""
    ORIGINAL: SampleCollection = get_test_collection("original")
    col = open_collection(ORIGINAL.col_file)
    nids: Iterable[int] = col.find_notes(query="")
    colnotes: Dict[int, ColNote] = {nid: M.colnote(col, nid) for nid in nids}
    write_decks(col, Dir(tmp_path), colnotes, {})
    assert not os.path.isdir(tmp_path / "[no deck]")


def test_unstaged_working_tree_changes_are_not_stashed_in_write_collection(
    tmp_path: Path,
    mocker: MockerFixture,
):
    """Do we preserve working tree changes during push ops?"""
    # Create repo and commit some contents.
    targetdir = Dir(tmp_path / "targetdir")
    targetdir.mkdir()
    col_file = File(tmp_path / "collection.anki2")
    col_file.touch()
    r: git.Repo = git.Repo.init(targetdir, initial_branch=BRANCH_NAME)
    (targetdir / "file").write_text("a", encoding="UTF-8")
    F.commitall(r, "a")
    r.create_tag(LCA)

    # Make working tree changes.
    (targetdir / "file").write_text("b", encoding="UTF-8")

    # Create mocks.
    kirepo = mocker.MagicMock()
    kirepo.repo = r
    kirepo.col_file = col_file
    parse = mocker.MagicMock()
    head_kirepo = mocker.MagicMock()
    col = mocker.MagicMock()

    # Set class properties to fool @beartype.
    head_kirepo.__class__ = KiRepo
    kirepo.__class__ = KiRepo
    col.__class__ = Collection

    # Patch some irrelevant functions.
    mocker.patch("ki.backup")
    mocker.patch("ki.functional.rglob")
    mocker.patch("ki.append_md5sum")
    mocker.patch("ki.commit_hashes_file")

    # Write collection.
    _ = write_collection([], {}, kirepo, parse, head_kirepo, col)

    # Check that unstaged working tree changes are still there.
    diffs = set(r.index.diff(None))
    assert len(diffs) == 1
    diff = diffs.pop()
    assert isinstance(diff, git.Diff)
    assert diff.a_path == "file"
    assert diff.b_path == "file"


def test_write_decks_warns_about_media_deck_name_collisions(
    tmp_path: Path, mocker: MockerFixture
):
    DECK: SampleCollection = get_test_collection(
        "deck_with_same_name_as_media_directory"
    )
    col = open_collection(DECK.col_file)
    nids: Iterable[int] = col.find_notes(query="")
    colnotes: Dict[int, ColNote] = {nid: M.colnote(col, nid) for nid in nids}
    mock = mocker.patch("ki.warn")
    with pytest.raises(ValueError) as err:
        write_decks(col, Dir(tmp_path), colnotes, {})
    assert "_media" in str(err)
