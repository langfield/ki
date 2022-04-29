#!/usr/bin/env python3
"""Tests for ki command line interface (CLI)."""
import os
import json
import random
import shutil
import tempfile
import functools
from pathlib import Path
from dataclasses import dataclass

import git
import pytest
import bitstring
import checksumdir
import prettyprinter as pp
from lark import Lark
from lark.exceptions import UnexpectedToken
from loguru import logger
from result import Ok, Err, OkErr, Result
from pytest_mock import MockerFixture
from click.testing import CliRunner
from anki.collection import Collection

from beartype import beartype, roar
from beartype.typing import List, Callable, Union

import ki
import ki.maybes as M
import ki.functional as F
from ki import (
    BRANCH_NAME,
    STAGE_SUFFIX,
    DELETED_SUFFIX,
    IGNORE,
    KI,
    MEDIA,
    HASHES_FILE,
    MODELS_FILE,
    CONFIG_FILE,
    LAST_PUSH_FILE,
    GITIGNORE_FILE,
    BACKUPS_DIR,
    NO_SM_DIR,
    NotetypeDict,
    GitChangeType,
    Notetype,
    ColNote,
    Delta,
    ExtantDir,
    ExtantFile,
    EmptyDir,
    NotetypeMismatchError,
    UnhealthyNoteWarning,
    NoteFieldValidationWarning,
    KiRepo,
    KiRepoRef,
    RepoRef,
    get_ephemeral_kirepo,
    get_note_payload,
    create_deck_dir,
    tidy_html_recursively,
    get_note_path,
    git_subprocess_pull,
    get_colnote,
    backup,
    get_ephemeral_repo,
    diff_repos,
    update_note,
    parse_notetype_dict,
    display_fields_health_warning,
    is_anki_note,
    parse_markdown_note,
    flatten_staging_repo,
    filter_note_path,
    lock,
    unsubmodule_repo,
    write_repository,
    get_target,
    push_flatnote_to_anki,
    get_models_recursively,
    append_md5sum,
    get_media_files,
)
from ki.types import (
    Leaves,
    ExtantStrangePath,
    ExpectedEmptyDirectoryButGotNonEmptyDirectoryError,
    StrangeExtantPathError,
    MissingNotetypeError,
    MissingFieldOrdinalError,
    MissingNoteIdError,
    ExpectedNonexistentPathError,
    UnPushedPathWarning,
    DeletedFileNotFoundWarning,
    DiffTargetFileNotFoundWarning,
    NotetypeKeyError,
    UnnamedNotetypeError,
    NoteFieldKeyError,
    PathCreationCollisionError,
    ExpectedDirectoryButGotFileError,
    GitHeadRefNotFoundError,
    MissingMediaDirectoryError,
    MissingMediaFileWarning,
)
from ki.monadic import monadic
from ki.transformer import FlatNote, NoteTransformer


# pylint:disable=unnecessary-pass, too-many-lines


TEST_DATA_PATH = "tests/data/"
COLLECTIONS_PATH = os.path.join(TEST_DATA_PATH, "collections/")
COLLECTION_FILENAME = "collection.anki2"
ORIG_COLLECTION_FILENAME = "original.anki2"
EDITED_COLLECTION_FILENAME = "edited.anki2"
MULTIDECK_COLLECTION_FILENAME = "multideck.anki2"
HTML_COLLECTION_FILENAME = "html.anki2"
MEDIA_COLLECTION_FILENAME = "media.anki2"
MEDIA_MEDIA_DIRNAME = MEDIA_COLLECTION_FILENAME.split(".")[0] + ".media"
MEDIA_MEDIA_DB_FILENAME = MEDIA_COLLECTION_FILENAME.split(".")[0] + ".media.db2"


COLLECTION_PATH = os.path.abspath(
    os.path.join(COLLECTIONS_PATH, ORIG_COLLECTION_FILENAME)
)
EDITED_COLLECTION_PATH = os.path.abspath(
    os.path.join(COLLECTIONS_PATH, EDITED_COLLECTION_FILENAME)
)
MULTIDECK_COLLECTION_PATH = os.path.abspath(
    os.path.join(COLLECTIONS_PATH, MULTIDECK_COLLECTION_FILENAME)
)
HTML_COLLECTION_PATH = os.path.abspath(
    os.path.join(COLLECTIONS_PATH, HTML_COLLECTION_FILENAME)
)
MEDIA_COLLECTION_PATH = os.path.abspath(
    os.path.join(COLLECTIONS_PATH, MEDIA_COLLECTION_FILENAME)
)
MEDIA_MEDIA_DIRECTORY_PATH = os.path.abspath(
    os.path.join(COLLECTIONS_PATH, MEDIA_MEDIA_DIRNAME)
)
MEDIA_MEDIA_DB_PATH = os.path.abspath(
    os.path.join(COLLECTIONS_PATH, MEDIA_MEDIA_DB_FILENAME)
)

GITREPO_PATH = os.path.abspath(os.path.join(TEST_DATA_PATH, "repos/", "original/"))
MULTI_GITREPO_PATH = os.path.join(TEST_DATA_PATH, "repos/", "multideck/")
REPODIR = os.path.splitext(COLLECTION_FILENAME)[0]
MULTIDECK_REPODIR = os.path.splitext(MULTIDECK_COLLECTION_FILENAME)[0]
HTML_REPODIR = os.path.splitext(HTML_COLLECTION_FILENAME)[0]
MEDIA_REPODIR = os.path.splitext(MEDIA_COLLECTION_FILENAME)[0]
MULTI_NOTE_PATH = "aa/bb/cc/cc.md"

NOTES_PATH = os.path.abspath(os.path.join(TEST_DATA_PATH, "notes/"))
MEDIA_PATH = os.path.abspath(os.path.join(TEST_DATA_PATH, "media/"))
SUBMODULE_DIRNAME = "submodule"

MEDIA_FILENAME = "bullhorn-lg.png"
MEDIA_FILE_PATH = os.path.join(MEDIA_PATH, MEDIA_FILENAME)

NOTE_0 = "Default/a.md"
NOTE_1 = "Default/f.md"
NOTE_2 = "note123412341234.md"
NOTE_3 = "note 3.md"
NOTE_4 = "Default/c.md"
NOTE_5 = "alpha_nid.md"
NOTE_6 = "no_nid.md"
MEDIA_NOTE = "air.md"

NOTE_0_PATH = os.path.join(NOTES_PATH, NOTE_0)
NOTE_1_PATH = os.path.join(NOTES_PATH, NOTE_1)
NOTE_2_PATH = os.path.join(NOTES_PATH, NOTE_2)
NOTE_3_PATH = os.path.join(NOTES_PATH, NOTE_3)
NOTE_4_PATH = os.path.join(NOTES_PATH, NOTE_4)
NOTE_5_PATH = os.path.join(NOTES_PATH, NOTE_5)
NOTE_6_PATH = os.path.join(NOTES_PATH, NOTE_6)
MEDIA_NOTE_PATH = os.path.join(NOTES_PATH, MEDIA_NOTE)

NOTE_0_ID = 1645010162168
NOTE_4_ID = 1645027705329

# A models dictionary mapping model ids to notetype dictionaries.
# Note that the `flds` field has been removed.
MODELS = {
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
NAMELESS_MODELS = {
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

# HELPER FUNCTIONS


def invoke(*args, **kwargs):
    """Wrap click CliRunner invoke()."""
    return CliRunner().invoke(*args, **kwargs)


@beartype
def clone(runner: CliRunner, collection: ExtantFile, directory: str = "") -> str:
    """Make a test `ki clone` call."""
    res = runner.invoke(
        ki.ki,
        ["clone", str(collection), str(directory)],
        standalone_mode=False,
        catch_exceptions=False,
    )
    if isinstance(res.return_value, Err):
        raise res.return_value.unwrap_err()
    return res.output


@beartype
def pull(runner: CliRunner) -> str:
    """Make a test `ki pull` call."""
    res = runner.invoke(ki.ki, ["pull"], standalone_mode=False, catch_exceptions=False)
    if isinstance(res.return_value, Err):
        raise res.return_value.unwrap_err()
    return res.output


@beartype
def push(runner: CliRunner) -> str:
    """Make a test `ki push` call."""
    res = runner.invoke(ki.ki, ["push"], standalone_mode=False, catch_exceptions=False)
    if isinstance(res.return_value, Err):
        raise res.return_value.unwrap_err()
    return res.output


@beartype
def get_col_file() -> ExtantFile:
    """Put `collection.anki2` in a tempdir and return its abspath."""
    # Copy collection to tempdir.
    tempdir = tempfile.mkdtemp()
    col_file = os.path.abspath(os.path.join(tempdir, COLLECTION_FILENAME))
    shutil.copyfile(COLLECTION_PATH, col_file)
    return F.test(Path(col_file))


@beartype
def get_multideck_col_file() -> ExtantFile:
    """Put `multideck.anki2` in a tempdir and return its abspath."""
    # Copy collection to tempdir.
    tempdir = tempfile.mkdtemp()
    col_file = os.path.abspath(os.path.join(tempdir, MULTIDECK_COLLECTION_FILENAME))
    shutil.copyfile(MULTIDECK_COLLECTION_PATH, col_file)
    return F.test(Path(col_file))


@beartype
def get_html_col_file() -> ExtantFile:
    """Put `html.anki2` in a tempdir and return its abspath."""
    # Copy collection to tempdir.
    tempdir = tempfile.mkdtemp()
    col_file = os.path.abspath(os.path.join(tempdir, HTML_COLLECTION_FILENAME))
    shutil.copyfile(HTML_COLLECTION_PATH, col_file)
    return F.test(Path(col_file))


@beartype
def get_media_col_file() -> ExtantFile:
    """Put `media.anki2` in a tempdir and return its abspath."""
    # Copy collection to tempdir.
    tempdir = tempfile.mkdtemp()
    col_file = os.path.abspath(os.path.join(tempdir, MEDIA_COLLECTION_FILENAME))
    media_dir = os.path.abspath(os.path.join(tempdir, MEDIA_MEDIA_DIRNAME))
    media_db = os.path.abspath(os.path.join(tempdir, MEDIA_MEDIA_DB_FILENAME))
    shutil.copyfile(MEDIA_COLLECTION_PATH, col_file)
    shutil.copytree(MEDIA_MEDIA_DIRECTORY_PATH, media_dir)
    shutil.copyfile(MEDIA_MEDIA_DB_PATH, media_db)
    return F.test(Path(col_file))


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
def randomly_swap_1_bit(path: ExtantFile) -> None:
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
    shutil.rmtree(os.path.join(repodir, ".git/"))
    checksum = checksumdir.dirhash(repodir)
    shutil.rmtree(tempdir)
    return checksum


@beartype
def get_notes(collection: ExtantFile) -> List[ColNote]:
    """Get a list of notes from a path."""
    cwd: ExtantDir = F.cwd()
    col = Collection(collection)
    F.chdir(cwd)

    notes: List[ColNote] = []
    for nid in set(col.find_notes("")):
        colnote: OkErr = get_colnote(col, nid)
        if colnote.is_err():
            raise colnote
        colnote: ColNote = colnote.unwrap()
        notes.append(colnote)

    return notes


@beartype
def get_repo_with_submodules(runner: CliRunner, col_file: ExtantFile) -> git.Repo:
    """Return repo with committed submodule."""
    # Clone collection in cwd.
    clone(runner, col_file)
    repo = git.Repo(REPODIR)
    cwd = F.cwd()
    os.chdir(REPODIR)

    # Create submodule out of GITREPO_PATH.
    submodule_name = SUBMODULE_DIRNAME
    shutil.copytree(GITREPO_PATH, submodule_name)
    git.Repo.init(submodule_name, initial_branch=BRANCH_NAME)
    sm = git.Repo(submodule_name)
    sm.git.add(all=True)
    _ = sm.index.commit("Initial commit.")

    # Add as a submodule.
    repo.git.submodule("add", Path(submodule_name).resolve())
    repo.git.add(all=True)
    _ = repo.index.commit("Add submodule.")

    # Go back to the original current working directory.
    os.chdir(cwd)

    return repo


# UTILS


def test_parse_markdown_note():
    """Does ki raise an error when it fails to parse nid?"""
    # Read grammar.
    # UNSAFE! Should we assume this always exists? A nice error message should
    # be printed on initialization if the grammar file is missing. No
    # computation should be done, and none of the click commands should work.
    grammar_path = Path(ki.__file__).resolve().parent / "grammar.lark"
    grammar = grammar_path.read_text(encoding="UTF-8")

    # Instantiate parser.
    parser = Lark(grammar, start="file", parser="lalr")
    transformer = NoteTransformer()

    with pytest.raises(UnexpectedToken):
        parse_markdown_note(parser, transformer, F.test(Path(NOTE_5_PATH)))
    with pytest.raises(UnexpectedToken):
        parse_markdown_note(parser, transformer, F.test(Path(NOTE_6_PATH)))


def test_get_batches():
    """Does it get batches from a list of strings?"""
    runner = CliRunner()
    with runner.isolated_filesystem():
        root = F.cwd()
        one = F.touch(root, "note1.md")
        two = F.touch(root, "note2.md")
        three = F.touch(root, "note3.md")
        four = F.touch(root, "note4.md")
        batches = list(F.get_batches([one, two, three, four], n=2))
        assert batches == [[one, two], [three, four]]


def test_is_anki_note():
    """Do the checks in ``is_anki_note()`` actually do anything?"""
    runner = CliRunner()
    with runner.isolated_filesystem():
        root = F.cwd()
        mda = F.touch(root, "note.mda")
        amd = F.touch(root, "note.amd")
        mdtxt = F.touch(root, "note.mdtxt")
        nd = F.touch(root, "note.nd")

        assert is_anki_note(mda) is False
        assert is_anki_note(amd) is False
        assert is_anki_note(mdtxt) is False
        assert is_anki_note(nd) is False

        note_file: ExtantFile = F.touch(root, "note.md")

        note_file.write_text("", encoding="UTF-8")
        assert is_anki_note(note_file) is False

        note_file.write_text("one line", encoding="UTF-8")
        assert is_anki_note(note_file) is False

        note_file.write_text("### Note\n## Note\n", encoding="UTF-8")
        assert is_anki_note(note_file) is False

        note_file.write_text("## Note\nnid: 00000000000000a\n", encoding="UTF-8")
        assert is_anki_note(note_file) is False

        note_file.write_text("## Note\nnid: 000000000000000\n", encoding="UTF-8")
        assert is_anki_note(note_file) is True


@beartype
def open_collection(col_file: ExtantFile) -> Collection:
    cwd: ExtantDir = F.cwd()
    col = Collection(col_file)
    F.chdir(cwd)
    return col


def test_update_note_raises_error_on_too_few_fields():
    """Do we raise an error when the field names don't match up?"""
    col = open_collection(get_col_file())
    note = col.get_note(set(col.find_notes("")).pop())
    field = "data"

    # Note that "Back" field is missing.
    flatnote = FlatNote("title", 0, "Basic", "Default", [], False, {"Front": field})
    notetype: Notetype = parse_notetype_dict(note.note_type())
    res: OkErr = update_note(note, flatnote, notetype, notetype)
    warning: Warning = res.unwrap_err()
    assert isinstance(warning, Warning)
    assert isinstance(warning, NoteFieldValidationWarning)
    assert "Wrong number of fields for model Basic!" in str(warning)


def test_update_note_raises_error_on_too_many_fields():
    """Do we raise an error when the field names don't match up?"""
    col = open_collection(get_col_file())
    note = col.get_note(set(col.find_notes("")).pop())
    field = "data"

    # Note that "Left" field is extra.
    fields = {"Front": field, "Back": field, "Left": field}
    flatnote = FlatNote("title", 0, "Basic", "Default", [], False, fields)

    notetype: Notetype = parse_notetype_dict(note.note_type())
    res: OkErr = update_note(note, flatnote, notetype, notetype)
    warning: Warning = res.unwrap_err()
    assert isinstance(warning, Warning)
    assert isinstance(warning, NoteFieldValidationWarning)
    assert "Wrong number of fields for model Basic!" in str(warning)


def test_update_note_raises_error_wrong_field_name():
    """Do we raise an error when the field names don't match up?"""
    col = open_collection(get_col_file())
    note = col.get_note(set(col.find_notes("")).pop())
    field = "data"

    # Field `Backus` has wrong name, should be `Back`.
    fields = {"Front": field, "Backus": field}
    flatnote = FlatNote("title", 0, "Basic", "Default", [], False, fields)

    notetype: Notetype = parse_notetype_dict(note.note_type())
    res: OkErr = update_note(note, flatnote, notetype, notetype)
    warning: Warning = res.unwrap_err()
    assert isinstance(warning, Warning)
    assert isinstance(warning, NoteFieldValidationWarning)
    assert "Inconsistent field names" in str(warning)
    assert "Backus" in str(warning)
    assert "Back" in str(warning)


def test_update_note_sets_tags():
    """Do we update tags of anki note?"""
    col = open_collection(get_col_file())
    note = col.get_note(set(col.find_notes("")).pop())
    field = "data"

    fields = {"Front": field, "Back": field}
    flatnote = FlatNote("", 0, "Basic", "Default", ["tag"], False, fields)

    assert note.tags == []
    notetype: Notetype = parse_notetype_dict(note.note_type())
    update_note(note, flatnote, notetype, notetype).unwrap()
    assert note.tags == ["tag"]


def test_update_note_sets_deck():
    col = open_collection(get_col_file())
    note = col.get_note(set(col.find_notes("")).pop())
    field = "data"

    fields = {"Front": field, "Back": field}
    flatnote = FlatNote("title", 0, "Basic", "deck", [], False, fields)

    # TODO: Remove implicit assumption that all cards are in the same deck, and
    # work with cards instead of notes.
    deck = col.decks.name(note.cards()[0].did)
    assert deck == "Default"
    notetype: Notetype = parse_notetype_dict(note.note_type())
    update_note(note, flatnote, notetype, notetype).unwrap()
    deck = col.decks.name(note.cards()[0].did)
    assert deck == "deck"


def test_update_note_sets_field_contents():
    col = open_collection(get_col_file())
    note = col.get_note(set(col.find_notes("")).pop())

    field = "TITLE\ndata"
    fields = {"Front": field, "Back": field}
    flatnote = FlatNote("title", 0, "Basic", "Default", [], True, fields)

    assert "TITLE" not in note.fields[0]

    notetype: Notetype = parse_notetype_dict(note.note_type())
    update_note(note, flatnote, notetype, notetype).unwrap()

    assert "TITLE" in note.fields[0]
    assert "</p>" in note.fields[0]


def test_update_note_removes_field_contents():
    col = open_collection(get_col_file())
    note = col.get_note(set(col.find_notes("")).pop())

    field = "y"
    fields = {"Front": field, "Back": field}
    flatnote = FlatNote("title", 0, "Basic", "Default", [], False, fields)

    assert "a" in note.fields[0]
    notetype: Notetype = parse_notetype_dict(note.note_type())
    update_note(note, flatnote, notetype, notetype).unwrap()
    assert "a" not in note.fields[0]


def test_update_note_raises_error_on_nonexistent_notetype_name():
    col = open_collection(get_col_file())
    note = col.get_note(set(col.find_notes("")).pop())

    field = "data"
    fields = {"Front": field, "Back": field}
    flatnote = FlatNote("title", 0, "Nonexistent", "Default", [], False, fields)

    notetype: Notetype = parse_notetype_dict(note.note_type())
    res: OkErr = update_note(note, flatnote, notetype, notetype)
    error: Exception = res.unwrap_err()
    assert isinstance(error, Exception)
    assert isinstance(error, NotetypeMismatchError)


def test_display_fields_health_warning_catches_missing_clozes(capfd):
    col = open_collection(get_col_file())
    note = col.get_note(set(col.find_notes("")).pop())

    field = "data"
    fields = {"Text": field, "Back Extra": ""}
    flatnote = FlatNote("title", 0, "Cloze", "Default", [], False, fields)

    clz: NotetypeDict = col.models.by_name("Cloze")
    cloze: Notetype = parse_notetype_dict(clz)
    notetype: Notetype = parse_notetype_dict(note.note_type())
    res: OkErr = update_note(note, flatnote, notetype, cloze)
    warning = res.unwrap_err()
    assert isinstance(warning, Exception)
    assert isinstance(warning, UnhealthyNoteWarning)

    captured = capfd.readouterr()
    assert "unknown error code" in captured.err


def test_update_note_changes_notetype():
    col = open_collection(get_col_file())
    note = col.get_note(set(col.find_notes("")).pop())

    field = "data"
    fields = {"Front": field, "Back": field}
    flatnote = FlatNote(
        "title", 0, "Basic (and reversed card)", "Default", [], False, fields
    )

    rev: NotetypeDict = col.models.by_name("Basic (and reversed card)")
    reverse: Notetype = parse_notetype_dict(rev)
    notetype: Notetype = parse_notetype_dict(note.note_type())
    res: OkErr = update_note(note, flatnote, notetype, reverse)
    res.unwrap()


def test_display_fields_health_warning_catches_empty_notes():
    col = open_collection(get_col_file())
    note = col.get_note(set(col.find_notes("")).pop())

    note.fields = []
    health = display_fields_health_warning(note)
    assert health == 1


def test_slugify_filters_unicode_when_asked():
    text = "\u1234"
    result = F.slugify(text, allow_unicode=False)

    # Filter out Ethiopian syllable see.
    assert result == ""


def test_slugify_handles_unicode():
    """Test that slugify handles unicode alphanumerics."""
    # Hiragana should be okay.
    text = "ゅ"
    result = F.slugify(text, allow_unicode=True)
    assert result == text

    # Emojis as well.
    text = "😶"
    result = F.slugify(text, allow_unicode=True)
    assert result == text


def test_slugify_handles_html_tags():
    text = '<img src="card11front.jpg" />'
    result = F.slugify(text, allow_unicode=True)

    assert result == "img-srccard11frontjpg"


def test_get_note_path_produces_nonempty_filenames():
    field_text = '<img src="card11front.jpg" />'
    runner = CliRunner()
    with runner.isolated_filesystem():
        deck_dir: ExtantDir = F.force_mkdir(Path("a"))

        path: ExtantFile = get_note_path(field_text, deck_dir)
        assert os.path.isfile(path)
        assert path.name == "img-srccard11frontjpg.md"

        # Check that it even works if the field is empty.
        path: ExtantFile = get_note_path("", deck_dir)
        assert os.path.isfile(path)


def test_update_note_converts_markdown_formatting_to_html():
    col = open_collection(get_col_file())
    note = col.get_note(set(col.find_notes("")).pop())

    # We MUST pass markdown=True to the FlatNote constructor, or else this will
    # not work.
    field = "*hello*"
    fields = {"Front": field, "Back": field}
    flatnote = FlatNote("title", 0, "Basic", "Default", [], True, fields)

    assert "a" in note.fields[0]
    notetype: Notetype = parse_notetype_dict(note.note_type())
    update_note(note, flatnote, notetype, notetype).unwrap()
    assert "<em>hello</em>" in note.fields[0]


@beartype
@dataclass(frozen=True)
class DiffReposArgs:
    a_repo: git.Repo
    b_repo: git.Repo
    head_1: RepoRef
    filter_fn: Callable
    parser: Lark
    transformer: NoteTransformer


@beartype
def get_diff_repos_args() -> DiffReposArgs:
    """
    A test 'fixture' (not really a pytest fixture, but a setup function) to be
    called when we need to test `diff_repos()`.

    Basically a section of the code from `push()`, but without any error
    handling, since we expect things to work out nicely, and for the
    repositories operated upon during tests to be valid.

    Returns the values needed to pass as arguments to `diff_repos()`.

    This makes ephemeral repositories, so we should make any changes we expect
    to see the results of in `deltas: List[Delta]` *before* calling this
    function. For example, if we wanted to add a note, and then expected to see
    a `GitChangeType.ADDED`, then we should do that in `REPODIR` before calling
    this function.
    """

    # Check that we are inside a ki repository, and get the associated collection.
    cwd: ExtantDir = F.cwd()
    kirepo: KiRepo = M.kirepo(cwd).unwrap()
    lock(kirepo)
    md5sum: str = F.md5(kirepo.col_file)

    # Get reference to HEAD of current repo.
    head: KiRepoRef = M.head_kirepo_ref(kirepo).unwrap()

    # Copy current kirepo into a temp directory (the STAGE), hard reset to HEAD.
    stage_kirepo: KiRepo = get_ephemeral_kirepo(STAGE_SUFFIX, head, md5sum).unwrap()
    stage_kirepo = flatten_staging_repo(stage_kirepo, kirepo).unwrap()

    # This statement cannot be any farther down because we must get a reference
    # to HEAD *before* we commit, and then after the following line, the
    # reference we got will be HEAD~1, hence the variable name.
    head_1: RepoRef = M.head_repo_ref(stage_kirepo.repo).unwrap()

    stage_kirepo.repo.git.add(all=True)
    stage_kirepo.repo.index.commit(f"Pull changes from ref {head.sha}")

    # Get filter function.
    filter_fn = functools.partial(filter_note_path, patterns=IGNORE, root=kirepo.root)

    # Read grammar.
    # TODO:! Should we assume this always exists? A nice error message should
    # be printed on initialization if the grammar file is missing. No
    # computation should be done, and none of the click commands should work.
    grammar_path = Path(ki.__file__).resolve().parent / "grammar.lark"
    grammar = grammar_path.read_text(encoding="UTF-8")

    # Instantiate parser.
    parser = Lark(grammar, start="file", parser="lalr")
    transformer = NoteTransformer()

    # Get deltas.
    a_repo: git.Repo = get_ephemeral_repo(DELETED_SUFFIX, head_1, md5sum)
    b_repo: git.Repo = head_1.repo

    return DiffReposArgs(a_repo, b_repo, head_1, filter_fn, parser, transformer)


def test_diff_repos_shows_no_changes_when_no_changes_have_been_made(capfd, tmp_path):
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):

        # Clone collection in cwd.
        clone(runner, col_file)
        os.chdir(REPODIR)

        args: DiffReposArgs = get_diff_repos_args()
        deltas: List[Delta] = diff_repos(
            args.a_repo,
            args.b_repo,
            args.head_1,
            args.filter_fn,
            args.parser,
            args.transformer,
        ).unwrap()

        changed = [str(delta.path) for delta in deltas]
        captured = capfd.readouterr()
        assert changed == []
        assert "last_push" not in captured.err


def test_diff_repos_yields_a_warning_when_a_deleted_file_cannot_be_found(tmp_path):
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):

        # Clone collection in cwd.
        clone(runner, col_file)
        os.chdir(REPODIR)

        os.remove("Default/a.md")
        repo = git.Repo(".")
        repo.git.add(all=True)
        repo.index.commit("CommitMessage")

        args: DiffReposArgs = get_diff_repos_args()

        # We pass in `b_repo` for both arguments in order to simulate the
        # deleted file being missing.
        deltas: List[Union[Delta, Warning]] = diff_repos(
            args.b_repo,
            args.b_repo,
            args.head_1,
            args.filter_fn,
            args.parser,
            args.transformer,
        ).unwrap()
        warnings = [d for d in deltas if isinstance(d, DeletedFileNotFoundWarning)]
        assert len(warnings) == 1
        warning = warnings.pop()
        assert "Default/a.md" in str(warning)


def test_diff_repos_yields_a_warning_when_a_file_cannot_be_found(tmp_path):
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):

        # Clone collection in cwd.
        clone(runner, col_file)
        os.chdir(REPODIR)

        shutil.copyfile(NOTE_2_PATH, NOTE_2)
        repo = git.Repo(".")
        repo.git.add(all=True)
        repo.index.commit("CommitMessage")

        args: DiffReposArgs = get_diff_repos_args()

        os.remove(Path(args.b_repo.working_dir) / NOTE_2)

        deltas: List[Union[Delta, Warning]] = diff_repos(
            args.b_repo,
            args.b_repo,
            args.head_1,
            args.filter_fn,
            args.parser,
            args.transformer,
        ).unwrap()
        warnings = [d for d in deltas if isinstance(d, DiffTargetFileNotFoundWarning)]
        assert len(warnings) == 1
        warning = warnings.pop()
        assert "note123412341234.md" in str(warning)


def test_unsubmodule_repo_removes_gitmodules(tmp_path):
    """
    When you have a ki repo with submodules, does calling
    `unsubmodule_repo()`on it remove them? We test this by checking if the
    `.gitmodules` file exists.
    """
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        repo = get_repo_with_submodules(runner, col_file)
        gitmodules_path = Path(repo.working_dir) / ".gitmodules"
        assert gitmodules_path.exists()
        unsubmodule_repo(repo)
        assert not gitmodules_path.exists()


def test_diff_repos_handles_submodules():
    """
    Does 'diff_repos()' correctly generate deltas
    when adding submodules and when removing submodules?
    """
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():
        repo = get_repo_with_submodules(runner, col_file)

        os.chdir(REPODIR)

        args: DiffReposArgs = get_diff_repos_args()
        deltas: List[Delta] = diff_repos(
            args.a_repo,
            args.b_repo,
            args.head_1,
            args.filter_fn,
            args.parser,
            args.transformer,
        ).unwrap()

        assert len(deltas) == 1
        delta = deltas[0]
        assert delta.status == GitChangeType.ADDED
        assert "submodule/Default/a.md" in str(delta.path)

        # Push changes.
        push(runner)

        # Remove submodule.
        shutil.rmtree(SUBMODULE_DIRNAME)
        repo.git.add(all=True)
        _ = repo.index.commit("Remove submodule.")

        args: DiffReposArgs = get_diff_repos_args()
        deltas: List[Delta] = diff_repos(
            args.a_repo,
            args.b_repo,
            args.head_1,
            args.filter_fn,
            args.parser,
            args.transformer,
        ).unwrap()

        for delta in deltas:
            assert delta.path.is_file()


def test_backup_is_no_op_when_backup_already_exists(capfd):
    """Do we print a nice message when we backup an already-backed-up file?"""
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():
        clone(runner, col_file)
        os.chdir(REPODIR)
        kirepo: KiRepo = M.kirepo(F.cwd()).unwrap()
        backup(kirepo)
        backup(kirepo)
        captured = capfd.readouterr()
        assert "Backup already exists." in captured.out


def test_git_subprocess_pull():
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, col_file)

        # Edit collection.
        shutil.copyfile(EDITED_COLLECTION_PATH, col_file)
        os.chdir(REPODIR)

        # Pull, poorly.
        with pytest.raises(ValueError):
            git_subprocess_pull("anki", "main")


def test_get_note_path():
    """Do we add ordinals to generated filenames if there are duplicates?"""
    runner = CliRunner()
    with runner.isolated_filesystem():
        deck_dir = F.cwd()
        dupe_path = deck_dir / "a.md"
        dupe_path.write_text("ay")
        note_path = get_note_path("a", deck_dir)
        assert str(note_path.name) == "a_1.md"


def test_tidy_html_recursively():
    """Does tidy wrapper print a nice error when tidy is missing?"""
    runner = CliRunner()
    with runner.isolated_filesystem():
        root = F.cwd()
        file = root / "a.html"
        file.write_text("ay")
        old_path = os.environ["PATH"]
        try:
            os.environ["PATH"] = ""
            error = tidy_html_recursively(root, False).unwrap_err()
            assert isinstance(error, FileNotFoundError)
            assert "'tidy'" in str(error)
        finally:
            os.environ["PATH"] = old_path


def test_create_deck_dir():
    deckname = "aa::bb::cc"
    runner = CliRunner()
    with runner.isolated_filesystem():
        root = F.cwd()
        path = create_deck_dir(deckname, root)
        assert path.is_dir()
        assert os.path.isdir("aa/bb/cc")


def test_create_deck_dir_strips_leading_periods():
    deckname = ".aa::bb::.cc"
    runner = CliRunner()
    with runner.isolated_filesystem():
        root = F.cwd()
        path = create_deck_dir(deckname, root)
        assert path.is_dir()
        assert os.path.isdir("aa/bb/cc")


def test_get_note_payload():
    col = open_collection(get_col_file())
    note = col.get_note(set(col.find_notes("")).pop())
    colnote: ColNote = get_colnote(col, note.id).unwrap()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Field-note content id for the note returned by our `col.get_note()`
        # call above. Perhaps this should not be hardcoded.
        fid = "1645010162168front"

        # We use the fid as the filename as well for convenience, but there is
        # no reason this must be the case.
        path = Path(fid)

        # Dump content to the file. This represents our tidied HTML source.
        heyoo = "HEYOOOOO"
        path.write_text(heyoo, encoding="UTF-8")

        result = get_note_payload(colnote, {fid: path})

        # Check that the dumped `front` field content is there.
        assert heyoo in result

        # The `back` field content.
        assert "\nb\n" in result


def test_write_repository_generates_deck_tree_correctly():
    """Does generated FS tree match example collection?"""
    true_note_path = os.path.abspath(os.path.join(MULTI_GITREPO_PATH, MULTI_NOTE_PATH))
    cloned_note_path = os.path.join(MULTIDECK_REPODIR, MULTI_NOTE_PATH)
    col_file = get_multideck_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        targetdir = F.test(Path(MULTIDECK_REPODIR))
        targetdir = F.mkdir(targetdir)
        ki_dir = F.mkdir(F.test(Path(MULTIDECK_REPODIR) / KI))
        media_dir = F.mkdir(F.test(Path(MULTIDECK_REPODIR) / MEDIA))
        leaves: OkErr = F.fmkleaves(
            ki_dir,
            files={CONFIG_FILE: CONFIG_FILE, LAST_PUSH_FILE: LAST_PUSH_FILE},
            dirs={BACKUPS_DIR: BACKUPS_DIR, NO_SM_DIR: NO_SM_DIR},
        )

        write_repository(col_file, targetdir, leaves, media_dir, silent=False)

        # Check that deck directory is created and all subdirectories.
        assert os.path.isdir(os.path.join(MULTIDECK_REPODIR, "Default"))
        assert os.path.isdir(os.path.join(MULTIDECK_REPODIR, "aa/bb/cc"))
        assert os.path.isdir(os.path.join(MULTIDECK_REPODIR, "aa/dd"))

        # Compute hashes.
        cloned_md5 = F.md5(ExtantFile(cloned_note_path))
        true_md5 = F.md5(ExtantFile(true_note_path))

        assert cloned_md5 == true_md5


def test_write_repository_handles_html():
    """Does generated repo handle html okay?"""
    col_file = get_html_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        targetdir = F.mkdir(F.test(Path(HTML_REPODIR)))
        ki_dir = F.mkdir(F.test(Path(HTML_REPODIR) / KI))
        media_dir = F.mkdir(F.test(Path(MULTIDECK_REPODIR) / MEDIA))
        leaves: OkErr = F.fmkleaves(
            ki_dir,
            files={CONFIG_FILE: CONFIG_FILE, LAST_PUSH_FILE: LAST_PUSH_FILE},
            dirs={BACKUPS_DIR: BACKUPS_DIR, NO_SM_DIR: NO_SM_DIR},
        )
        write_repository(col_file, targetdir, leaves, media_dir, silent=False).unwrap()

        note_file = targetdir / "Default" / "あだ名.md"
        contents: str = note_file.read_text()

        # The tidy call should add the DOCTYPE tag and indentation.
        assert "<!DOCTYPE html>\n<title></title>" in contents
        assert '<div class="word-card">\n  <table class="kanji-match">' in contents


@beartype
def test_write_repository_propogates_errors_from_get_colnote(mocker: MockerFixture):
    """Do errors get forwarded nicdely?"""
    col_file = get_html_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        targetdir = F.mkdir(F.test(Path(HTML_REPODIR)))
        ki_dir = F.mkdir(F.test(Path(HTML_REPODIR) / KI))
        media_dir = F.mkdir(F.test(Path(MULTIDECK_REPODIR) / MEDIA))
        leaves: OkErr = F.fmkleaves(
            ki_dir,
            files={CONFIG_FILE: CONFIG_FILE, LAST_PUSH_FILE: LAST_PUSH_FILE},
            dirs={BACKUPS_DIR: BACKUPS_DIR, NO_SM_DIR: NO_SM_DIR},
        )

        mocker.patch(
            "ki.get_colnote", return_value=Err(NoteFieldKeyError("'bad_field_key'", 0))
        )
        error: Exception = write_repository(
            col_file, targetdir, leaves, media_dir, silent=False
        ).unwrap_err()
        assert isinstance(error, Exception)
        assert isinstance(error, NoteFieldKeyError)
        assert "'bad_field_key'" in str(error)


def test_maybe_kirepo_displays_nice_errors(tmp_path):
    """Does a nice error get printed when kirepo metadata is missing?"""
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):

        # Case where `.ki/` directory doesn't exist.
        clone(runner, col_file)
        targetdir: ExtantDir = F.test(Path(REPODIR))
        shutil.rmtree(targetdir / KI)
        error: Exception = M.kirepo(targetdir).unwrap_err()
        assert "fatal: not a ki repository" in str(error)
        shutil.rmtree(targetdir)

        # Case where `.ki/` is a file instead of a directory.
        clone(runner, col_file)
        targetdir: ExtantDir = F.test(Path(REPODIR))
        shutil.rmtree(targetdir / KI)
        (targetdir / KI).touch()
        error: Exception = M.kirepo(targetdir).unwrap_err()
        assert "fatal: not a ki repository" in str(error)
        shutil.rmtree(targetdir)

        # Case where `.ki/backups` directory doesn't exist.
        clone(runner, col_file)
        targetdir: ExtantDir = F.test(Path(REPODIR))
        shutil.rmtree(targetdir / KI / BACKUPS_DIR)
        error: Exception = M.kirepo(targetdir).unwrap_err()
        assert "Directory not found" in str(error)
        assert "'.ki/backups'" in str(error)
        shutil.rmtree(targetdir)

        # Case where `.ki/backups` is a file instead of a directory.
        clone(runner, col_file)
        targetdir: ExtantDir = F.test(Path(REPODIR))
        shutil.rmtree(targetdir / KI / BACKUPS_DIR)
        (targetdir / KI / BACKUPS_DIR).touch()
        error: Exception = M.kirepo(targetdir).unwrap_err()
        assert "A directory was expected" in str(error)
        assert "'.ki/backups'" in str(error)
        shutil.rmtree(targetdir)

        # Case where `.ki/config` file doesn't exist.
        clone(runner, col_file)
        targetdir: ExtantDir = F.test(Path(REPODIR))
        os.remove(targetdir / KI / CONFIG_FILE)
        error: Exception = M.kirepo(targetdir).unwrap_err()
        assert "File not found" in str(error)
        assert "'.ki/config'" in str(error)
        shutil.rmtree(targetdir)

        # Case where `.ki/config` is a directory instead of a file.
        clone(runner, col_file)
        targetdir: ExtantDir = F.test(Path(REPODIR))
        os.remove(targetdir / KI / CONFIG_FILE)
        os.mkdir(targetdir / KI / CONFIG_FILE)
        error: Exception = M.kirepo(targetdir).unwrap_err()
        assert "A file was expected" in str(error)
        assert "'.ki/config'" in str(error)
        shutil.rmtree(targetdir)

        # Case where `.ki/hashes` file doesn't exist.
        clone(runner, col_file)
        targetdir: ExtantDir = F.test(Path(REPODIR))
        os.remove(targetdir / KI / HASHES_FILE)
        error: Exception = M.kirepo(targetdir).unwrap_err()
        assert "File not found" in str(error)
        assert "'.ki/hashes'" in str(error)
        shutil.rmtree(targetdir)

        # Case where `.ki/models` file doesn't exist.
        clone(runner, col_file)
        targetdir: ExtantDir = F.test(Path(REPODIR))
        os.remove(targetdir / MODELS_FILE)
        error: Exception = M.kirepo(targetdir).unwrap_err()
        assert "File not found" in str(error)
        assert "'models.json'" in str(error)
        shutil.rmtree(targetdir)

        # Case where collection file doesn't exist.
        clone(runner, col_file)
        targetdir: ExtantDir = F.test(Path(REPODIR))
        os.remove(col_file)
        error: Exception = M.kirepo(targetdir).unwrap_err()
        assert "File not found" in str(error)
        assert "'.anki2'" in str(error)
        assert "database" in str(error)
        assert "collection.anki2" in str(error)
        shutil.rmtree(targetdir)


def test_get_target(tmp_path):
    """Do we print a nice error when the targetdir is nonempty?"""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        os.mkdir("file")
        Path("file/subfile").touch()
        col_file: ExtantFile = F.touch(F.cwd(), "file.anki2")
        error: Exception = get_target(F.cwd(), col_file, "").unwrap_err()
        assert "fatal: destination path" in str(error)
        assert "file" in str(error)


def test_maybe_emptydir(tmp_path):
    """Do we print a nice error when the directory is unexpectedly nonempty?"""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path("file").touch()
        error: Exception = M.emptydir(F.cwd()).unwrap_err()
        assert isinstance(error, Exception)
        assert isinstance(error, ExpectedEmptyDirectoryButGotNonEmptyDirectoryError)
        assert "but it is nonempty" in str(error)
        assert str(Path.cwd()) in str(error)


def test_maybe_emptydir_handles_non_directories(tmp_path):
    """Do we print a nice error when the path is not a directory?"""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        file = Path("file")
        file.touch()
        error: Exception = M.emptydir(file).unwrap_err()
        assert isinstance(error, Exception)
        assert isinstance(error, ExpectedDirectoryButGotFileError)
        assert str(file) in str(error)


def test_maybe_xdir(tmp_path):
    """Do we print a nice error when there is a non-file non-directory thing?"""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        os.mkfifo("pipe")
        error: Exception = M.xdir(Path("pipe")).unwrap_err()
        assert isinstance(error, Exception)
        assert isinstance(error, StrangeExtantPathError)
        assert "pseudofile" in str(error)
        assert "pipe" in str(error)


def test_maybe_xfile(tmp_path):
    """Do we print a nice error when there is a non-file non-directory thing?"""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        os.mkfifo("pipe")
        error: Exception = M.xfile(Path("pipe")).unwrap_err()
        assert isinstance(error, Exception)
        assert isinstance(error, StrangeExtantPathError)
        assert "pseudofile" in str(error)
        assert "pipe" in str(error)


def test_push_flatnote_to_anki():
    """Do we print a nice error when a notetype is missing?"""
    col = open_collection(get_col_file())

    field = "data"
    fields = {"Front": field, "Back": field}
    flatnote = FlatNote("title", 0, "NonexistentModel", "Default", [], False, fields)
    error: Exception = push_flatnote_to_anki(col, flatnote).unwrap_err()
    assert isinstance(error, Exception)
    assert isinstance(error, MissingNotetypeError)
    assert "NonexistentModel" in str(error)


def test_maybe_head_repo_ref():
    runner = CliRunner()
    with runner.isolated_filesystem():
        repo = git.Repo.init("repo")
        error = M.head_repo_ref(repo).unwrap_err()
        assert isinstance(error, Exception)
        assert isinstance(error, GitHeadRefNotFoundError)
        assert "ValueError raised while trying to get ref 'HEAD' from repo" in str(
            error
        )


def test_maybe_head_kirepo_ref():
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Initialize the kirepo *without* committing.
        os.mkdir("collection")
        targetdir = F.test(Path("collection"))
        silent = False
        ki_dir: EmptyDir = F.mksubdir(targetdir, Path(KI))
        media_dir = F.mkdir(F.test(targetdir / MEDIA))
        leaves: Leaves = F.fmkleaves(
            ki_dir,
            files={CONFIG_FILE: CONFIG_FILE, LAST_PUSH_FILE: LAST_PUSH_FILE},
            dirs={BACKUPS_DIR: BACKUPS_DIR, NO_SM_DIR: NO_SM_DIR},
        ).unwrap()
        md5sum = F.md5(col_file)
        ignore_path = targetdir / GITIGNORE_FILE
        ignore_path.write_text(".ki/\n")
        write_repository(col_file, targetdir, leaves, media_dir, silent).unwrap()
        repo = git.Repo.init(targetdir, initial_branch=BRANCH_NAME)
        _ = git.Repo.init(leaves.dirs[NO_SM_DIR], initial_branch=BRANCH_NAME)
        append_md5sum(ki_dir, col_file.name, md5sum, silent)

        # Since we didn't commit, there will be no HEAD.
        kirepo = M.kirepo(F.test(Path(repo.working_dir)))
        error = M.head_kirepo_ref(kirepo).unwrap_err()
        assert isinstance(error, Exception)
        assert isinstance(error, GitHeadRefNotFoundError)
        assert "ValueError raised while trying to get ref 'HEAD' from repo" in str(
            error
        )


@beartype
def test_push_flatnote_to_anki_handles_note_key_errors(mocker: MockerFixture):
    """Do we print a nice error when a KeyError is raised on note[]?"""
    col = open_collection(get_col_file())

    field = "data"
    fields = {"Front": field, "Back": field}
    flatnote = FlatNote("title", 0, "Basic", "Default", [], False, fields)
    mocker.patch("anki.notes.Note.__getitem__", side_effect=KeyError("bad_field_key"))
    error: Exception = push_flatnote_to_anki(col, flatnote).unwrap_err()
    assert isinstance(error, Exception)
    assert isinstance(error, NoteFieldKeyError)
    assert "Expected field" in str(error)
    assert "'bad_field_key'" in str(error)


def test_parse_notetype_dict():
    nt = NT

    # This field ordinal doesn't exist.
    nt["sortf"] = 3
    error = parse_notetype_dict(nt).unwrap_err()
    assert isinstance(error, Exception)
    assert isinstance(error, MissingFieldOrdinalError)
    assert "3" in str(error)


def test_get_colnote_prints_nice_error_when_nid_doesnt_exist():
    col = open_collection(get_col_file())
    nid = 44444444444444444
    error = get_colnote(col, nid).unwrap_err()
    assert isinstance(error, Exception)
    assert isinstance(error, MissingNoteIdError)
    assert str(nid) in str(error)


@beartype
def test_get_colnote_propagates_errors_from_parse_notetype_dict(mocker: MockerFixture):
    col = open_collection(get_col_file())
    note = col.get_note(set(col.find_notes("")).pop())

    mocker.patch("ki.parse_notetype_dict", return_value=Err(UnnamedNotetypeError(NT)))
    error: Exception = get_colnote(col, note.id).unwrap_err()
    assert isinstance(error, UnnamedNotetypeError)
    assert "Failed to find 'name' field" in str(error)


@beartype
def test_get_colnote_propagates_errors_key_errors_from_sort_field(
    mocker: MockerFixture,
):
    mocker.patch("anki.notes.Note.__getitem__", side_effect=KeyError("bad_field_key"))
    col = open_collection(get_col_file())
    note = col.get_note(set(col.find_notes("")).pop())
    error: Exception = get_colnote(col, note.id).unwrap_err()
    assert isinstance(error, NoteFieldKeyError)
    assert "Expected field" in str(error)
    assert "'bad_field_key'" in str(error)


@monadic
@beartype
def sample() -> Result[int, Exception]:
    return Ok("")


@monadic
@beartype
def unannotated():
    return Ok(0)


def test_monadic_lift_decorator():
    """Does our `Result` lift decorator catch type errors?"""
    with pytest.raises(
        roar.BeartypeCallHintReturnViolation,
        match="not instance of int",
    ):
        sample()
    with pytest.raises(
        roar.BeartypeCallHintReturnViolation,
        match="return 0 unannotated",
    ):
        unannotated()


def test_nopath(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        file = Path("file")
        file.touch()
        error = M.nopath(file).unwrap_err()
        assert isinstance(error, FileExistsError)


def test_get_ephemeral_kirepo(tmp_path):
    """
    Do errors in `M.nopath()` call in `get_ephemeral_kirepo()` get forwarded to
    the caller and printed nicely?

    In `get_ephemeral_kirepo()`, we construct a `NoPath` for the `.ki`
    subdirectory, which doesn't exist yet at that point, because
    `get_ephemeral_repo()` is just a git clone operation, and the `.ki`
    subdirectory is in the `.gitignore` file. It is possible but
    extraordinarily improbable that this path is created in between the
    `Repo.clone_from()` call and the `M.nopath()` call.

    In this test function, we simulate this possibility by the deleting the
    `.gitignore` file and committing the `.ki` subdirectory.
    """
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        clone(runner, col_file)
        os.chdir(REPODIR)
        os.remove(".gitignore")
        kirepo: KiRepo = M.kirepo(F.cwd()).unwrap()
        kirepo.repo.git.add(all=True)
        kirepo.repo.index.commit("Add ki directory.")
        head: KiRepoRef = M.head_kirepo_ref(kirepo).unwrap()
        error: Exception = get_ephemeral_kirepo(
            Path("suffix"), head, md5sum="md5"
        ).unwrap_err()
        assert isinstance(error, FileExistsError)
        assert isinstance(error, ExpectedNonexistentPathError)
        assert ".ki" in str(error)


def test_filter_note_path(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        directory = Path("directory")
        directory.mkdir()
        path = directory / "file"
        path.touch()
        patterns: List[str] = ["directory"]
        root: ExtantDir = F.cwd()
        warning = filter_note_path(path, patterns, root).unwrap_err()
        assert isinstance(warning, Warning)
        assert isinstance(warning, UnPushedPathWarning)
        assert "directory/file" in str(warning)


def test_get_models_recursively(tmp_path):
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        clone(runner, col_file)
        os.chdir(REPODIR)
        os.remove("models.json")
        with open(Path("models.json"), "w", encoding="UTF-8") as models_f:
            json.dump(MODELS, models_f, ensure_ascii=False, indent=4)
        kirepo: KiRepo = M.kirepo(F.cwd()).unwrap()
        error = get_models_recursively(kirepo).unwrap_err()
        assert isinstance(error, Exception)
        assert isinstance(error, NotetypeKeyError)
        assert "not found in notetype" in str(error)
        assert "flds" in str(error)
        assert "Basic" in str(error)


def test_get_models_recursively_prints_a_nice_error_when_models_dont_have_a_name(
    tmp_path,
):
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        clone(runner, col_file)
        os.chdir(REPODIR)
        os.remove("models.json")
        with open(Path("models.json"), "w", encoding="UTF-8") as models_f:
            json.dump(NAMELESS_MODELS, models_f, ensure_ascii=False, indent=4)
        kirepo: KiRepo = M.kirepo(F.cwd()).unwrap()
        error = get_models_recursively(kirepo).unwrap_err()
        assert isinstance(error, Exception)
        assert isinstance(error, UnnamedNotetypeError)
        assert "Failed to find 'name' field" in str(error)
        assert "1645010146011" in str(error)


def test_get_ephemeral_repo_handles_submodules(tmp_path):
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        repo: git.Repo = get_repo_with_submodules(runner, col_file)

        os.chdir(repo.working_dir)

        # Edit a file within the submodule.
        file = Path(repo.working_dir) / SUBMODULE_DIRNAME / "Default" / "a.md"
        logger.debug(f"Adding 'z' to file '{file}'")
        with open(file, "a", encoding="UTF-8") as note_f:
            note_f.write("\nz\n")

        subrepo = git.Repo(Path(repo.working_dir) / SUBMODULE_DIRNAME)
        subrepo.git.add(all=True)
        subrepo.index.commit(".")
        repo.git.add(all=True)
        repo.index.commit(".")

        kirepo: KiRepo = M.kirepo(F.cwd()).unwrap()
        head: KiRepoRef = M.head_kirepo_ref(kirepo).unwrap()

        # Just want to check that this doesn't return an exception, so we
        # unwrap, but don't assert anything.
        kirepo = get_ephemeral_kirepo(Path("suffix"), head, md5sum="md5").unwrap()


def test_ftest_handles_strange_paths(tmp_path):
    """Do we print a nice error when there is a non-file non-directory thing?"""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        os.mkfifo("pipe")
        pipe = F.test(Path("pipe"))
        assert isinstance(pipe, ExtantStrangePath)


def test_fparent_handles_fs_root():
    parent = F.parent(F.test(Path("/")))
    assert isinstance(parent, ExtantDir)
    assert str(parent) == "/"


def test_fmkleaves_handles_collisions(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        root = F.test(F.cwd())
        error = F.fmkleaves(root, files={"a": "a", "b": "a"}).unwrap_err()
        assert isinstance(error, Exception)
        assert isinstance(error, PathCreationCollisionError)
        assert "'a'" in str(error)
        assert len(os.listdir(".")) == 0

        error = F.fmkleaves(root, files={"a": "a"}, dirs={"b": "a"}).unwrap_err()
        assert isinstance(error, Exception)
        assert isinstance(error, PathCreationCollisionError)
        assert "'a'" in str(error)
        assert len(os.listdir(".")) == 0

        error = F.fmkleaves(root, dirs={"a": "a", "b": "a"}).unwrap_err()
        assert isinstance(error, Exception)
        assert isinstance(error, PathCreationCollisionError)
        assert "'a'" in str(error)
        assert len(os.listdir(".")) == 0


def test_get_media_files_returns_nice_errors():
    """Does `get_media_files()` handle case where media directory doesn't exist?"""
    col_file: ExtantFile = get_media_col_file()
    col: Collection = open_collection(col_file)
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Remove the media directory.
        media_dir = col_file.parent / (str(col_file.stem) + ".media")
        shutil.rmtree(media_dir)

        nids = set(col.find_notes(query=""))
        error = get_media_files(col, nids).unwrap_err()
        assert isinstance(error, MissingMediaDirectoryError)
        assert "media.media" in str(error)
        assert "bad Anki collection media directory" in str(error)


def test_write_repository_displays_missing_media_warnings(capfd):
    col_file: ExtantFile = get_media_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        targetdir = F.test(Path(MEDIA_REPODIR))
        targetdir = F.mkdir(targetdir)
        ki_dir = F.mkdir(F.test(Path(MEDIA_REPODIR) / KI))
        media_dir = F.mkdir(F.test(Path(MEDIA_REPODIR) / MEDIA))
        leaves: OkErr = F.fmkleaves(
            ki_dir,
            files={CONFIG_FILE: CONFIG_FILE, LAST_PUSH_FILE: LAST_PUSH_FILE},
            dirs={BACKUPS_DIR: BACKUPS_DIR, NO_SM_DIR: NO_SM_DIR},
        )

        # Remove the contents of the media directory.
        col_media_dir = F.test(col_file.parent / (str(col_file.stem) + ".media"))
        for path in col_media_dir.iterdir():
            if path.is_file():
                os.remove(path)

        write_repository(col_file, targetdir, leaves, media_dir, silent=False).unwrap()

        captured = capfd.readouterr()
        assert "Missing or bad media file" in captured.out
        assert "media.media/1sec.mp3" in captured.out


def test_get_media_files_finds_notetype_media():
    """Does `get_media_files()` get files like `collection.media/_vonNeumann.jpg`?"""
    col_file: ExtantFile = get_media_col_file()
    col: Collection = open_collection(col_file)
    runner = CliRunner()
    with runner.isolated_filesystem():

        nids = set(col.find_notes(query=""))
        medias = get_media_files(col, nids).unwrap()
        media_files = {f for f in medias if isinstance(f, ExtantFile)}
        assert len(media_files) == 2
        media_files = sorted(list(media_files))
        von_neumann: ExtantFile = media_files[1]
        assert "_vonNeumann" in str(von_neumann)
