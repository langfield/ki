#!/usr/bin/env python3
"""Tests for ki command line interface (CLI)."""
import os
import gc
import sys
import time
import shutil
import sqlite3
import tempfile
import subprocess
from pathlib import Path
from distutils.dir_util import copy_tree
from importlib.metadata import version

import git
import pytest
from loguru import logger
from pytest_mock import MockerFixture

import anki
from anki.collection import Note, Collection

from beartype import beartype
from beartype.typing import List, Tuple, Optional, Union

import ki
import ki.functional as F
from ki import MEDIA, LCA, _clone1, do, stardo, get_guid, add_db_note
from ki.types import (
    Notetype,
    ColNote,
    Dir,
    File,
    TargetExistsError,
    NotKiRepoError,
    UpdatesRejectedError,
    SQLiteLockError,
    GitHeadRefNotFoundError,
    CollectionChecksumError,
    MissingFieldOrdinalError,
    AnkiAlreadyOpenError,
)
from ki.functional import curried
from tests.test_ki import (
    GITREPO_PATH,
    invoke,
    clone,
    pull,
    push,
    is_git_repo,
    randomly_swap_1_bit,
    checksum_git_repository,
    get_notes,
    get_test_collection,
    SampleCollection,
)


DATA = Path(__file__).parent / "data"
PARSE_NOTETYPE_DICT_CALLS_PRIOR_TO_FLATNOTE_PUSH = 2

# pylint: disable=unnecessary-pass, too-many-lines, invalid-name, duplicate-code
# pylint: disable=missing-function-docstring, too-many-locals, no-value-for-parameter
# pylint: disable=unused-argument


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
    model, fullnames, nid, fields = spec
    note = col.get_note(nid)
    assert len(note.keys()) == len(fields)
    stardo(lambda k, fld: note.__setitem__(k, fld), zip(note.keys(), fields))
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


# CLI


def test_bad_command_is_bad():
    """Typos should result in errors."""
    result = invoke(ki.ki, ["clome"])
    assert result.exit_code == 2
    assert "Error: No such command 'clome'." in result.output


def test_runas_module():
    """Can this package be run as a Python module?"""
    command = "python -m ki --help"
    completed = subprocess.run(command, shell=True, capture_output=True, check=True)
    assert completed.returncode == 0


def test_entrypoint():
    """Is entrypoint script installed? (setup.py)"""
    result = invoke(ki.ki, ["--help"])
    assert result.exit_code == 0


def test_version():
    """Does --version display information as expected?"""
    expected_version = version("ki")
    result = invoke(ki.ki, ["--version"])

    assert result.stdout.rstrip() == f"ki, version {expected_version}"
    assert result.exit_code == 0


def test_command_availability():
    """Are commands available?"""
    results = []
    results.append(invoke(ki.ki, ["clone", "--help"]))
    results.append(invoke(ki.ki, ["pull", "--help"]))
    results.append(invoke(ki.ki, ["push", "--help"]))
    for result in results:
        assert result.exit_code == 0


def test_cli():
    """Does CLI stop execution w/o a command argument?"""
    with pytest.raises(SystemExit):
        ki.ki()
        pytest.fail("CLI doesn't abort asking for a command argument")


# COMMON


def test_fails_without_ki_subdirectory():
    """Do pull and push know whether they're in a ki-generated git repo?"""
    tempdir = tempfile.mkdtemp()
    copy_tree(GITREPO_PATH, tempdir)
    os.chdir(tempdir)
    with pytest.raises(NotKiRepoError):
        pull()
    with pytest.raises(NotKiRepoError):
        push()


def test_computes_and_stores_md5sum():
    """Does ki add new hash to `.ki/hashes`?"""
    ORIGINAL: SampleCollection = get_test_collection("original")
    clone(ORIGINAL.col_file)

    # Check that hash is written.
    hashes = read(".ki/hashes")
    assert f"a68250f8ee3dc8302534f908bcbafc6a  {ORIGINAL.filename}" in hashes
    assert f"199216c39eeabe23a1da016a99ffd3e2  {ORIGINAL.filename}" not in hashes

    EDITED: SampleCollection = get_test_collection("edited")
    shutil.copyfile(EDITED.path, ORIGINAL.col_file)
    pull()

    # Check that edited hash is written and old hash is still there.
    hashes = read(".ki/hashes")
    assert f"a68250f8ee3dc8302534f908bcbafc6a  {ORIGINAL.filename}" in hashes
    assert f"199216c39eeabe23a1da016a99ffd3e2  {ORIGINAL.filename}" in hashes


def test_no_op_pull_push_cycle_is_idempotent():
    """Do pull/push not misbehave if you keep doing both?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])
    clone(a)
    assert pull() == "ki pull: up to date.\n\n"
    push()
    assert pull() == "ki pull: up to date.\n\n"
    push()
    assert pull() == "ki pull: up to date.\n\n"
    push()
    assert pull() == "ki pull: up to date.\n\n"
    push()


def test_output():
    """Does it print nice things?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])
    repo, _ = clone(a)
    edit(a, ("Basic", ["Default"], 1, ["aa", "bb"]))
    edit(a, ("Basic", ["Default"], 2, ["f", "g"]))
    pull()

    p = Path("Default/aa.md")
    assert p.is_file()
    append(p, "e\n")
    write_basic("Default", ("r", "s"))
    write_basic("Default", ("s", "t"))

    # Commit.
    repo.git.add(all=True)
    repo.index.commit("Added 'e'.")

    # Push changes.
    out = push()
    assert "Overwrote" in out


# CLONE


def test_clone_fails_if_collection_doesnt_exist():
    """Does ki clone only if `.anki2` file exists?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])
    os.remove(a)
    with pytest.raises(FileNotFoundError):
        clone(a)
    assert not a.is_dir()


def test_clone_fails_if_collection_is_already_open():
    """Does ki print a nice error message when Anki is accidentally left open?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])
    os.remove(a)
    _ = opencol(a)
    with pytest.raises(AnkiAlreadyOpenError):
        clone(a)


def test_clone_creates_directory():
    """Does it create the directory?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])
    clone(a)
    os.chdir("../")
    assert os.path.isdir("a")


def test_clone_errors_when_directory_is_populated():
    """Does it disallow overwrites?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])

    # Create directory where we want to clone.
    os.chdir(F.mkdtemp())
    os.mkdir("a")
    write("a/hi", "hi\n")

    # Should error out because directory already exists.
    with pytest.raises(TargetExistsError):
        _clone1(str(a))


def test_clone_cleans_up_on_error():
    """Does it clean up on nontrivial errors?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])
    clone(a)
    os.chdir("../")
    assert Path("a").is_dir()
    F.rmtree(F.chk(Path("a")))
    cwd = os.getcwd()
    old_path = os.environ["PATH"]
    try:
        with pytest.raises(git.GitCommandNotFound):
            os.environ["PATH"] = ""
            _clone1(str(a))
        assert os.getcwd() == cwd
        assert not Path("a").is_dir()
    finally:
        os.environ["PATH"] = old_path


def test_clone_clean_up_preserves_directories_that_exist_a_priori():
    """
    When clone fails and the cleanup function is called, does it not delete
    targetdirs that already existed?
    """
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])

    os.chdir(F.mkdtemp())
    os.mkdir("a")
    assert os.path.isdir("a")
    cwd = os.getcwd()
    old_path = os.environ["PATH"]
    try:
        with pytest.raises(git.GitCommandNotFound):
            os.environ["PATH"] = ""
            _clone1(str(a))
        assert os.getcwd() == cwd
        assert os.path.isdir("a")
        assert len(os.listdir("a")) == 0
    finally:
        os.environ["PATH"] = old_path


def test_clone_succeeds_when_directory_exists_but_is_empty():
    """Does it clone into empty directories?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])
    os.chdir(F.mkdtemp())
    os.mkdir("a")
    clone(a)


def test_clone_generates_expected_notes():
    """Do generated note files match content of an example collection?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])
    os.chdir(F.mkdtemp())
    clone(a)
    assert os.path.isdir("Default")
    assert (
        (Path("Default") / "a.md").read_text()
        == """# Note
```
guid: q/([o$8RAO
notetype: Basic
```

### Tags
```
```

## Front
a

## Back
b
"""
    )


def test_clone_generates_deck_tree_correctly():
    """Does generated FS tree match example collection?"""
    a: File = mkcol(
        [
            ("Basic", ["aa"], 1, ["a", "aa"]),
            ("Basic", ["aa::bb"], 2, ["bb", "bb"]),
            ("Basic", ["aa::bb::cc"], 3, ["cc", "cc"]),
            ("Basic", ["aa::dd"], 4, ["dd", "dd"]),
            ("Basic", ["Default"], 5, ["hello", "hello"]),
            ("Basic", ["Default"], 6, ["hello my enemy", "goodbye"]),
        ]
    )

    # Create empty decks.
    col = opencol(a)
    do(col.decks.id, [":a:::b:", "blank::blank", "blank::Hello"])
    col.close(save=True)

    os.chdir(F.mkdtemp())
    clone(a)
    assert os.path.isdir("Default")
    assert os.path.isdir("aa/bb/cc")
    assert os.path.isdir("aa/dd")
    assert (
        (Path("aa") / "bb" / "cc" / "cc.md").read_text()
        == """# Note
```
guid: (<hy(zm;W
notetype: Basic
```

### Tags
```
```

## Front
cc

## Back
cc
"""
    )


def test_clone_generates_ki_subdirectory():
    """Does clone command generate .ki/ directory?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])
    os.chdir(F.mkdtemp())
    clone(a)
    assert Path(".ki/").is_dir()


def test_cloned_collection_is_git_repository():
    """Does clone run `git init` and stuff?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])
    os.chdir(F.mkdtemp())
    clone(a)
    os.chdir("../")
    assert is_git_repo("a")


def test_clone_commits_directory_contents():
    """Does clone leave user with an up-to-date repo?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])
    os.chdir(F.mkdtemp())
    repo, _ = clone(a)
    changes = repo.head.commit.diff()
    commits = list(repo.iter_commits("HEAD"))
    assert len(changes) == 0 and len(commits) == 1


def test_clone_leaves_collection_file_unchanged():
    """Does clone leave the collection alone?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])
    os.chdir(F.mkdtemp())
    original_md5 = F.md5(a)
    clone(a)
    updated_md5 = F.md5(a)
    assert original_md5 == updated_md5


def test_clone_directory_argument_works():
    """Does clone obey the target directory argument?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])
    tempdir = tempfile.mkdtemp()
    target = os.path.join(tempdir, "TARGET")
    assert not os.path.isdir(target)
    assert not os.path.isfile(target)
    clone(a, target)
    assert os.path.isdir(target)


def test_clone_writes_media_files():
    """Does clone copy media files from the media directory into 'MEDIA'?"""
    a: File = mkcol([("Basic", ["Default"], 1, ["a", "b[sound:1sec.mp3]"])])
    col = opencol(a)
    col.media.add_file(DATA / "media/1sec.mp3")
    col.close(save=True)
    clone(a)
    assert (Path(MEDIA) / "1sec.mp3").is_file()


def test_clone_handles_cards_from_a_single_note_in_distinct_decks():
    c = ("Basic (and reversed card)", ["top::a", "top::b"], 1, ["a", "b"])
    a: File = mkcol([c])
    clone(a)
    two = Path("top/b/a_Card 2.md")
    orig = Path("top/a/a.md")
    assert os.path.islink(two)
    assert os.path.isfile(orig)


def test_clone_url_decodes_media_src_attributes():
    back = '<img src="Screenshot%202019-05-01%20at%2014.40.56.png">'
    a: File = mkcol([("Basic", ["Default"], 1, ["a", back])])
    clone(a)
    contents = read("Default/a.md")
    assert '<img src="Screenshot 2019-05-01 at 14.40.56.png">' in contents


def test_clone_leaves_no_working_tree_changes():
    """Does everything get committed at the end of a `clone()`?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])
    repo, _ = clone(a)
    assert not repo.is_dirty()


# PULL


def test_pull_fails_if_collection_no_longer_exists():
    """Does ki pull only if `.anki2` file exists?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])
    clone(a)
    os.remove(a)
    with pytest.raises(FileNotFoundError):
        pull()


def test_pull_fails_if_collection_file_is_corrupted():
    """Does `pull()` fail gracefully when the collection file is bad?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])
    clone(a)
    a.write_text("bad_contents")
    with pytest.raises(SQLiteLockError):
        pull()


def test_pull_writes_changes_correctly():
    """Does ki get the changes from modified collection file?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])
    clone(a)
    f = Path("Default") / "f.md"
    assert not f.exists()
    n1 = ("Basic", ["Default"], 1, ["aa", "bb"])
    n3 = ("Basic", ["Default"], 3, ["f", "g"])
    editcol(a, adds=[n3], edits=[n1], deletes=[2])
    pull()
    assert f.is_file()


def test_pull_unchanged_collection_is_no_op():
    """Does ki remove remote before quitting?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])
    clone(a)
    orig_hash = checksum_git_repository(".")
    pull()
    new_hash = checksum_git_repository(".")
    assert orig_hash == new_hash


def test_pull_avoids_unnecessary_merge_conflicts():
    """Does ki prevent gratuitous merge conflicts?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])
    clone(a)
    assert not os.path.isfile("Default/f.md")
    n1 = ("Basic", ["Default"], 1, ["aa", "bb"])
    n3 = ("Basic", ["Default"], 3, ["f", "g"])
    editcol(a, adds=[n3], edits=[n1], deletes=[2])
    out = pull()
    assert "Automatic merge failed; fix" not in out


def test_pull_still_works_from_subdirectories():
    """Does pull still work if you're farther down in the directory tree than the repo route?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])
    clone(a)
    assert not os.path.isfile("Default/f.md")
    n1 = ("Basic", ["Default"], 1, ["aa", "bb"])
    n3 = ("Basic", ["Default"], 3, ["f", "g"])
    editcol(a, adds=[n3], edits=[n1], deletes=[2])
    os.chdir("Default")
    pull()


def test_pull_displays_errors_from_rev():
    """Does 'pull()' return early when the last push tag is missing?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])
    repo, _ = clone(a)
    repo.delete_tag(LCA)
    n1 = ("Basic", ["Default"], 1, ["aa", "bb"])
    n3 = ("Basic", ["Default"], 3, ["f", "g"])
    editcol(a, adds=[n3], edits=[n1], deletes=[2])
    with pytest.raises(ValueError) as err:
        pull()
    assert LCA in str(err)


def test_pull_handles_unexpectedly_changed_checksums(mocker: MockerFixture):
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])
    clone(a)
    n1 = ("Basic", ["Default"], 1, ["aa", "bb"])
    n3 = ("Basic", ["Default"], 3, ["f", "g"])
    editcol(a, adds=[n3], edits=[n1], deletes=[2])
    mocker.patch("ki.F.md5", side_effect=["good", "good", "good", "bad"])
    with pytest.raises(CollectionChecksumError):
        pull()


def test_pull_displays_errors_from_repo_initialization(mocker: MockerFixture):
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])
    clone(a)
    n1 = ("Basic", ["Default"], 1, ["aa", "bb"])
    n3 = ("Basic", ["Default"], 3, ["f", "g"])
    editcol(a, adds=[n3], edits=[n1], deletes=[2])
    git.Repo.init(Path("."))
    effects = [git.InvalidGitRepositoryError()]
    mocker.patch("ki.M.repo", side_effect=effects)
    with pytest.raises(git.InvalidGitRepositoryError):
        pull()


def test_pull_removes_files_deleted_in_remote():
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])
    b = mkcol([("Basic", ["Default"], 2, ["c", "d"])])
    clone(a)
    assert (Path("Default") / "a.md").is_file()
    shutil.copyfile(b, a)
    pull()
    assert not Path("Default/a.md").is_file()


def test_pull_does_not_duplicate_decks_converted_to_subdecks_of_new_top_level_decks():
    BEFORE: SampleCollection = get_test_collection("duplicated_subdeck_before")
    AFTER: SampleCollection = get_test_collection("duplicated_subdeck_after")
    clone(BEFORE.col_file)
    shutil.copyfile(AFTER.path, BEFORE.col_file)
    pull()
    if os.path.isdir("onlydeck"):
        for _, _, filenames in os.walk("onlydeck"):
            assert len(filenames) == 0


def test_dsl_pull_leaves_no_working_tree_changes():
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])
    b = mkcol([("Basic", ["Default"], 2, ["c", "d"])])
    repo, _ = clone(a)
    shutil.copyfile(b, a)
    pull()
    assert not repo.is_dirty()


def test_pull_leaves_no_working_tree_changes():
    """Does everything get committed at the end of a `pull()`?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])
    b: File = mkcol([n2])
    repo, _ = clone(a)
    shutil.copyfile(b, a)
    pull()
    assert not repo.is_dirty()


def test_pull_doesnt_update_collection_hash_unless_merge_succeeds():
    """If we leave changes in the work tree, can we pull again after failure?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])
    clone(a)
    guid = "ed85e553fd0a6de8a58512acd265e76e13eb4303"
    write("Default/a.md", mkbasic(guid, ("r", "s")))
    n1 = ("Basic", ["Default"], 1, ["aa", "bb"])
    n3 = ("Basic", ["Default"], 3, ["f", "g"])
    editcol(a, adds=[n3], edits=[n1], deletes=[2])
    pull()
    out = pull()
    assert "Your local changes to the following files" in out
    assert "Default/a.md" in out


# PUSH


def test_push_writes_changes_correctly():
    """If there are committed changes, does push change the collection file?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])
    assert len(get_notes(a)) == 2
    repo, _ = clone(a)

    # Edit a note.
    append("Default/a.md", "e\n")

    # Delete a note.
    os.remove("Default/c.md")

    # Add a note.
    write_basic("Default", ("r", "s"))

    # Commit and push.
    F.commitall(repo, ".")
    out = push()
    assert "ADD                    1" in out
    assert "DELETE                 1" in out
    assert "MODIFY                 1" in out
    notes = get_notes(a)
    assert len(notes) == 2

    # Check c.md was deleted.
    nids = [note.n.id for note in notes]
    assert 1 in nids
    assert 2 not in nids


def test_push_verifies_md5sum():
    """Does ki only push if md5sum matches last pull?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])
    clone(a)
    randomly_swap_1_bit(a)
    with pytest.raises(UpdatesRejectedError):
        push()


def test_push_generates_correct_backup():
    """Does push store a backup identical to old collection file?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])
    old_hash = F.md5(a)
    repo, _ = clone(a)
    append("Default/a.md", "e\n")
    repo.git.add(all=True)
    repo.index.commit("Added 'e'.")
    push()
    assert os.path.isdir(".ki/backups")
    os.chdir(".ki/backups")
    paths = os.listdir()
    backup_exists = False
    for path in paths:
        if F.md5(F.chk(Path(path))) == old_hash:
            backup_exists = True
    assert backup_exists


def test_push_doesnt_write_uncommitted_changes():
    """Does push only write changes that have been committed?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])
    clone(a)
    append("Default/a.md", "e\n")

    # DON'T COMMIT, push.
    out = push()
    assert "ki push: up to date." in out
    assert len(os.listdir(".ki/backups")) == 0


def test_push_doesnt_fail_after_pull():
    """Does push work if we pull and then edit and then push?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])
    repo, _ = clone(a)
    assert not os.path.isfile("Default/f.md")
    n1 = ("Basic", ["Default"], 1, ["aa", "bb"])
    n3 = ("Basic", ["Default"], 3, ["f", "g"])
    editcol(a, adds=[n3], edits=[n1], deletes=[2])
    pull()
    assert os.path.isfile("Default/f.md")

    # Modify local file.
    assert os.path.isfile("Default/aa.md")
    append("Default/aa.md", "e\n")

    # Add two new files.
    write_basic("Default", ("r", "s"))
    write_basic("Default", ("s", "t"))
    F.commitall(repo, ".")
    repo.close()
    del repo
    gc.collect()

    # Push changes.
    out = push()
    assert "ADD                    2" in out
    assert "DELETE                 0" in out
    assert "MODIFY                 1" in out


def test_no_op_push_is_idempotent():
    """Does push not misbehave if you keep pushing?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])
    clone(a)
    push()
    push()
    push()
    push()
    push()
    push()


def test_push_deletes_notes():
    """Does push remove deleted notes from collection?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])
    repo, _ = clone(a)
    assert os.path.isfile("Default/a.md")
    os.remove("Default/a.md")

    # Commit the deletion.
    repo.git.add(all=True)
    repo.index.commit("Added 'e'.")
    push()

    # Check that note is gone.
    clone(a)
    assert not os.path.isfile("Default/a.md")


def test_push_still_works_from_subdirectories():
    """Does push still work if you're farther down in the directory tree than the repo route?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])

    repo, _ = clone(a)

    # Remove a note file.
    assert os.path.isfile("Default/a.md")
    os.remove("Default/a.md")

    # Commit the deletion.
    repo.git.add(all=True)
    repo.index.commit("Added 'e'.")

    # Push changes.
    os.chdir("Default")
    push()


def test_push_deletes_added_notes():
    """Does push remove deleted notes added with ki?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])
    repo, _ = clone(a)

    # Add new files.
    origs = os.listdir("Default")
    write_basic("Default", ("r", "s"))
    write_basic("Default", ("s", "t"))
    F.commitall(repo, ".")
    out = push()
    assert "ADD                    2" in out

    # Make sure 2 new files actually got added.
    os.chdir("Default")
    paths = os.listdir()
    notes = [path for path in paths if path[-3:] == ".md"]
    assert len(notes) == 4

    # Delete new files in `Default/`.
    do(os.remove, filter(lambda f: f not in origs, paths))
    F.commitall(repo, ".")
    out = push()
    assert "DELETE                 2" in out

    # Check that notes are gone.
    clone(a)
    contents = os.listdir("Default")
    notes = [path for path in contents if path[-3:] == ".md"]
    assert notes == ["c.md", "a.md"]


def test_push_honors_ignore_patterns():
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])

    # Clone collection in cwd.
    repo, _ = clone(a)

    # Add and commit a new file that is not a note.
    Path("dummy_file").touch()

    repo.git.add(all=True)
    repo.index.commit(".")

    # Since the output is currently very verbose, we should print a warning
    # for every such file.
    out = push()
    assert "up to date" in out


def test_push_displays_errors_from_head_ref_maybes(mocker: MockerFixture):
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])

    # Clone, edit, and commit.
    repo, _ = clone(a)
    write_basic("Default", ("r", "s"))
    F.commitall(repo, ".")

    mocker.patch(
        "ki.M.head_ki",
        side_effect=GitHeadRefNotFoundError(repo, Exception("<exc>")),
    )
    with pytest.raises(GitHeadRefNotFoundError):
        push()


def test_push_displays_errors_from_head(mocker: MockerFixture):
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])

    # Clone, edit, and commit.
    repo, _ = clone(a)
    write_basic("Default", ("r", "s"))
    F.commitall(repo, ".")

    mocker.patch(
        "ki.M.head_ki",
        side_effect=[
            GitHeadRefNotFoundError(repo, Exception("<exc>")),
        ],
    )
    with pytest.raises(GitHeadRefNotFoundError):
        push()


def test_push_displays_errors_from_notetype_parsing_in_write_collection_during_model_adding(
    mocker: MockerFixture,
):
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])

    # Clone, edit, and commit.
    repo, _ = clone(a)

    write_basic("Default", ("r", "s"))
    F.commitall(repo, ".")

    col = opencol(a)
    note = col.get_note(set(col.find_notes("")).pop())
    _: Notetype = ki.M.notetype(note.note_type())
    col.close()

    effects = [MissingFieldOrdinalError(3, "<notetype>")]

    mocker.patch("ki.M.notetype", side_effect=effects)

    with pytest.raises(MissingFieldOrdinalError):
        push()


def test_push_displays_errors_from_notetype_parsing_during_push_flatnote_to_anki(
    mocker: MockerFixture,
):
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])

    # Clone, edit, and commit.
    repo, _ = clone(a)
    write_basic("Default", ("r", "s"))
    F.commitall(repo, ".")

    col = opencol(a)
    note = col.get_note(set(col.find_notes("")).pop())
    notetype: Notetype = ki.M.notetype(note.note_type())
    col.close()

    effects = [notetype] * PARSE_NOTETYPE_DICT_CALLS_PRIOR_TO_FLATNOTE_PUSH
    effects += [MissingFieldOrdinalError(3, "<notetype>")]

    mocker.patch("ki.M.notetype", side_effect=effects)

    with pytest.raises(MissingFieldOrdinalError):
        push()


def test_push_writes_media():
    a = mkcol([])
    repo, _ = clone(a)

    # Add a new note file containing media, and the corresponding media file.
    write_basic("Default", ("air", '<img src="bullhorn-lg.png">'))
    col = opencol(a)
    col.media.add_file(DATA / "media/bullhorn-lg.png")
    col.close(save=True)
    F.commitall(repo, ".")
    repo.close()
    out = push()
    assert "ADD                    1" in out

    # Annihilate the repo root and re-clone.
    os.chdir("../")
    assert os.path.isdir("a")
    shutil.rmtree("a")
    assert not os.path.isdir("a")
    clone(a)

    # Check that added note and media file exist.
    col = opencol(a)
    check = col.media.check()
    assert os.path.isfile("Default/air.md")
    assert col.media.have("bullhorn-lg.png")
    assert len(check.missing) == 0
    assert len(check.unused) == 0


def test_push_handles_foreign_models():
    """Just check that we don't return an exception from `push()`."""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])
    repo, _ = clone(a)
    shutil.copytree(DATA / "repos/japanese-core-2000", "Default/japan")
    F.commitall(repo, ".")
    out = push()
    assert "ADD                    1" in out


def test_push_fails_if_database_is_locked():
    """Does ki print a nice error message when Anki is accidentally left open?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    a: File = mkcol([n1])
    repo, _ = clone(a)
    shutil.copytree(DATA / "repos/japanese-core-2000", "Default/japan")
    F.commitall(repo, ".")
    con = sqlite3.connect(a)
    con.isolation_level = "EXCLUSIVE"
    con.execute("BEGIN EXCLUSIVE")
    with pytest.raises(SQLiteLockError):
        push()


def test_push_is_nontrivial_when_pulled_changes_are_reverted():
    """
    If you push, make changes in Anki, then pull those changes, then undo them
    within the ki repo, then push again, the push should *not* be a no-op. The
    changes are currently applied in Anki, and the push should undo them.
    """
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])
    b: File = mkcol([n1, n2])
    repo, _ = clone(a)

    # Remove a note file.
    assert os.path.isfile("Default/a.md")
    os.remove("Default/a.md")
    F.commitall(repo, ".")

    # Push changes.
    out = push()
    notes = get_notes(a)
    notes = [colnote.n["Front"] for colnote in notes]
    assert notes == ["c"]

    # Revert the collection.
    os.remove(a)
    shutil.copyfile(b, a)

    # Pull again.
    out = pull()

    # Remove again.
    assert os.path.isfile("Default/a.md")
    os.remove("Default/a.md")
    F.commitall(repo, ".")

    # Push changes.
    out = push()
    notes = get_notes(a)
    fronts = [colnote.n["Front"] for colnote in notes]
    assert fronts == ["c"]
    assert "DELETE                 1" in out


def test_push_doesnt_unnecessarily_deduplicate_notetypes():
    """
    Does push refrain from adding a new notetype if the requested notetype
    already exists in the collection?
    """
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])
    b: File = mkcol([n1, n2])
    repo, _ = clone(a)

    col = opencol(a)
    models = col.models.all_names_and_ids()
    col.close(save=False)

    # Remove a note.
    assert os.path.isfile("Default/a.md")
    os.remove("Default/a.md")
    F.commitall(repo, ".")
    out = push()
    assert "DELETE                 1" in out
    shutil.copyfile(b, a)

    # Pull again.
    pull()

    # Remove again.
    assert os.path.isfile("Default/a.md")
    os.remove("Default/a.md")
    F.commitall(repo, ".")
    out = push()
    assert "DELETE                 1" in out

    # Push changes.
    push()

    col = opencol(a)
    assert len(models) == len(col.models.all_names_and_ids())
    col.close(save=False)


def test_push_is_nontrivial_when_pushed_changes_are_reverted_in_repository():
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    a: File = mkcol([n1])
    repo, _ = clone(a)

    # Remove a note file, push.
    tmp = F.mkdtemp() / "tmp.md"
    shutil.move("Default/a.md", tmp)
    F.commitall(repo, ".")
    out = push()
    assert "DELETE                 1" in out

    # Put file back, commit.
    shutil.move(tmp, "Default/a.md")
    F.commitall(repo, ".")

    # Push should be nontrivial.
    out = push()
    assert "ADD                    1" in out


def test_push_changes_deck_for_moved_notes():
    n1 = ("Basic", ["aa::bb::cc"], 1, ["cc", "cc"])
    n2 = ("Basic", ["aa::dd"], 2, ["dd", "dd"])
    a: File = mkcol([n1, n2])
    repo, _ = clone(a)

    # Move a note.
    assert os.path.isfile("aa/bb/cc/cc.md")
    shutil.move("aa/bb/cc/cc.md", "aa/dd/cc.md")
    assert not os.path.isfile("aa/bb/cc/cc.md")

    # Commit the move and push.
    repo.git.add(all=True)
    repo.index.commit("Move.")
    push()

    # Check that deck has changed.
    notes: List[ColNote] = get_notes(a)
    notes = list(filter(lambda colnote: colnote.n.id == 1, notes))
    assert len(notes) == 1
    assert notes[0].deck == "aa::dd"


def test_push_handles_tags_containing_trailing_commas():
    COMMAS: SampleCollection = get_test_collection("commas")
    repo, _ = clone(COMMAS.col_file)
    s = read("Default/c.md")
    s = s.replace("tag2", "tag3")
    write("Default/c.md", s)
    repo.git.add(all=True)
    repo.index.commit("e")
    repo.close()
    push()


def test_push_correctly_encodes_quotes_in_html_tags():
    """This is a weird test, not sure it can be refactored."""
    BROKEN: SampleCollection = get_test_collection("broken_media_links")

    # Clone collection in cwd.
    repo, _ = clone(BROKEN.col_file)
    note_file = (
        Path("🧙‍Recommendersysteme")
        / "wie-sieht-die-linkstruktur-von-einem-hub-in-einem-web-graphe.md"
    )
    s = read(note_file)
    s = s.replace("guter", "guuter")
    write(note_file, s)

    repo.git.add(all=True)
    repo.index.commit("e")
    repo.close()
    push()

    notes = get_notes(BROKEN.col_file)
    colnote = notes.pop()
    back: str = colnote.n["Back"]
    col = Collection(BROKEN.col_file)
    escaped: str = col.media.escape_media_filenames(back)
    col.close()
    assert '<img src="paste-64c7a314b90f3e9ef1b2d94edb396e07a121afdf.jpg">' in escaped


def test_push_rejects_updates_on_reset_to_prior_commit():
    """Does ki correctly verify md5sum?"""
    repo, _ = clone(mkcol([("Basic", ["Default"], 1, ["a", "b"])]))
    shutil.rmtree("Default")
    F.commitall(repo, ".")
    out = push()
    assert "DELETE                 1" in out

    # This actually *should* fail, because when we reset to the previous
    # commit, we annihilate the record of the latest collection hash. Thus
    # ki sees a collection which has changed since the last common ancestor
    # revision, and thus updates are rejected.
    repo.git.reset(["--hard", "HEAD~1"])
    with pytest.raises(UpdatesRejectedError):
        push()


def test_push_leaves_working_tree_clean():
    """Does the push command commit the hashes file?"""
    repo, _ = clone(mkcol([("Basic", ["Default"], 1, ["a", "b"])]))
    shutil.rmtree("Default")
    F.commitall(repo, ".")
    out = push()
    assert "DELETE                 1" in out
    assert not repo.is_dirty()
