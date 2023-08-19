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
import pprint as pp
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
from beartype.typing import List, Tuple, Dict, Union, Optional

import ki
import ki.maybes as M
import ki.functional as F
from ki import MEDIA, LCA, _clone1, do, get_guid, add_db_note
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
    open_collection,
    GITREPO_PATH,
    MULTI_GITREPO_PATH,
    MULTI_NOTE_PATH,
    MULTI_NOTE_ID,
    SUBMODULE_DIRNAME,
    NOTE_0,
    NOTE_1,
    NOTE_2,
    NOTE_3,
    NOTE_4,
    NOTE_7,
    NOTE_2_PATH,
    NOTE_3_PATH,
    NOTE_0_ID,
    NOTE_4_ID,
    MEDIA_NOTE,
    MEDIA_NOTE_PATH,
    MEDIA_FILE_PATH,
    MEDIA_FILENAME,
    TEST_DATA_PATH,
    invoke,
    clone,
    pull,
    push,
    is_git_repo,
    randomly_swap_1_bit,
    checksum_git_repository,
    get_notes,
    get_repo_with_submodules,
    get_repo_with_submodules_from_file,
    JAPANESE_GITREPO_PATH,
    BRANCH_NAME,
    get_test_collection,
    SampleCollection,
)


PARSE_NOTETYPE_DICT_CALLS_PRIOR_TO_FLATNOTE_PUSH = 2

# pylint: disable=unnecessary-pass, too-many-lines, invalid-name, duplicate-code
# pylint: disable=missing-function-docstring, too-many-locals, no-value-for-parameter
# pylint: disable=unused-argument


EDITED: SampleCollection = get_test_collection("edited")


NoteSpec = Tuple[str, int, List[str]]


@curried
@beartype
def mkbasic(col: Collection, spec: NoteSpec) -> None:
    fullname, nid, fields = spec
    did = col.decks.id(fullname)
    guid = get_guid(list(fields))
    mid = col.models.id_for_name("Basic")
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
    if cids:
        note.col.set_deck(cids, did)


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
def mkcol(ns: List[NoteSpec], empty_decks: Optional[List[str]] = None) -> File:
    file = F.touch(F.mkdtemp(), "a.anki2")
    col = opencol(file)
    do(mkbasic(col), ns)
    if empty_decks:
        do(col.decks.id, empty_decks)
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
def editbasic(col: Collection, spec: NoteSpec) -> None:
    fullname, nid, fields = spec
    front, back = fields
    note = col.get_note(nid)
    note["Front"] = front
    note["Back"] = back
    did: int = col.decks.id(fullname, create=True)
    cids = [c.id for c in note.cards()]
    if cids:
        col.set_deck(cids, did)
    note.flush()


@beartype
def edit(f: File, spec: NoteSpec) -> File:
    """Edit a note with specified nid."""
    col = opencol(f)
    editbasic(col, spec)
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
    do(mkbasic(col), adds)
    do(editbasic(col), edits)
    col.remove_notes(deletes)
    col.close(save=True)
    return f


@beartype
def mknote(deck: str, fields: Tuple[str, str]) -> None:
    """Write a markdown note to a deck from the root of a ki repository."""
    front, back = fields
    parts = deck.split("::")
    path = Path("./" + "/".join(parts + [f"{front}.md"]))
    s = f"""# Note
```
guid: {get_guid(list(fields))}
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
    path.write_text(s, encoding="UTF-8")


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
    # Clone collection in cwd.
    clone(ORIGINAL.col_file)

    # Check that hash is written.
    with open(".ki/hashes", encoding="UTF-8") as hashes_file:
        hashes = hashes_file.read()
        assert f"a68250f8ee3dc8302534f908bcbafc6a  {ORIGINAL.filename}" in hashes
        assert f"199216c39eeabe23a1da016a99ffd3e2  {ORIGINAL.filename}" not in hashes

    # Edit collection.
    shutil.copyfile(EDITED.path, ORIGINAL.col_file)

    # Pull edited collection.
    pull()

    # Check that edited hash is written and old hash is still there.
    with open(".ki/hashes", encoding="UTF-8") as hashes_file:
        hashes = hashes_file.read()
        assert f"a68250f8ee3dc8302534f908bcbafc6a  {ORIGINAL.filename}" in hashes
        assert f"199216c39eeabe23a1da016a99ffd3e2  {ORIGINAL.filename}" in hashes


def test_no_op_pull_push_cycle_is_idempotent():
    """Do pull/push not misbehave if you keep doing both?"""
    a = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])
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
    a = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])
    repo, _ = clone(a)
    edit(a, ("Default", 1, ["aa", "bb"]))
    edit(a, ("Default", 2, ["f", "g"]))
    pull()

    p = Path("Default/aa.md")
    assert p.is_file()
    with p.open("a", encoding="UTF-8") as f:
        f.write("e\n")
    mknote("Default", ("r", "s"))
    mknote("Default", ("s", "t"))

    # Commit.
    repo.git.add(all=True)
    repo.index.commit("Added 'e'.")

    # Push changes.
    out = push()
    assert "Overwrote" in out


# CLONE


def test_clone_fails_if_collection_doesnt_exist():
    """Does ki clone only if `.anki2` file exists?"""
    a = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])
    os.remove(a)
    with pytest.raises(FileNotFoundError):
        clone(a)
    assert not a.is_dir()


def test_clone_fails_if_collection_is_already_open():
    """Does ki print a nice error message when Anki is accidentally left open?"""
    a = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])
    os.remove(a)
    _ = open_collection(a)
    with pytest.raises(AnkiAlreadyOpenError):
        clone(a)


def test_clone_creates_directory():
    """Does it create the directory?"""
    a = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])
    clone(a)
    os.chdir("../")
    assert os.path.isdir("a")


def test_clone_errors_when_directory_is_populated():
    """Does it disallow overwrites?"""
    a: File = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])

    # Create directory where we want to clone.
    os.chdir(F.mkdtemp())
    os.mkdir("a")
    (Path("a") / "hi").write_text("hi\n", encoding="UTF-8")

    # Should error out because directory already exists.
    with pytest.raises(TargetExistsError):
        _clone1(str(a))


def test_clone_cleans_up_on_error():
    """Does it clean up on nontrivial errors?"""
    a: File = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])
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
    a: File = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])

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
    a: File = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])
    os.chdir(F.mkdtemp())
    os.mkdir("a")
    clone(a)


def test_clone_generates_expected_notes():
    """Do generated note files match content of an example collection?"""
    a: File = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])
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
            ("aa", 1, ["a", "aa"]),
            ("aa::bb", 2, ["bb", "bb"]),
            ("aa::bb::cc", 3, ["cc", "cc"]),
            ("aa::dd", 4, ["dd", "dd"]),
            ("Default", 5, ["hello", "hello"]),
            ("Default", 6, ["hello my enemy", "goodbye"]),
        ],
        empty_decks=[":a:::b:", "blank::blank", "blank::Hello"],
    )
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
    a: File = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])
    os.chdir(F.mkdtemp())
    clone(a)
    assert Path(".ki/").is_dir()


def test_cloned_collection_is_git_repository():
    """Does clone run `git init` and stuff?"""
    a: File = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])
    os.chdir(F.mkdtemp())
    clone(a)
    os.chdir("../")
    assert is_git_repo("a")


def test_clone_commits_directory_contents():
    """Does clone leave user with an up-to-date repo?"""
    a: File = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])
    os.chdir(F.mkdtemp())
    repo, _ = clone(a)
    changes = repo.head.commit.diff()
    commits = list(repo.iter_commits("HEAD"))
    assert len(changes) == 0 and len(commits) == 1


def test_clone_leaves_collection_file_unchanged():
    """Does clone leave the collection alone?"""
    a: File = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])
    os.chdir(F.mkdtemp())
    original_md5 = F.md5(a)
    clone(a)
    updated_md5 = F.md5(a)
    assert original_md5 == updated_md5


def test_clone_directory_argument_works():
    """Does clone obey the target directory argument?"""
    a: File = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])
    tempdir = tempfile.mkdtemp()
    target = os.path.join(tempdir, "TARGET")
    assert not os.path.isdir(target)
    assert not os.path.isfile(target)
    clone(a, target)
    assert os.path.isdir(target)


def test_clone_writes_media_files():
    """Does clone copy media files from the media directory into 'MEDIA'?"""
    MEDIACOL: SampleCollection = get_test_collection("media")
    clone(MEDIACOL.col_file)
    assert (Path(MEDIA) / "1sec.mp3").is_file()


def test_clone_handles_cards_from_a_single_note_in_distinct_decks():
    SPLIT: SampleCollection = get_test_collection("split")

    clone(SPLIT.col_file)
    two = Path("top") / "b" / "a_Card 2.md"
    orig = Path("top") / "a" / "a.md"

    if sys.platform == "win32":
        assert two.read_text(encoding="UTF-8") == r"../../top/a/a.md"
    else:
        assert os.path.islink(two)
    assert os.path.isfile(orig)


def test_clone_writes_plaintext_posix_symlinks_on_windows():
    SYMLINKS: SampleCollection = get_test_collection("symlinks")

    repo, _ = clone(SYMLINKS.col_file)

    # Verify that there are no symlinks in the cloned sample repo.
    for root, _, files in os.walk("."):
        for file in files:
            path = os.path.join(root, file)
            if sys.platform == "win32":
                assert not os.path.islink(path)

    winlinks = [
        Path("Default") / "B" / "sample_cloze-ol.md",
        Path("Default") / "B" / "sample_cloze-ol_1.md",
        Path("Default") / "C" / "sample_cloze-ol.md",
        Path("Default") / "C" / "sample_cloze-ol_1.md",
        Path("Default") / "C" / "sample_cloze-ol_2.md",
        Path("Default") / "C" / "sample_cloze-ol_3.md",
        Path("Default") / "C" / "sample_cloze-ol_4.md",
    ]
    winlinks = {str(link) for link in winlinks}

    # Check that each windows symlink has the correct file mode.
    for entry in repo.commit().tree.traverse():
        path = entry.path
        if isinstance(entry, git.Blob) and path in winlinks:
            mode = oct(entry.mode)
            assert mode == "0o120000"


def test_clone_url_decodes_media_src_attributes():
    DOUBLE: SampleCollection = get_test_collection("no_double_encodings")
    clone(DOUBLE.col_file)
    path = Path("DeepLearning for CV") / "list-some-pros-and-cons-of-dl.md"
    with open(path, "r", encoding="UTF-8") as f:
        contents: str = f.read()
    assert '<img src="Screenshot 2019-05-01 at 14.40.56.png">' in contents


def test_clone_leaves_no_working_tree_changes():
    """Does everything get committed at the end of a `clone()`?"""
    a: File = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])
    repo, _ = clone(a)
    assert not repo.is_dirty()


# PULL


def test_pull_fails_if_collection_no_longer_exists():
    """Does ki pull only if `.anki2` file exists?"""
    a: File = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])
    clone(a)
    os.remove(a)
    with pytest.raises(FileNotFoundError):
        pull()


def test_pull_fails_if_collection_file_is_corrupted():
    """Does `pull()` fail gracefully when the collection file is bad?"""
    a: File = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])
    clone(a)
    a.write_text("bad_contents")
    with pytest.raises(SQLiteLockError):
        pull()


def test_pull_writes_changes_correctly():
    """Does ki get the changes from modified collection file?"""
    a: File = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])
    clone(a)
    f = Path("Default") / "f.md"
    assert not f.exists()
    editcol(
        a,
        adds=[("Default", 3, ["f", "g"])],
        edits=[("Default", 1, ["aa", "bb"])],
        deletes=[2],
    )
    out = pull()
    assert f.is_file()


def test_pull_unchanged_collection_is_no_op():
    """Does ki remove remote before quitting?"""
    a: File = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])
    clone(a)
    orig_hash = checksum_git_repository(".")
    pull()
    new_hash = checksum_git_repository(".")
    assert orig_hash == new_hash


def test_pull_avoids_unnecessary_merge_conflicts():
    """Does ki prevent gratuitous merge conflicts?"""
    a: File = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])
    clone(a)
    assert not os.path.isfile(NOTE_1)
    shutil.copyfile(EDITED.path, a)
    out = pull()
    assert "Automatic merge failed; fix" not in out


def test_pull_still_works_from_subdirectories():
    """Does pull still work if you're farther down in the directory tree than the repo route?"""
    a: File = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])
    clone(a)
    assert not os.path.isfile(NOTE_1)
    shutil.copyfile(EDITED.path, a)
    os.chdir("Default")
    pull()


def test_pull_displays_errors_from_rev():
    """Does 'pull()' return early when the last push tag is missing?"""
    a: File = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])
    repo, _ = clone(a)
    repo.delete_tag(LCA)
    shutil.copyfile(EDITED.path, a)
    with pytest.raises(ValueError) as err:
        pull()
    assert LCA in str(err)


def test_pull_handles_unexpectedly_changed_checksums(mocker: MockerFixture):
    a: File = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])
    clone(a)
    shutil.copyfile(EDITED.path, a)
    mocker.patch("ki.F.md5", side_effect=["good", "good", "good", "bad"])
    with pytest.raises(CollectionChecksumError):
        pull()


def test_pull_displays_errors_from_repo_initialization(mocker: MockerFixture):
    a: File = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])
    clone(a)
    shutil.copyfile(EDITED.path, a)
    git.Repo.init(Path("."))
    effects = [git.InvalidGitRepositoryError()]
    mocker.patch("ki.M.repo", side_effect=effects)
    with pytest.raises(git.InvalidGitRepositoryError):
        pull()


def test_pull_handles_non_standard_submodule_branch_names():
    a: File = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])
    repo: git.Repo = get_repo_with_submodules_from_file(a)

    # Copy a new note into the submodule.
    note_path = Path(SUBMODULE_DIRNAME) / "Default" / NOTE_2
    shutil.copyfile(NOTE_2_PATH, note_path)

    # Get a reference to the submodule repo.
    subrepo = git.Repo(SUBMODULE_DIRNAME)
    subrepo.git.branch(["-m", "main", "brain"])

    # Commit changes in submodule and parent repo.
    subrepo.git.add(all=True)
    subrepo.index.commit("Add a new note.")
    repo.git.add(all=True)
    repo.index.commit("Update submodule.")
    push()

    # Edit collection (implicitly removes submodule).
    shutil.copyfile(EDITED.path, a)
    pull()


def test_pull_handles_uncommitted_submodule_commits():
    UNCOMMITTED_SM: SampleCollection = get_test_collection(
        "uncommitted_submodule_commits"
    )
    UNCOMMITTED_SM_EDITED: SampleCollection = get_test_collection(
        "uncommitted_submodule_commits_edited"
    )
    japanese_gitrepo_path = Path(JAPANESE_GITREPO_PATH).resolve()

    JAPANESE_SUBMODULE_DIRNAME = "japanese-core-2000"

    # Clone collection.
    repo, _ = clone(UNCOMMITTED_SM.col_file)

    # Check that the content of a note in the collection is correct.
    with open(Path(JAPANESE_SUBMODULE_DIRNAME) / "それ.md", "r", encoding="UTF-8") as f:
        note_text = f.read()
        expected = "that, that one\nthat, that one\nthis, this one"
        assert expected in note_text

    # Delete `japanese-core-2000/` subdirectory, and commit.
    F.rmtree(F.chk(Path(JAPANESE_SUBMODULE_DIRNAME)))
    repo.git.add(all=True)
    repo.index.commit("Delete cloned `japanese-core-2000` folder.")
    repo.close()

    # Push the deletion.
    push()

    # Copy a new directory of notes to `japanese-core-2000/` subdirectory,
    # and initialize it as a git repository.
    submodule_name = JAPANESE_SUBMODULE_DIRNAME
    shutil.copytree(japanese_gitrepo_path, submodule_name)
    git.Repo.init(submodule_name, initial_branch=BRANCH_NAME)
    sm = git.Repo(submodule_name)
    sm.git.add(all=True)
    _ = sm.index.commit("Initial commit.")

    # Add as a submodule.
    repo.git.submodule("add", Path(submodule_name).resolve())
    repo.git.add(all=True)
    _ = repo.index.commit("Add submodule.")
    repo.close()

    # Push changes.
    push()

    # Add a new line to a note, and commit the addition in the submodule.
    with open(Path(JAPANESE_SUBMODULE_DIRNAME) / "それ.md", "a", encoding="UTF-8") as f:
        f.write("A new line at the bottom.")
    sm.git.add(all=True)
    _ = sm.index.commit("Added a new line.")
    sm.close()

    # Edit collection.
    shutil.copyfile(UNCOMMITTED_SM_EDITED.col_file, UNCOMMITTED_SM.col_file)

    # Pull changes from collection to root ki repository.
    out = pull()
    assert "fatal: remote error: " not in out
    assert "CONFLICT" not in out

    with open(Path(JAPANESE_SUBMODULE_DIRNAME) / "それ.md", "r", encoding="UTF-8") as f:
        note_text = f.read()
    expected_mackerel = "\nholy mackerel\n"
    expected_this = "\nthis, this one\n"
    assert expected_mackerel in note_text
    assert expected_this in note_text


def test_pull_removes_files_deleted_in_remote():
    a = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])
    b = mkcol([("Default", 2, ["c", "d"])])
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
    a = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])
    b = mkcol([("Default", 2, ["c", "d"])])
    repo, _ = clone(a)
    shutil.copyfile(b, a)
    pull()
    assert not repo.is_dirty()


def test_pull_leaves_no_working_tree_changes():
    """Does everything get committed at the end of a `pull()`?"""
    a: File = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])
    DELETED: SampleCollection = get_test_collection("deleted")
    repo, _ = clone(a)
    shutil.copyfile(DELETED.path, a)
    pull()
    assert not repo.is_dirty()


def test_pull_succeeds_with_new_submodules():
    """Does a nontrivial pull succeed when we add a new submodule?"""
    MULTIDECK: SampleCollection = get_test_collection("multideck")
    submodule_py_path = os.path.abspath("submodule.py")

    # Clone collection in cwd.
    repo, _ = clone(MULTIDECK.col_file)
    os.chdir("../")

    rem_path = F.mkdir(F.chk(Path("aa_remote")))
    rem = git.Repo.init(rem_path, initial_branch=BRANCH_NAME)
    os.chdir(rem_path)
    Path("some_file").write_text("hello", encoding="UTF-8")
    rem.git.add(".")
    rem.git.commit(["-m", "hello"])
    os.chdir("..")
    rem.git.checkout(["-b", "alt"])
    remote_path = str(Path(os.path.abspath(rem.working_dir)) / ".git")

    # Here we call submodule.py
    subprocess.run(
        [
            "python3",
            submodule_py_path,
            "--kirepo",
            MULTIDECK.repodir,
            "--deck",
            "aa",
            "--remote",
            remote_path,
        ],
        check=False,
        capture_output=True,
        encoding="UTF-8",
    )

    # Make change in Anki, adding a card to the submodule.
    col = M.collection(MULTIDECK.col_file)
    nt = col.models.current()
    note = col.new_note(nt)
    did = col.decks.id("aa::bb", create=False)
    col.add_note(note, did)
    col.close(save=True)

    os.chdir(repo.working_dir)
    pull()


def test_pull_doesnt_update_collection_hash_unless_merge_succeeds():
    """If we leave changes in the work tree, can we pull again after failure?"""
    a: File = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])
    clone(a)
    shutil.copyfile(NOTE_2_PATH, os.path.join("Default", "a.md"))
    shutil.copyfile(EDITED.path, a)
    pull()
    out = pull()
    assert out != "ki pull: up to date.\n"
    assert "Aborting" in out


# PUSH


def test_push_writes_changes_correctly():
    """If there are committed changes, does push change the collection file?"""
    a: File = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])
    old_notes = get_notes(a)
    repo, _ = clone(a)

    # Edit a note.
    with open(NOTE_0, "a", encoding="UTF-8") as note_file:
        note_file.write("e\n")

    # Delete a note.
    os.remove(NOTE_4)

    # Add a note.
    shutil.copyfile(NOTE_2_PATH, NOTE_2)

    # Commit.
    repo.git.add(all=True)
    repo.index.commit("Added 'e'.")

    # Push and check for changes.
    push()
    new_notes = get_notes(a)

    # Check NOTE_4 was deleted.
    new_ids = [note.n.id for note in new_notes]
    assert NOTE_4_ID not in new_ids

    # Check that note with nid 1 was edited.
    old_note_0 = ""
    for note in new_notes:
        if note.n.id == 1:
            old_note_0 = str(note)
    assert len(old_note_0) > 0
    found_0 = False
    for note in new_notes:
        if note.n.id == 1:
            assert old_note_0 == str(note)
            found_0 = True
    assert found_0

    # Check NOTE_2 was added.
    assert len(old_notes) == 2
    assert len(new_notes) == 2


def test_push_verifies_md5sum():
    """Does ki only push if md5sum matches last pull?"""
    a: File = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])
    clone(a)
    randomly_swap_1_bit(a)
    with pytest.raises(UpdatesRejectedError):
        push()


def test_push_generates_correct_backup():
    """Does push store a backup identical to old collection file?"""
    a: File = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])
    old_hash = F.md5(a)
    repo, _ = clone(a)
    with open(NOTE_0, "a", encoding="UTF-8") as note_file:
        note_file.write("e\n")
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
    a: File = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])
    clone(a)
    with open(NOTE_0, "a", encoding="UTF-8") as note_file:
        note_file.write("e\n")

    # DON'T COMMIT, push.
    out = push()
    assert "ki push: up to date." in out
    assert len(os.listdir(".ki/backups")) == 0


def test_push_doesnt_fail_after_pull():
    """Does push work if we pull and then edit and then push?"""
    a: File = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])
    repo, _ = clone(a)
    assert not os.path.isfile(NOTE_1)
    shutil.copyfile(EDITED.path, a)
    pull()
    assert os.path.isfile(NOTE_1)

    # Modify local file.
    assert os.path.isfile(NOTE_7)
    with open(NOTE_7, "a", encoding="UTF-8") as note_file:
        note_file.write("e\n")

    # Add new file.
    shutil.copyfile(NOTE_2_PATH, NOTE_2)
    # Add new file.
    shutil.copyfile(NOTE_3_PATH, NOTE_3)

    # Commit.
    repo.git.add(all=True)
    repo.index.commit("Added 'e'.")
    repo.close()
    del repo
    gc.collect()

    # Push changes.
    push()


def test_no_op_push_is_idempotent():
    """Does push not misbehave if you keep pushing?"""
    a: File = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])
    clone(a)
    push()
    push()
    push()
    push()
    push()
    push()


def test_push_deletes_notes():
    """Does push remove deleted notes from collection?"""
    a: File = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])
    repo, _ = clone(a)
    assert os.path.isfile(NOTE_0)
    os.remove(NOTE_0)

    # Commit the deletion.
    repo.git.add(all=True)
    repo.index.commit("Added 'e'.")
    push()

    # Check that note is gone.
    clone(a)
    assert not os.path.isfile(NOTE_0)


def test_push_still_works_from_subdirectories():
    """Does push still work if you're farther down in the directory tree than the repo route?"""
    a: File = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])

    repo, _ = clone(a)

    # Remove a note file.
    assert os.path.isfile(NOTE_0)
    os.remove(NOTE_0)

    # Commit the deletion.
    repo.git.add(all=True)
    repo.index.commit("Added 'e'.")

    # Push changes.
    os.chdir("Default")
    push()


def test_push_deletes_added_notes():
    """Does push remove deleted notes added with ki?"""
    a: File = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])
    repo, _ = clone(a)

    # Add new files.
    contents = os.listdir("Default")
    shutil.copyfile(NOTE_2_PATH, os.path.join("Default", NOTE_2))
    shutil.copyfile(NOTE_3_PATH, os.path.join("Default", NOTE_3))

    # Commit/push the additions.
    repo.git.add(all=True)
    repo.index.commit("Added 'e'.")
    push()

    # Make sure 2 new files actually got added.
    os.chdir("Default")
    post_push_contents = os.listdir()
    notes = [path for path in post_push_contents if path[-3:] == ".md"]
    assert len(notes) == 4

    # Delete added files.
    for file in post_push_contents:
        if file not in contents:
            os.remove(file)

    # Commit the deletions.
    repo.git.add(all=True)
    repo.index.commit("Added 'e'.")

    # Push changes.
    push()

    # Check that notes are gone.
    clone(a)
    contents = os.listdir("Default")
    notes = [path for path in contents if path[-3:] == ".md"]
    assert len(notes) == 2


def test_push_honors_ignore_patterns():
    a: File = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])

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
    a: File = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])

    # Clone, edit, and commit.
    repo, _ = clone(a)
    shutil.copyfile(NOTE_2_PATH, os.path.join("Default", NOTE_2))
    repo.git.add(all=True)
    repo.index.commit(".")

    mocker.patch(
        "ki.M.head_ki",
        side_effect=GitHeadRefNotFoundError(repo, Exception("<exc>")),
    )
    with pytest.raises(GitHeadRefNotFoundError):
        push()


def test_push_displays_errors_from_head(mocker: MockerFixture):
    a: File = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])

    # Clone, edit, and commit.
    repo, _ = clone(a)
    shutil.copyfile(NOTE_2_PATH, os.path.join("Default", NOTE_2))
    repo.git.add(all=True)
    repo.index.commit(".")

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
    a: File = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])

    # Clone, edit, and commit.
    repo, _ = clone(a)

    shutil.copyfile(NOTE_2_PATH, os.path.join("Default", NOTE_2))
    repo.git.add(all=True)
    repo.index.commit(".")

    col = open_collection(a)
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
    a: File = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])

    # Clone, edit, and commit.
    repo, _ = clone(a)
    shutil.copyfile(NOTE_2_PATH, os.path.join("Default", NOTE_2))
    repo.git.add(all=True)
    repo.index.commit(".")

    col = open_collection(a)
    note = col.get_note(set(col.find_notes("")).pop())
    notetype: Notetype = ki.M.notetype(note.note_type())
    col.close()

    effects = [notetype] * PARSE_NOTETYPE_DICT_CALLS_PRIOR_TO_FLATNOTE_PUSH
    effects += [MissingFieldOrdinalError(3, "<notetype>")]

    mocker.patch("ki.M.notetype", side_effect=effects)

    with pytest.raises(MissingFieldOrdinalError):
        push()


def test_push_handles_submodules():
    ORIGINAL = get_test_collection("original")

    repo = get_repo_with_submodules(ORIGINAL)
    os.chdir(repo.working_dir)

    # Edit a file within the submodule.
    file = Path(repo.working_dir) / SUBMODULE_DIRNAME / "Default" / "a.md"
    with open(file, "a", encoding="UTF-8") as note_f:
        note_f.write("\nz\n\n")

    # Copy a new note into the submodule.
    shutil.copyfile(
        NOTE_2_PATH, Path(repo.working_dir) / SUBMODULE_DIRNAME / "Default" / NOTE_2
    )

    subrepo = git.Repo(Path(repo.working_dir) / SUBMODULE_DIRNAME)
    subrepo.git.add(all=True)
    subrepo.index.commit(".")
    repo.git.add(all=True)
    repo.index.commit(".")

    push()

    colnotes = get_notes(ORIGINAL.col_file)
    notes: List[Note] = [colnote.n for colnote in colnotes]
    assert len(notes) == 3
    assert "<br>z<br>" in notes[0]["Back"]


def test_push_writes_media():
    MEDIACOL: SampleCollection = get_test_collection("media")

    # Clone.
    repo, _ = clone(MEDIACOL.col_file)

    # Add a new note file containing media, and the corresponding media file.
    media_note_path = Path("Default") / MEDIA_NOTE
    media_file_path = Path("Default") / MEDIA / MEDIA_FILENAME
    shutil.copyfile(MEDIA_NOTE_PATH, media_note_path)
    shutil.copyfile(MEDIA_FILE_PATH, media_file_path)

    # Commit the additions.
    repo.git.add(all=True)
    repo.index.commit("Add air.md")
    repo.close()

    # Push the commit.
    push()

    # Annihilate the repo root.
    os.chdir("../")
    F.rmtree(F.chk(Path(MEDIACOL.repodir)))

    # Re-clone the pushed collection.
    clone(MEDIACOL.col_file)

    # Check that added note and media file exist.
    col = open_collection(MEDIACOL.col_file)
    check = col.media.check()
    assert os.path.isfile(Path("Default") / MEDIA_NOTE)
    assert col.media.have(MEDIA_FILENAME)
    assert len(check.missing) == 0
    assert len(check.unused) == 0


def test_push_handles_foreign_models():
    """Just check that we don't return an exception from `push()`."""
    a: File = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])
    japan_path = (Path(TEST_DATA_PATH) / "repos" / "japanese-core-2000").resolve()
    repo, _ = clone(a)
    shutil.copytree(japan_path, Path("Default") / "japan")
    repo.git.add(all=True)
    repo.index.commit("japan")
    push()


def test_push_fails_if_database_is_locked():
    """Does ki print a nice error message when Anki is accidentally left open?"""
    a: File = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])
    japan_path = (Path(TEST_DATA_PATH) / "repos" / "japanese-core-2000").resolve()
    repo, _ = clone(a)
    shutil.copytree(japan_path, Path("Default") / "japan")
    repo.git.add(all=True)
    repo.index.commit("japan")
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
    a: File = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])
    COPY: SampleCollection = get_test_collection("original")
    repo, _ = clone(a)

    # Remove a note file.
    assert os.path.isfile(NOTE_0)
    os.remove(NOTE_0)

    # Commit the deletion.
    repo.git.add(all=True)
    repo.index.commit("Deleted.")

    # Push changes.
    out = push()
    notes = get_notes(a)
    notes = [colnote.n["Front"] for colnote in notes]
    assert notes == ["c"]

    # Revert the collection.
    os.remove(a)
    shutil.copyfile(COPY.col_file, a)

    # Pull again.
    out = pull()

    # Remove again.
    assert os.path.isfile(NOTE_0)
    os.remove(NOTE_0)
    repo.git.add(all=True)
    repo.index.commit("Deleted.")

    # Push changes.
    out = push()
    notes = get_notes(a)
    notes = [colnote.n["Front"] for colnote in notes]
    assert "a" not in notes
    assert notes == ["c"]
    assert "ki push: up to date." not in out


def test_push_doesnt_unnecessarily_deduplicate_notetypes():
    """
    Does push refrain from adding a new notetype if the requested notetype
    already exists in the collection?
    """
    ORIGINAL: SampleCollection = get_test_collection("original")
    COPY: SampleCollection = get_test_collection("original")

    # Clone collection in cwd.
    repo, _ = clone(ORIGINAL.col_file)

    col = open_collection(ORIGINAL.col_file)
    orig_models = col.models.all_names_and_ids()
    col.close(save=False)

    # Remove a note file.
    assert os.path.isfile(NOTE_0)
    os.remove(NOTE_0)

    # Commit the deletion.
    repo.git.add(all=True)
    repo.index.commit("Deleted.")

    # Push changes.
    push()

    # Revert the collection.
    os.remove(ORIGINAL.col_file)
    shutil.copyfile(COPY.col_file, ORIGINAL.col_file)

    # Pull again.
    pull()

    # Remove again.
    assert os.path.isfile(NOTE_0)
    os.remove(NOTE_0)
    repo.git.add(all=True)
    repo.index.commit("Deleted.")

    # Push changes.
    push()

    col = open_collection(ORIGINAL.col_file)
    models = col.models.all_names_and_ids()
    assert len(orig_models) == len(models)
    col.close(save=False)


def test_push_is_nontrivial_when_pushed_changes_are_reverted_in_repository():
    """
    The following operation should be nontrivial:
    - Clone
    - Delete a note
    - Push
    - Add note back
    - Push again

    The last push, in particular, should add the note back in.
    """
    a: File = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])

    # Clone collection in cwd.
    repo, _ = clone(a)

    # Remove a note file.
    assert os.path.isfile(NOTE_0)
    temp_note_0_file = F.mkdtemp() / "NOTE_0"
    shutil.move(NOTE_0, temp_note_0_file)
    assert not os.path.isfile(NOTE_0)

    # Commit the deletion.
    repo.git.add(all=True)
    repo.index.commit("Deleted.")

    # Push changes.
    out = push()

    # Put file back.
    shutil.move(temp_note_0_file, NOTE_0)
    repo.git.add(all=True)
    repo.index.commit("Added.")

    # Push again.
    out = push()
    assert "ki push: up to date." not in out


def test_push_changes_deck_for_moved_notes():
    MULTIDECK: SampleCollection = get_test_collection("multideck")

    # Clone collection in cwd.
    repo, _ = clone(MULTIDECK.col_file)

    # Move a note.
    target = "aa/dd/cc.md"
    assert os.path.isfile(MULTI_NOTE_PATH)
    shutil.move(MULTI_NOTE_PATH, target)
    assert not os.path.isfile(MULTI_NOTE_PATH)

    # Commit the move.
    repo.git.add(all=True)
    repo.index.commit("Move.")

    # Push changes.
    push()

    # Check that deck has changed.
    notes: List[ColNote] = get_notes(MULTIDECK.col_file)
    notes = filter(lambda colnote: colnote.n.id == MULTI_NOTE_ID, notes)
    notes = list(notes)
    assert len(notes) == 1
    colnote = notes.pop()
    assert colnote.deck == "aa::dd"


def test_push_is_trivial_for_committed_submodule_contents():
    UNCOMMITTED_SM: SampleCollection = get_test_collection(
        "uncommitted_submodule_commits"
    )
    japanese_gitrepo_path = Path(JAPANESE_GITREPO_PATH).resolve()

    JAPANESE_SUBMODULE_DIRNAME = "japanese-core-2000"

    # Clone collection in cwd.
    repo, out = clone(UNCOMMITTED_SM.col_file)

    # Delete a directory.
    F.rmtree(F.chk(Path(JAPANESE_SUBMODULE_DIRNAME)))
    repo.git.add(all=True)
    repo.index.commit("Delete cloned `japanese-core-2000` folder.")

    # Push deletion.
    out = push()

    # Add a submodule.
    submodule_name = JAPANESE_SUBMODULE_DIRNAME
    shutil.copytree(japanese_gitrepo_path, submodule_name)
    git.Repo.init(submodule_name, initial_branch=BRANCH_NAME)
    sm = git.Repo(submodule_name)
    sm.git.add(all=True)
    _ = sm.index.commit("Initial commit.")
    repo.git.submodule("add", Path(submodule_name).resolve())
    repo.git.add(all=True)
    _ = repo.index.commit("Add submodule.")

    out = push()
    out = push()
    assert "ki push: up to date." in out


def test_push_prints_informative_warning_on_push_when_subrepo_was_added_instead_of_submodule():
    a: File = mkcol([("Default", 1, ["a", "b"]), ("Default", 2, ["c", "d"])])
    japanese_gitrepo_path = Path(JAPANESE_GITREPO_PATH).resolve()
    JAPANESE_SUBMODULE_DIRNAME = "japanese-core-2000"

    # Clone collection in cwd.
    repo, _ = clone(a)

    # Add a *subrepo* (not submodule).
    submodule_name = JAPANESE_SUBMODULE_DIRNAME
    shutil.copytree(japanese_gitrepo_path, submodule_name)

    p = subprocess.run(
        ["git", "add", "--all"], check=True, capture_output=True, encoding="UTF-8"
    )
    if "warning" in p.stderr:
        repo.index.commit("Add subrepo.")
        repo.close()
        out = push()
        assert "'git submodule add'" in out


def test_push_handles_tags_containing_trailing_commas():
    COMMAS: SampleCollection = get_test_collection("commas")
    repo, _ = clone(COMMAS.col_file)

    c_file = Path("Default") / "c.md"
    with open(c_file, "r", encoding="UTF-8") as read_f:
        contents = read_f.read().replace("tag2", "tag3")
        with open(c_file, "w", encoding="UTF-8") as write_f:
            write_f.write(contents)

    repo.git.add(all=True)
    repo.index.commit("e")
    repo.close()
    push()


def test_push_correctly_encodes_quotes_in_html_tags():
    BROKEN: SampleCollection = get_test_collection("broken_media_links")

    # Clone collection in cwd.
    repo, _ = clone(BROKEN.col_file)
    note_file = (
        Path("🧙‍Recommendersysteme")
        / "wie-sieht-die-linkstruktur-von-einem-hub-in-einem-web-graphe.md"
    )
    with open(note_file, "r", encoding="UTF-8") as read_f:
        contents = read_f.read().replace("guter", "guuter")
        with open(note_file, "w", encoding="UTF-8") as write_f:
            write_f.write(contents)

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
    KOREAN: SampleCollection = get_test_collection("tiny_korean")

    # Clone collection in cwd.
    repo, _ = clone(KOREAN.col_file)
    shutil.rmtree(Path("TTMIK Supplement") / "TTMIK Level 3")
    F.commitall(repo, "msg")
    push()

    # This actually *should* fail, because when we reset to the previous
    # commit, we annihilate the record of the latest collection hash. Thus
    # ki sees a collection which has changed since the last common ancestor
    # revision, and thus updates are rejected.
    repo.git.reset(["--hard", "HEAD~1"])
    with pytest.raises(UpdatesRejectedError):
        push()


def test_push_leaves_working_tree_clean():
    """Does the push command commit the hashes file?"""
    KOREAN: SampleCollection = get_test_collection("tiny_korean")

    # Clone collection in cwd.
    repo, _ = clone(KOREAN.col_file)
    shutil.rmtree(Path("TTMIK Supplement") / "TTMIK Level 3")
    F.commitall(repo, "msg")
    push()
    assert not repo.is_dirty()
