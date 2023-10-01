#!/usr/bin/env python3
"""Tests for ki command line interface (CLI)."""
import os
import gc
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
from anki.collection import Collection

from beartype.typing import List

import ki
import ki.functional as F
from ki import MEDIA, LCA, _clone1, do
from ki.types import (
    Notetype,
    ColNote,
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
    DATA,
    read,
    write,
    append,
    opencol,
    mkcol,
    edit,
    editcol,
    mkbasic,
    write_basic,
    runcmd,
)


# pylint: disable=unnecessary-pass, too-many-lines, invalid-name, duplicate-code
# pylint: disable=missing-function-docstring, too-many-locals, no-value-for-parameter
# pylint: disable=unused-argument

PARSE_NOTETYPE_DICT_CALLS_PRIOR_TO_FLATNOTE_PUSH = 2


# CLI


@pytest.mark.skip
def test_bad_command_is_bad():
    """Typos should result in errors."""
    result = invoke(ki.ki, ["clome"])
    assert result.exit_code == 2
    assert "Error: No such command 'clome'." in result.output


@pytest.mark.skip
def test_runas_module():
    """Can this package be run as a Python module?"""
    command = "python -m ki --help"
    completed = subprocess.run(command, shell=True, capture_output=True, check=True)
    assert completed.returncode == 0


@pytest.mark.skip
def test_entrypoint():
    """Is entrypoint script installed? (setup.py)"""
    result = invoke(ki.ki, ["--help"])
    assert result.exit_code == 0


@pytest.mark.skip
def test_version():
    """Does --version display information as expected?"""
    expected_version = version("ki")
    result = invoke(ki.ki, ["--version"])

    assert result.stdout.rstrip() == f"ki, version {expected_version}"
    assert result.exit_code == 0


@pytest.mark.skip
def test_command_availability():
    """Are commands available?"""
    results = []
    results.append(invoke(ki.ki, ["clone", "--help"]))
    results.append(invoke(ki.ki, ["pull", "--help"]))
    results.append(invoke(ki.ki, ["push", "--help"]))
    for result in results:
        assert result.exit_code == 0


@pytest.mark.skip
def test_cli():
    """Does CLI stop execution w/o a command argument?"""
    with pytest.raises(SystemExit):
        ki.ki()
        pytest.fail("CLI doesn't abort asking for a command argument")


# COMMON


@pytest.mark.skip
def test_fails_without_ki_subdirectory():
    """Do pull and push know whether they're in a ki-generated git repo?"""
    tempdir = tempfile.mkdtemp()
    copy_tree(GITREPO_PATH, tempdir)
    os.chdir(tempdir)
    with pytest.raises(NotKiRepoError):
        pull()
    with pytest.raises(NotKiRepoError):
        push()


@pytest.mark.skip
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


@pytest.mark.skip
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


@pytest.mark.skip
def test_output():
    """Does it print nice things?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])
    repo, _ = clone(a)
    edit(a, ("Basic", ["Default"], 1, ["aa", "bb"]))
    edit(a, ("Basic", ["Default"], 2, ["f", "g"]))
    pull()

    p = Path("Default/aa.Card 1.md")
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


@pytest.mark.skip
def test_clone_fails_if_collection_doesnt_exist():
    """Does ki clone only if `.anki2` file exists?"""
    a: File = mkcol([])
    os.remove(a)
    with pytest.raises(FileNotFoundError):
        clone(a)
    assert not a.is_dir()


@pytest.mark.skip
def test_clone_fails_if_collection_is_already_open():
    """Does ki print a nice error message when Anki is accidentally left open?"""
    a: File = mkcol([])
    _ = opencol(a)
    with pytest.raises(AnkiAlreadyOpenError):
        clone(a)


@pytest.mark.skip
def test_clone_creates_directory():
    """Does it create the directory?"""
    clone(mkcol([]))
    os.chdir("../")
    assert os.path.isdir("a")


@pytest.mark.skip
def test_clone_errors_when_directory_is_populated():
    """Does it disallow overwrites?"""
    os.chdir(F.mkdtemp())
    os.mkdir("a")
    write("a/hi", "hi\n")
    with pytest.raises(TargetExistsError):
        _clone1(str(mkcol([])))


@pytest.mark.skip
def test_clone_cleans_up_on_error():
    """Does it clean up on nontrivial errors?"""
    a: File = mkcol([])
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


@pytest.mark.skip
def test_clone_clean_up_preserves_directories_that_exist_a_priori():
    """
    When clone fails and the cleanup function is called, does it not delete
    targetdirs that already existed?
    """
    os.chdir(F.mkdtemp())
    os.mkdir("a")
    assert os.path.isdir("a")
    cwd = os.getcwd()
    old_path = os.environ["PATH"]
    try:
        with pytest.raises(git.GitCommandNotFound):
            os.environ["PATH"] = ""
            _clone1(str(mkcol([])))
        assert os.getcwd() == cwd
        assert os.path.isdir("a")
        assert len(os.listdir("a")) == 0
    finally:
        os.environ["PATH"] = old_path


@pytest.mark.skip
def test_clone_succeeds_when_directory_exists_but_is_empty():
    """Does it clone into empty directories?"""
    os.chdir(F.mkdtemp())
    os.mkdir("a")
    clone(mkcol([]))


@pytest.mark.skip
def test_clone_generates_expected_notes():
    """Do generated note files match content of an example collection?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    a: File = mkcol([n1])
    os.chdir(F.mkdtemp())
    clone(a)
    assert os.path.isdir("Default")
    assert (
        Path("Default/a.Card 1.md").read_text()
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


@pytest.mark.skip
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
        (Path("aa") / "bb" / "cc" / "cc.Card 1.md").read_text()
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


@pytest.mark.skip
def test_clone_generates_ki_subdirectory():
    """Does clone command generate .ki/ directory?"""
    os.chdir(F.mkdtemp())
    clone(mkcol([]))
    assert Path(".ki/").is_dir()


@pytest.mark.skip
def test_cloned_collection_is_git_repository():
    """Does clone run `git init` and stuff?"""
    os.chdir(F.mkdtemp())
    clone(mkcol([]))
    os.chdir("../")
    assert is_git_repo("a")


@pytest.mark.skip
def test_clone_commits_directory_contents():
    """Does clone leave user with an up-to-date repo?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    os.chdir(F.mkdtemp())
    repo, _ = clone(mkcol([n1]))
    changes = repo.head.commit.diff()
    commits = list(repo.iter_commits("HEAD"))
    assert len(changes) == 0 and len(commits) == 1


@pytest.mark.skip
def test_clone_leaves_collection_file_unchanged():
    """Does clone leave the collection alone?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    a: File = mkcol([n1])
    os.chdir(F.mkdtemp())
    original_md5 = F.md5(a)
    clone(a)
    updated_md5 = F.md5(a)
    assert original_md5 == updated_md5


@pytest.mark.skip
def test_clone_directory_argument_works():
    """Does clone obey the target directory argument?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    a: File = mkcol([n1])
    tempdir = tempfile.mkdtemp()
    target = os.path.join(tempdir, "TARGET")
    assert not os.path.isdir(target)
    assert not os.path.isfile(target)
    clone(a, target)
    assert os.path.isdir(target)


@pytest.mark.skip
def test_clone_writes_media_files():
    """Does clone copy media files from the media directory into 'MEDIA'?"""
    a: File = mkcol([("Basic", ["Default"], 1, ["a", "b[sound:1sec.mp3]"])])
    col = opencol(a)
    col.media.add_file(DATA / "media/1sec.mp3")
    col.close(save=True)
    clone(a)
    assert (Path(MEDIA) / "1sec.mp3").is_file()


@pytest.mark.skip
def test_clone_handles_cards_from_a_single_note_in_distinct_decks():
    n1 = ("Basic (and reversed card)", ["top::a", "top::b"], 1, ["a", "b"])
    a: File = mkcol([n1])
    clone(a)
    two = Path("top/b/a.Card 2.md")
    orig = Path("top/a/a.Card 1.md")
    assert os.path.islink(two)
    assert os.path.isfile(orig)


@pytest.mark.skip
def test_clone_url_decodes_media_src_attributes():
    back = '<img src="Screenshot%202019-05-01%20at%2014.40.56.png">'
    a: File = mkcol([("Basic", ["Default"], 1, ["a", back])])
    clone(a)
    contents = read("Default/a.Card 1.md")
    assert '<img src="Screenshot 2019-05-01 at 14.40.56.png">' in contents


@pytest.mark.skip
def test_clone_leaves_no_working_tree_changes():
    """Does everything get committed at the end of a `clone()`?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    repo, _ = clone(mkcol([n1]))
    assert not repo.is_dirty()


# PULL


@pytest.mark.skip
def test_pull_fails_if_collection_no_longer_exists():
    """Does ki pull only if `.anki2` file exists?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    a: File = mkcol([n1])
    clone(a)
    os.remove(a)
    with pytest.raises(FileNotFoundError):
        pull()


@pytest.mark.skip
def test_pull_fails_if_collection_file_is_corrupted():
    """Does `pull()` fail gracefully when the collection file is bad?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    a: File = mkcol([n1])
    clone(a)
    a.write_text("bad_contents")
    with pytest.raises(SQLiteLockError):
        pull()


@pytest.mark.skip
def test_pull_writes_changes_correctly():
    """Does ki get the changes from modified collection file?"""
    a: File = mkcol([])
    clone(a)
    f = Path("Default/f.Card 1.md")
    assert not f.exists()
    n3 = ("Basic", ["Default"], 3, ["f", "g"])
    editcol(a, adds=[n3])
    pull()
    assert f.is_file()


@pytest.mark.skip
def test_pull_unchanged_collection_is_no_op():
    """Does ki remove remote before quitting?"""
    a: File = mkcol([])
    clone(a)
    orig_hash = checksum_git_repository(".")
    pull()
    new_hash = checksum_git_repository(".")
    assert orig_hash == new_hash


@pytest.mark.skip
def test_pull_avoids_unnecessary_merge_conflicts():
    """Does ki prevent gratuitous merge conflicts?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])
    clone(a)
    assert not os.path.isfile("Default/f.Card 1.md")
    n1 = ("Basic", ["Default"], 1, ["aa", "bb"])
    n3 = ("Basic", ["Default"], 3, ["f", "g"])
    editcol(a, adds=[n3], edits=[n1], deletes=[2])
    out = pull()
    assert "Automatic merge failed; fix" not in out
    assert "Fast-forward" in out


@pytest.mark.skip
def test_pull_still_works_from_subdirectories():
    """Does pull still work if you're farther down in the directory tree than the repo root?"""
    a: File = mkcol([])
    clone(a)
    assert not os.path.isfile("Default/f.Card 1.md")
    n3 = ("Basic", ["Default"], 3, ["f", "g"])
    editcol(a, adds=[n3])
    os.chdir("Default")
    out = pull()
    assert "Fast-forward" in out


@pytest.mark.skip
def test_pull_displays_errors_from_rev():
    """Does 'pull()' return early when the last push tag is missing?"""
    a: File = mkcol([])
    repo, _ = clone(a)
    repo.delete_tag(LCA)
    n3 = ("Basic", ["Default"], 3, ["f", "g"])
    editcol(a, adds=[n3])
    with pytest.raises(ValueError) as err:
        pull()
    assert LCA in str(err)


@pytest.mark.skip
def test_pull_handles_unexpectedly_changed_checksums(mocker: MockerFixture):
    a: File = mkcol([])
    clone(a)
    n3 = ("Basic", ["Default"], 3, ["f", "g"])
    editcol(a, adds=[n3])
    mocker.patch("ki.F.md5", side_effect=["good", "good", "good", "bad"])
    with pytest.raises(CollectionChecksumError):
        pull()


@pytest.mark.skip
def test_pull_displays_errors_from_repo_initialization(mocker: MockerFixture):
    a: File = mkcol([])
    clone(a)
    n3 = ("Basic", ["Default"], 3, ["f", "g"])
    editcol(a, adds=[n3])
    git.Repo.init(Path("."))
    effects = [git.InvalidGitRepositoryError()]
    mocker.patch("ki.M.repo", side_effect=effects)
    with pytest.raises(git.InvalidGitRepositoryError):
        pull()


@pytest.mark.skip
def test_pull_removes_files_deleted_in_remote():
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])
    clone(a)
    assert Path("Default/a.Card 1.md").is_file()
    editcol(a, deletes=[1])
    pull()
    assert not Path("Default/a.Card 1.md").is_file()


@pytest.mark.skip
def test_pull_does_not_duplicate_decks_converted_to_subdecks_of_new_top_level_decks():
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    a: File = mkcol([n1])
    clone(a)
    n1 = ("Basic", ["outer::onlydeck"], 1, ["a", "b"])
    editcol(a, edits=[n1])
    out = pull()
    assert "Fast-forward" in out
    assert os.path.isdir("outer")
    assert os.path.isdir("Default")
    assert not os.path.isdir("onlydeck")


@pytest.mark.skip
def test_dsl_pull_leaves_no_working_tree_changes():
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])
    repo, _ = clone(a)
    editcol(a, deletes=[1])
    pull()
    assert not repo.is_dirty()


@pytest.mark.skip
def test_pull_doesnt_update_collection_hash_unless_merge_succeeds():
    """If we leave changes in the work tree, can we pull again after failure?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    a: File = mkcol([n1])
    clone(a)
    guid = "ed85e553fd0a6de8a58512acd265e76e13eb4303"
    write("Default/a.Card 1.md", mkbasic(guid, ("r", "s")))
    n1 = ("Basic", ["Default"], 1, ["aa", "bb"])
    editcol(a, edits=[n1])
    out = pull()
    assert "Your local changes to the following files" in out
    assert ".ki/notes/712f285b6f243852414f.md" in out


# PUSH


def test_push_writes_changes_correctly():
    """If there are committed changes, does push change the collection file?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])
    col = opencol(a)
    assert len(get_notes(col)) == 2
    col.close(save=False)
    repo, _ = clone(a)

    # Edit a note.
    append("Default/a.Card 1.md", "e\n")

    # Delete a note.
    os.remove(os.path.realpath("Default/c.Card 1.md"))
    os.remove("Default/c.Card 1.md")

    # Add a note.
    write_basic("Default", ("r", "s"))

    # Commit and push.
    logger.debug(runcmd("git diff"))
    F.commitall(repo, ".")
    out = push()
    logger.debug(out)
    assert "ADD                    1" in out
    assert "DELETE                 1" in out
    assert "MODIFY                 1" in out
    col = opencol(a)
    notes = get_notes(col)
    assert len(notes) == 2

    # Check c.Card 1.md was deleted.
    nids = [note.n.id for note in notes]
    assert 1 in nids
    assert 2 not in nids
    col.close(save=False)


@pytest.mark.skip
def test_push_verifies_md5sum():
    """Does ki only push if md5sum matches last pull?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])
    clone(a)
    randomly_swap_1_bit(a)
    with pytest.raises(UpdatesRejectedError):
        push()


@pytest.mark.skip
def test_push_generates_correct_backup():
    """Does push store a backup identical to old collection file?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])
    old_hash = F.md5(a)
    repo, _ = clone(a)
    append("Default/a.Card 1.md", "e\n")
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


@pytest.mark.skip
def test_push_doesnt_write_uncommitted_changes():
    """Does push only write changes that have been committed?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])
    clone(a)
    append("Default/a.Card 1.md", "e\n")

    # DON'T COMMIT, push.
    out = push()
    assert "ki push: up to date." in out
    assert len(os.listdir(".ki/backups")) == 0


@pytest.mark.skip
def test_push_doesnt_fail_after_pull():
    """Does push work if we pull and then edit and then push?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])
    repo, _ = clone(a)
    assert not os.path.isfile("Default/f.Card 1.md")
    n1 = ("Basic", ["Default"], 1, ["aa", "bb"])
    n3 = ("Basic", ["Default"], 3, ["f", "g"])
    editcol(a, adds=[n3], edits=[n1], deletes=[2])
    pull()
    assert os.path.isfile("Default/f.Card 1.md")

    # Modify local file.
    assert os.path.isfile("Default/aa.Card 1.md")
    append("Default/aa.Card 1.md", "e\n")

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


@pytest.mark.skip
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


@pytest.mark.skip
def test_push_deletes_notes():
    """Does push remove deleted notes from collection?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    n2 = ("Basic", ["Default"], 2, ["c", "d"])
    a: File = mkcol([n1, n2])
    repo, _ = clone(a)
    assert os.path.isfile("Default/a.Card 1.md")
    os.remove("Default/a.Card 1.md")

    # Commit the deletion.
    repo.git.add(all=True)
    repo.index.commit("Added 'e'.")
    push()

    # Check that note is gone.
    clone(a)
    assert not os.path.isfile("Default/a.Card 1.md")


@pytest.mark.skip
def test_push_still_works_from_subdirectories():
    """Does push still work if you're farther down in the directory tree than the repo route?"""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    a: File = mkcol([n1])
    repo, _ = clone(a)

    # Remove a note file.
    assert os.path.isfile("Default/a.Card 1.md")
    os.remove("Default/a.Card 1.md")
    F.commitall(repo, ".")
    os.chdir("Default")
    out = push()
    assert "DELETE                 1" in out


@pytest.mark.skip
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
    assert notes == ["a.Card 1.md", "c.Card 1.md"]


@pytest.mark.skip
def test_push_honors_ignore_patterns():
    repo, _ = clone(mkcol([]))

    # Add and commit a new file that is not a note.
    Path("dummy_file").touch()
    F.commitall(repo, ".")
    out = push()

    # Since the output is currently very verbose, we should print a warning for
    # every such file.
    assert "up to date" in out


@pytest.mark.skip
def test_push_displays_errors_from_head_ref_maybes(mocker: MockerFixture):
    repo, _ = clone(mkcol([]))
    write_basic("Default", ("r", "s"))
    F.commitall(repo, ".")

    mocker.patch(
        "ki.M.head_ki",
        side_effect=GitHeadRefNotFoundError(repo, Exception("<exc>")),
    )
    with pytest.raises(GitHeadRefNotFoundError):
        push()


@pytest.mark.skip
def test_push_displays_errors_from_head(mocker: MockerFixture):
    repo, _ = clone(mkcol([]))
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


@pytest.mark.skip
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


@pytest.mark.skip
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


@pytest.mark.skip
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
    assert os.path.isfile("Default/air.Card 1.md")
    assert col.media.have("bullhorn-lg.png")
    assert len(check.missing) == 0
    assert len(check.unused) == 0


@pytest.mark.skip
def test_push_handles_foreign_models():
    """Just check that we don't return an exception from `push()`."""
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    a: File = mkcol([n1])
    repo, _ = clone(a)
    shutil.copytree(DATA / "repos/japanese-core-2000", "Default/japan")
    F.commitall(repo, ".")
    out = push()
    assert "ADD                    1" in out


@pytest.mark.skip
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


@pytest.mark.skip
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
    assert os.path.isfile("Default/a.Card 1.md")
    os.remove("Default/a.Card 1.md")
    F.commitall(repo, ".")

    # Push changes.
    out = push()
    col = opencol(a)
    notes = get_notes(col)
    fronts = [colnote.n["Front"] for colnote in notes]
    col.close(save=False)
    assert fronts == ["c"]

    # Revert the collection.
    os.remove(a)
    shutil.copyfile(b, a)

    # Pull again.
    out = pull()

    # Remove again.
    assert os.path.isfile("Default/a.Card 1.md")
    os.remove("Default/a.Card 1.md")
    F.commitall(repo, ".")

    # Push changes.
    out = push()
    col = opencol(a)
    notes = get_notes(col)
    fronts = [colnote.n["Front"] for colnote in notes]
    col.close(save=False)
    assert fronts == ["c"]
    assert "DELETE                 1" in out


@pytest.mark.skip
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
    assert os.path.isfile("Default/a.Card 1.md")
    os.remove("Default/a.Card 1.md")
    F.commitall(repo, ".")
    out = push()
    assert "DELETE                 1" in out
    shutil.copyfile(b, a)

    # Pull again.
    pull()

    # Remove again.
    assert os.path.isfile("Default/a.Card 1.md")
    os.remove("Default/a.Card 1.md")
    F.commitall(repo, ".")
    out = push()
    assert "DELETE                 1" in out

    # Push changes.
    push()

    col = opencol(a)
    assert len(models) == len(col.models.all_names_and_ids())
    col.close(save=False)


@pytest.mark.skip
def test_push_is_nontrivial_when_pushed_changes_are_reverted_in_repository():
    n1 = ("Basic", ["Default"], 1, ["a", "b"])
    a: File = mkcol([n1])
    repo, _ = clone(a)

    # Remove a note file, push.
    tmp = F.mkdtemp() / "tmp.md"
    src = os.path.realpath("Default/a.Card 1.md")
    shutil.move(src, tmp)
    logger.debug(runcmd("git status"))
    logger.debug(runcmd("git diff"))
    F.commitall(repo, ".")
    out = push()
    logger.debug(out)
    assert "DELETE                 1" in out

    # Put file back, commit.
    shutil.move(tmp, src)
    F.commitall(repo, ".")

    # Push should be nontrivial.
    out = push()
    logger.debug(out)
    assert "ADD                    1" in out


@pytest.mark.skip
def test_push_changes_deck_for_moved_notes():
    n1 = ("Basic", ["aa::bb::cc"], 1, ["cc", "cc"])
    n2 = ("Basic", ["aa::dd"], 2, ["dd", "dd"])
    a: File = mkcol([n1, n2])
    repo, _ = clone(a)

    # Move a note.
    assert os.path.isfile("aa/bb/cc/cc.Card 1.md")
    shutil.move("aa/bb/cc/cc.Card 1.md", "aa/dd/cc.Card 1.md")
    assert not os.path.isfile("aa/bb/cc/cc.Card 1.md")

    # Commit the move and push.
    repo.git.add(all=True)
    repo.index.commit("Move.")
    push()

    # Check that deck has changed.
    col = opencol(a)
    notes: List[ColNote] = get_notes(col)
    notes = list(filter(lambda colnote: colnote.n.id == 1, notes))
    assert len(notes) == 1
    assert notes[0].deck == "aa::dd"
    col.close(save=False)


@pytest.mark.skip
def test_push_handles_tags_containing_trailing_commas():
    COMMAS: SampleCollection = get_test_collection("commas")
    repo, _ = clone(COMMAS.col_file)
    s = read("Default/c.Card 1.md")
    s = s.replace("tag2", "tag3")
    write("Default/c.Card 1.md", s)
    repo.git.add(all=True)
    repo.index.commit("e")
    repo.close()
    push()


@pytest.mark.skip
def test_push_correctly_encodes_quotes_in_html_tags():
    """This is a weird test, not sure it can be refactored."""
    BROKEN: SampleCollection = get_test_collection("broken_media_links")

    # Clone collection in cwd.
    repo, _ = clone(BROKEN.col_file)
    note_file = (
        Path("🧙‍Recommendersysteme")
        / "wie-sieht-die-linkstruktur-von-einem-hub-in-einem-web-graphe.Card 1.md"
    )
    s = read(note_file)
    s = s.replace("guter", "guuter")
    write(note_file, s)

    repo.git.add(all=True)
    repo.index.commit("e")
    repo.close()
    push()

    col = opencol(BROKEN.col_file)
    notes = get_notes(col)
    colnote = notes.pop()
    back: str = colnote.n["Back"]
    escaped: str = col.media.escape_media_filenames(back)
    col.close(save=False)
    assert '<img src="paste-64c7a314b90f3e9ef1b2d94edb396e07a121afdf.jpg">' in escaped


@pytest.mark.skip
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


@pytest.mark.skip
def test_push_leaves_working_tree_clean():
    """Does the push command commit the hashes file?"""
    repo, _ = clone(mkcol([("Basic", ["Default"], 1, ["a", "b"])]))
    shutil.rmtree("Default")
    F.commitall(repo, ".")
    out = push()
    assert "DELETE                 1" in out
    assert not repo.is_dirty()


@pytest.mark.skip
def test_push_doesnt_collapse_cards_into_a_single_deck():
    n1 = ("Basic (and reversed card)", ["top::a", "top::b"], 1, ["aa", "bb"])
    a: File = mkcol([n1])
    repo, _ = clone(a)
    s = read("top/a/aa.Card 1.md")
    s = s.replace("aa", "cc")
    write("top/a/aa.Card 1.md", s)
    F.commitall(repo, ".")
    out = push()
    assert "up to date" not in out
    assert "MODIFY                 1" in out
    col = opencol(a)
    ns = get_notes(col)
    assert len(ns) == 1
    decks = list(map(lambda c: col.decks.get(c.did)["name"], ns[0].n.cards()))
    col.close(save=False)
    assert decks == ["top::a", "top::b"]
