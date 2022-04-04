#!/usr/bin/env python3
"""Tests for ki command line interface (CLI)."""
import os
import random
import shutil
import tempfile
import subprocess
from pathlib import Path
from distutils.dir_util import copy_tree
from importlib.metadata import version

import git
import pytest
import bitstring
import checksumdir
from lark.exceptions import UnexpectedToken
from loguru import logger
from apy.anki import Anki
from click.testing import CliRunner

from beartype import beartype
from beartype.typing import List

import ki
from ki.note import KiNote
from ki.transformer import FlatNote


# pylint:disable=unnecessary-pass, too-many-lines


TEST_DATA_PATH = "tests/data/"
COLLECTIONS_PATH = os.path.join(TEST_DATA_PATH, "collections/")
COLLECTION_FILENAME = "collection.anki2"
ORIG_COLLECTION_FILENAME = "original.anki2"
EDITED_COLLECTION_FILENAME = "edited.anki2"
MULTIDECK_COLLECTION_FILENAME = "multideck.anki2"
HTML_COLLECTION_FILENAME = "html.anki2"
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
GITREPO_PATH = os.path.join(TEST_DATA_PATH, "repos/", "original/")
MULTI_GITREPO_PATH = os.path.join(TEST_DATA_PATH, "repos/", "multideck/")
REPODIR = os.path.splitext(COLLECTION_FILENAME)[0]
MULTIDECK_REPODIR = os.path.splitext(MULTIDECK_COLLECTION_FILENAME)[0]
HTML_REPODIR = os.path.splitext(HTML_COLLECTION_FILENAME)[0]
MULTI_NOTE_PATH = "aa/bb/cc/cc.md"

NOTES_PATH = os.path.abspath(os.path.join(TEST_DATA_PATH, "notes/"))

NOTE_0 = "Default/a.md"
NOTE_1 = "Default/f.md"
NOTE_2 = "note123412341234.md"
NOTE_3 = "note 3.md"
NOTE_4 = "Default/c.md"
NOTE_5 = "alpha_nid.md"
NOTE_6 = "no_nid.md"

NOTE_0_PATH = os.path.join(NOTES_PATH, NOTE_0)
NOTE_1_PATH = os.path.join(NOTES_PATH, NOTE_1)
NOTE_2_PATH = os.path.join(NOTES_PATH, NOTE_2)
NOTE_3_PATH = os.path.join(NOTES_PATH, NOTE_3)
NOTE_4_PATH = os.path.join(NOTES_PATH, NOTE_4)
NOTE_5_PATH = os.path.join(NOTES_PATH, NOTE_5)
NOTE_6_PATH = os.path.join(NOTES_PATH, NOTE_6)

NOTE_0_ID = 1645010162168
NOTE_4_ID = 1645027705329

# HELPER FUNCTIONS


def invoke(*args, **kwargs):
    """Wrap click CliRunner invoke()."""
    return CliRunner().invoke(*args, **kwargs)


@beartype
def clone(runner: CliRunner, repository: str, directory: str = "") -> str:
    """Make a test `ki clone` call."""
    res = runner.invoke(
        ki.ki,
        ["clone", repository, directory],
        standalone_mode=False,
        catch_exceptions=False,
    )
    return res.output


@beartype
def pull(runner: CliRunner) -> str:
    """Make a test `ki pull` call."""
    res = runner.invoke(ki.ki, ["pull"], standalone_mode=False, catch_exceptions=False)
    return res.output


@beartype
def push(runner: CliRunner) -> str:
    """Make a test `ki push` call."""
    res = runner.invoke(ki.ki, ["push"], standalone_mode=False, catch_exceptions=False)
    return res.output


@beartype
def get_collection_path() -> str:
    """Put `collection.anki2` in a tempdir and return its abspath."""
    # Copy collection to tempdir.
    tempdir = tempfile.mkdtemp()
    collection_path = os.path.abspath(os.path.join(tempdir, COLLECTION_FILENAME))
    shutil.copyfile(COLLECTION_PATH, collection_path)
    assert os.path.isfile(collection_path)
    return collection_path


@beartype
def get_multideck_collection_path() -> str:
    """Put `multideck.anki2` in a tempdir and return its abspath."""
    # Copy collection to tempdir.
    tempdir = tempfile.mkdtemp()
    collection_path = os.path.abspath(
        os.path.join(tempdir, MULTIDECK_COLLECTION_FILENAME)
    )
    shutil.copyfile(MULTIDECK_COLLECTION_PATH, collection_path)
    assert os.path.isfile(collection_path)
    return collection_path


@beartype
def get_html_collection_path() -> str:
    """Put `html.anki2` in a tempdir and return its abspath."""
    # Copy collection to tempdir.
    tempdir = tempfile.mkdtemp()
    collection_path = os.path.abspath(
        os.path.join(tempdir, HTML_COLLECTION_FILENAME)
    )
    shutil.copyfile(HTML_COLLECTION_PATH, collection_path)
    assert os.path.isfile(collection_path)
    return collection_path


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
def randomly_swap_1_bit(path: str) -> None:
    """Randomly swap a bit in a file."""
    assert os.path.isfile(path)

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
def get_notes(collection: str) -> List[KiNote]:
    """Get a list of notes from a path."""
    # Import with apy.
    query = ""
    with Anki(path=collection) as a:
        notes: List[KiNote] = []
        for i in set(a.col.find_notes(query)):
            notes.append(KiNote(a, a.col.get_note(i)))
    return notes


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

    assert result.stdout == f"ki, version {expected_version}{os.linesep}"
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
    gitrepo_path = os.path.abspath(GITREPO_PATH)
    runner = CliRunner()
    with runner.isolated_filesystem():
        tempdir = tempfile.mkdtemp()
        copy_tree(gitrepo_path, tempdir)
        os.chdir(tempdir)
        with pytest.raises(FileNotFoundError):
            pull(runner)
        with pytest.raises(FileNotFoundError):
            push(runner)


def test_computes_and_stores_md5sum():
    """Does ki add new hash to `.ki/hashes`?"""
    collection_path = get_collection_path()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        # ki.clone(collection_path)
        clone(runner, collection_path)

        # Check that hash is written.
        with open(os.path.join(REPODIR, ".ki/hashes"), encoding="UTF-8") as hashes_file:
            hashes = hashes_file.read()
            assert "a68250f8ee3dc8302534f908bcbafc6a  collection.anki2" in hashes
            assert "199216c39eeabe23a1da016a99ffd3e2  collection.anki2" not in hashes

        # Edit collection.
        shutil.copyfile(EDITED_COLLECTION_PATH, collection_path)

        # Pull edited collection.
        os.chdir(REPODIR)
        pull(runner)
        os.chdir("../")

        # Check that edited hash is written and old hash is still there.
        with open(os.path.join(REPODIR, ".ki/hashes"), encoding="UTF-8") as hashes_file:
            hashes = hashes_file.read()
            assert "a68250f8ee3dc8302534f908bcbafc6a  collection.anki2" in hashes
            assert "199216c39eeabe23a1da016a99ffd3e2  collection.anki2" in hashes


def test_no_op_pull_push_cycle_is_idempotent():
    """Do pull/push not misbehave if you keep doing both?"""
    collection_path = get_collection_path()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, collection_path)
        assert os.path.isdir(REPODIR)

        os.chdir(REPODIR)
        out = pull(runner)
        assert "Merge made by the" not in out
        push(runner)
        out = pull(runner)
        assert "Merge made by the" not in out
        push(runner)
        out = pull(runner)
        assert "Merge made by the" not in out
        push(runner)
        out = pull(runner)
        assert "Merge made by the" not in out
        push(runner)


def test_output():
    """Does it print nice things?"""
    collection_path = get_collection_path()
    runner = CliRunner()
    with runner.isolated_filesystem():
        out = clone(runner, collection_path)
        logger.debug(f"\nCLONE:\n{out}")

        # Edit collection.
        shutil.copyfile(EDITED_COLLECTION_PATH, collection_path)

        # Pull edited collection.
        os.chdir(REPODIR)
        out = pull(runner)
        logger.debug(f"\nPULL:\n{out}")

        # Modify local repository.
        assert os.path.isfile(NOTE_0)
        with open(NOTE_0, "a", encoding="UTF-8") as note_file:
            note_file.write("e\n")
        shutil.copyfile(NOTE_2_PATH, NOTE_2)
        shutil.copyfile(NOTE_3_PATH, NOTE_3)

        # Commit.
        os.chdir("../")
        repo = git.Repo(REPODIR)
        repo.git.add(all=True)
        repo.index.commit("Added 'e'.")

        # Push changes.
        os.chdir(REPODIR)
        out = push(runner)
        logger.debug(f"\nPUSH:\n{out}")
        assert "Overwrote" in out


# CLONE


def test_clone_fails_if_collection_doesnt_exist():
    """Does ki clone only if `.anki2` file exists?"""
    collection_path = get_collection_path()
    os.remove(collection_path)
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, collection_path)
        assert not os.path.isdir(REPODIR)


def test_clone_creates_directory():
    """Does it create the directory?"""
    collection_path = get_collection_path()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, collection_path)

        assert os.path.isdir(REPODIR)


def test_clone_handles_html():
    """Does it tidy html and stuff?"""
    collection_path = get_html_collection_path()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, collection_path)
        assert os.path.isdir(HTML_REPODIR)

        path = Path(".") / "html" / "Default" / "あだ名.md"
        contents = path.read_text()
        assert "<!DOCTYPE html>" in contents


def test_clone_errors_when_directory_is_populated():
    """Does it disallow overwrites?"""
    collection_path = get_collection_path()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Create directory where we want to clone.
        os.mkdir(REPODIR)
        with open(os.path.join(REPODIR, "hi"), "w", encoding="UTF-8") as hi_file:
            hi_file.write("hi\n")

        # Should error out because directory already exists.
        out = clone(runner, collection_path)
        assert "is not an empty" in out


def test_clone_cleans_up_on_error():
    """Does it clean up on nontrivial errors?"""
    collection_path = get_html_collection_path()
    runner = CliRunner()
    with runner.isolated_filesystem():

        clone(runner, collection_path)
        assert os.path.isdir(HTML_REPODIR)
        shutil.rmtree(HTML_REPODIR)
        old_path = os.environ["PATH"]
        try:
            os.environ["PATH"] = ""
            out = clone(runner, collection_path)
            assert "No such file or directory: 'tidy'" in out
            assert "Failed: exiting." in out
            assert not os.path.isdir(HTML_REPODIR)
        finally:
            os.environ["PATH"] = old_path


def test_clone_succeeds_when_directory_exists_but_is_empty():
    """Does it clone into empty directories?"""
    collection_path = get_collection_path()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Create directory where we want to clone.
        os.mkdir(REPODIR)
        clone(runner, collection_path)


def test_clone_generates_expected_notes():
    """Do generated note files match content of an example collection?"""
    true_note_path = os.path.abspath(os.path.join(GITREPO_PATH, NOTE_0))
    cloned_note_path = os.path.join(REPODIR, NOTE_0)
    collection_path = get_collection_path()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, collection_path)

        # Check that deck directory is created.
        assert os.path.isdir(os.path.join(REPODIR, "Default"))

        # Compute hashes.
        cloned_md5 = ki.md5(cloned_note_path)
        true_md5 = ki.md5(true_note_path)

        assert cloned_md5 == true_md5


def test_clone_generates_deck_tree_correctly():
    """Does generated FS tree match example collection?"""
    true_note_path = os.path.abspath(os.path.join(MULTI_GITREPO_PATH, MULTI_NOTE_PATH))
    cloned_note_path = os.path.join(MULTIDECK_REPODIR, MULTI_NOTE_PATH)
    collection_path = get_multideck_collection_path()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        out = clone(runner, collection_path)
        logger.debug(f"\n{out}")

        # Check that deck directory is created and all subdirectories.
        assert os.path.isdir(os.path.join(MULTIDECK_REPODIR, "Default"))
        assert os.path.isdir(os.path.join(MULTIDECK_REPODIR, "aa/bb/cc"))
        assert os.path.isdir(os.path.join(MULTIDECK_REPODIR, "aa/dd"))

        # Compute hashes.
        cloned_md5 = ki.md5(cloned_note_path)
        true_md5 = ki.md5(true_note_path)

        assert cloned_md5 == true_md5


def test_clone_generates_ki_subdirectory():
    """Does clone command generate .ki/ directory?"""
    collection_path = get_collection_path()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, collection_path)

        # Check kidir exists.
        kidir = os.path.join(REPODIR, ".ki/")
        assert os.path.isdir(kidir)


def test_cloned_collection_is_git_repository():
    """Does clone run `git init` and stuff?"""
    collection_path = get_collection_path()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, collection_path)

        assert is_git_repo(REPODIR)


def test_clone_commits_directory_contents():
    """Does clone leave user with an up-to-date repo?"""
    collection_path = get_collection_path()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, collection_path)

        # Construct repo object.
        repo = git.Repo(REPODIR)

        # Make sure there are no changes.
        changes = repo.head.commit.diff()
        assert len(changes) == 0

        # Make sure there is exactly 1 commit.
        commits = list(repo.iter_commits("HEAD"))
        assert len(commits) == 1


def test_clone_leaves_collection_file_unchanged():
    """Does clone leave the collection alone?"""
    collection_path = get_collection_path()
    original_md5 = ki.md5(collection_path)
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, collection_path)

        updated_md5 = ki.md5(collection_path)
        assert original_md5 == updated_md5


def test_clone_directory_argument_works():
    """Does clone obey the target directory argument?"""
    collection_path = get_collection_path()
    runner = CliRunner()
    with runner.isolated_filesystem():

        tempdir = tempfile.mkdtemp()
        target = os.path.join(tempdir, "TARGET")
        assert not os.path.isdir(target)
        assert not os.path.isfile(target)

        # Clone collection in cwd.
        clone(runner, collection_path, target)
        assert os.path.isdir(target)


# PULL


def test_pull_fails_if_collection_no_longer_exists():
    """Does ki pull only if `.anki2` file exists?"""
    collection_path = get_collection_path()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, collection_path)

        # Delete collection and try to pull.
        os.remove(collection_path)
        with pytest.raises(FileNotFoundError):
            os.chdir(REPODIR)
            pull(runner)


def test_pull_writes_changes_correctly():
    """Does ki get the changes from modified collection file?"""
    collection_path = get_collection_path()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, collection_path)
        assert not os.path.isfile(os.path.join(REPODIR, NOTE_1))

        # Edit collection.
        shutil.copyfile(EDITED_COLLECTION_PATH, collection_path)

        # Pull edited collection.
        os.chdir(REPODIR)
        pull(runner)
        assert os.path.isfile(NOTE_1)


def test_pull_unchanged_collection_is_no_op():
    """Does ki remove remote before quitting?"""
    collection_path = get_collection_path()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, collection_path)
        orig_hash = checksum_git_repository(REPODIR)

        # Pull updated collection.
        os.chdir(REPODIR)
        pull(runner)
        os.chdir("../")
        new_hash = checksum_git_repository(REPODIR)

        assert orig_hash == new_hash


def test_pull_avoids_unnecessary_merge_conflicts():
    """Does ki prevent gratuitous merge conflicts?"""
    collection_path = get_collection_path()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, collection_path)
        assert not os.path.isfile(os.path.join(REPODIR, NOTE_1))

        # Edit collection.
        shutil.copyfile(EDITED_COLLECTION_PATH, collection_path)

        # Pull edited collection.
        os.chdir(REPODIR)
        out = pull(runner)
        assert "Automatic merge failed; fix" not in out


# PUSH


def test_push_writes_changes_correctly():
    """If there are committed changes, does push change the collection file?"""
    collection_path = get_collection_path()
    old_notes = get_notes(collection_path)
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, collection_path)

        # Edit a note.
        note = os.path.join(REPODIR, NOTE_0)
        with open(note, "a", encoding="UTF-8") as note_file:
            note_file.write("e\n")

        # Delete a note.
        note = os.path.join(REPODIR, NOTE_4)
        os.remove(note)

        # Add a note.
        shutil.copyfile(NOTE_2_PATH, os.path.join(REPODIR, NOTE_2))

        # Commit.
        repo = git.Repo(REPODIR)
        repo.git.add(all=True)
        repo.index.commit("Added 'e'.")

        # Push and check for changes.
        os.chdir(REPODIR)
        push(runner)
        new_notes = get_notes(collection_path)

        # Check NOTE_4 was deleted.
        new_ids = [note.n.id for note in new_notes]
        assert NOTE_4_ID not in new_ids

        # Check NOTE_0 was edited.
        old_note_0 = ""
        for note in new_notes:
            if note.n.id == NOTE_0_ID:
                old_note_0 = str(note)
        assert len(old_note_0) > 0
        found_0 = False
        for note in new_notes:
            if note.n.id == NOTE_0_ID:
                assert old_note_0 == str(note)
                found_0 = True
        assert found_0

        # Check NOTE_2 was added.
        assert len(old_notes) == len(new_notes) == 2


def test_push_verifies_md5sum():
    """Does ki only push if md5sum matches last pull?"""
    collection_path = get_collection_path()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, collection_path)

        # Swap a bit.
        randomly_swap_1_bit(collection_path)

        # Make sure ki complains.
        os.chdir(REPODIR)
        out = push(runner)
        assert "Failed to push some refs to" in out


@pytest.mark.xfail
def test_push_generates_correct_backup():
    """Does push store a backup identical to old collection file?"""
    collection_path = get_collection_path()
    old_hash = ki.md5(collection_path)
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, collection_path)

        # Make change in repo.
        note = os.path.join(REPODIR, NOTE_0)
        with open(note, "a", encoding="UTF-8") as note_file:
            note_file.write("e\n")

        # Commit.
        repo = git.Repo(REPODIR)
        repo.git.add(all=True)
        repo.index.commit("Added 'e'.")

        os.chdir(REPODIR)
        push(runner)
        assert os.path.isdir(".ki/backups")

        os.chdir(".ki/backups")
        paths = os.listdir()

        backup = False
        for path in paths:
            if ki.md5(path) == old_hash:
                backup = True

        assert backup


def test_push_doesnt_write_uncommitted_changes():
    """Does push only write changes that have been committed?"""
    collection_path = get_collection_path()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, collection_path)

        # Make change in repo.
        note = os.path.join(REPODIR, NOTE_0)
        with open(note, "a", encoding="UTF-8") as note_file:
            note_file.write("e\n")

        # DON'T COMMIT, push.
        os.chdir(REPODIR)
        out = push(runner)
        assert "ki push: up to date." in out
        assert not os.path.isdir(".ki/backups")


@pytest.mark.xfail
def test_push_doesnt_fail_after_pull():
    """Does push work if we pull and then edit and then push?"""
    collection_path = get_collection_path()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, collection_path)
        assert not os.path.isfile(os.path.join(REPODIR, NOTE_1))

        # Edit collection.
        shutil.copyfile(EDITED_COLLECTION_PATH, collection_path)

        # Pull edited collection.
        os.chdir(REPODIR)
        pull(runner)
        assert os.path.isfile(NOTE_1)

        # Modify local file.
        assert os.path.isfile(NOTE_0)
        with open(NOTE_0, "a", encoding="UTF-8") as note_file:
            note_file.write("e\n")

        # Add new file.
        shutil.copyfile(NOTE_2_PATH, NOTE_2)
        # Add new file.
        shutil.copyfile(NOTE_3_PATH, NOTE_3)

        # Commit.
        os.chdir("../")
        repo = git.Repo(REPODIR)
        repo.git.add(all=True)
        repo.index.commit("Added 'e'.")

        # Push changes.
        os.chdir(REPODIR)
        push(runner)


def test_no_op_push_is_idempotent():
    """Does push not misbehave if you keep pushing?"""
    collection_path = get_collection_path()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, collection_path)
        assert os.path.isdir(REPODIR)

        os.chdir(REPODIR)
        push(runner)
        push(runner)
        push(runner)
        push(runner)
        push(runner)
        push(runner)


@pytest.mark.xfail
def test_push_deletes_notes():
    """Does push remove deleted notes from collection?"""
    collection_path = get_collection_path()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, collection_path)

        # Remove a note file.
        os.chdir(REPODIR)
        assert os.path.isfile(NOTE_0)
        os.remove(NOTE_0)

        # Commit the deletion.
        os.chdir("../")
        repo = git.Repo(REPODIR)
        repo.git.add(all=True)
        repo.index.commit("Added 'e'.")

        # Push changes.
        os.chdir(REPODIR)
        out = push(runner)
        logger.debug(f"\nPUSH:\n{out}")

    # Check that note is gone.
    with runner.isolated_filesystem():
        clone(runner, collection_path)
        assert not os.path.isfile(NOTE_0)


@pytest.mark.xfail
def test_push_deletes_added_notes():
    """Does push remove deleted notes added with ki?"""
    collection_path = get_collection_path()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, collection_path)

        # Add new files.
        os.chdir(REPODIR)
        contents = os.listdir()
        shutil.copyfile(NOTE_2_PATH, NOTE_2)
        shutil.copyfile(NOTE_3_PATH, NOTE_3)

        # Commit the additions.
        os.chdir("../")
        repo = git.Repo(REPODIR)
        repo.git.add(all=True)
        repo.index.commit("Added 'e'.")

        # Push changes.
        os.chdir(REPODIR)
        out = push(runner)
        logger.debug(f"\nPUSH:\n{out}")

        # Make sure 2 new files actually got added.
        post_push_contents = os.listdir()
        notes = [path for path in post_push_contents if path[-3:] == ".md"]
        assert len(notes) == 4

        # Delete added files.
        for file in post_push_contents:
            if file not in contents:
                logger.debug(f"Removing '{file}'")
                os.remove(file)

        # Commit the deletions.
        os.chdir("../")
        repo = git.Repo(REPODIR)
        repo.git.add(all=True)
        repo.index.commit("Added 'e'.")
        os.chdir(REPODIR)

        # Push changes.
        out = push(runner)
        logger.debug(f"\nPUSH:\n{out}")

    # Check that notes are gone.
    with runner.isolated_filesystem():
        clone(runner, collection_path)
        contents = os.listdir(REPODIR)
        notes = [path for path in contents if path[-3:] == ".md"]
        logger.debug(f"Notes: {notes}")
        assert len(notes) == 2


# UTILS


def test_parse_markdown_notes():
    """Does ki raise an error when it fails to parse nid?"""
    with pytest.raises(UnexpectedToken):
        ki.parse_markdown_notes(NOTE_5_PATH)
    with pytest.raises(UnexpectedToken):
        ki.parse_markdown_notes(NOTE_6_PATH)


def test_clone_helper_checks_for_colpath_existence():
    """Does``_clone()`` check that the collection path exists?"""
    with pytest.raises(FileNotFoundError):

        # pylint: disable=protected-access
        ki._clone(Path("/tmp/NONEXISTENT_PATH.anki2"), Path("/tmp/TARGET"), "", False)


def test_get_batches():
    """Does it get batches from a list of strings?"""
    batches = list(ki.get_batches(["0", "1", "2", "3"], n=2))
    assert batches == [["0", "1"], ["2", "3"]]


def test_is_anki_note():
    """Do asserts in ``is_anki_note()`` actually do anything?"""
    assert ki.is_anki_note(Path("note.mda")) is False
    assert ki.is_anki_note(Path("note.amd")) is False
    assert ki.is_anki_note(Path("note.md.txt")) is False
    assert ki.is_anki_note(Path("note.nd")) is False

    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("note.md").write_text("", encoding="UTF-8")
        assert ki.is_anki_note(Path("note.md")) is False

        Path("note.md").write_text("one line", encoding="UTF-8")
        assert ki.is_anki_note(Path("note.md")) is False

        Path("note.md").write_text("### Note\n## Note\n", encoding="UTF-8")
        assert ki.is_anki_note(Path("note.md")) is False

        Path("note.md").write_text("## Note\nnid: 00000000000000a\n", encoding="UTF-8")
        assert ki.is_anki_note(Path("note.md")) is False

        Path("note.md").write_text("## Note\nnid: 000000000000000\n", encoding="UTF-8")
        assert ki.is_anki_note(Path("note.md")) is True


def test_update_kinote_raises_error_on_too_few_fields():
    """Do we raise an error when the field names don't match up?"""
    collection_path = get_collection_path()
    query = ""

    with Anki(path=collection_path) as a:
        i = set(a.col.find_notes(query)).pop()
        kinote = KiNote(a, a.col.get_note(i))
        field = "data"

        # Note that "Back" field is missing.
        flatnote = FlatNote("title", 0, "Basic", "Default", [], False, {"Front": field})

        with pytest.raises(ValueError):
            ki.update_kinote(kinote, flatnote)


def test_update_kinote_raises_error_on_too_many_fields():
    """Do we raise an error when the field names don't match up?"""
    collection_path = get_collection_path()
    query = ""

    with Anki(path=collection_path) as a:
        i = set(a.col.find_notes(query)).pop()
        kinote = KiNote(a, a.col.get_note(i))
        field = "data"

        # Note that "Back" field is missing.
        flatnote = FlatNote(
            "title",
            0,
            "Basic",
            "Default",
            [],
            False,
            {"Front": field, "Back": field, "Left": field},
        )

        with pytest.raises(ValueError):
            ki.update_kinote(kinote, flatnote)


def test_update_kinote_raises_error_wrong_field_name():
    """Do we raise an error when the field names don't match up?"""
    collection_path = get_collection_path()
    query = ""

    with Anki(path=collection_path) as a:
        i = set(a.col.find_notes(query)).pop()
        kinote = KiNote(a, a.col.get_note(i))
        field = "data"

        # Note that "Back" field is missing.
        flatnote = FlatNote(
            "title", 0, "Basic", "Default", [], False, {"Front": field, "Backus": field}
        )

        with pytest.raises(ValueError):
            ki.update_kinote(kinote, flatnote)


def test_update_kinote_sets_tags():
    """Do we update tags of anki note?"""
    collection_path = get_collection_path()
    query = ""
    with Anki(path=collection_path) as a:
        i = set(a.col.find_notes(query)).pop()
        kinote = KiNote(a, a.col.get_note(i))
        field = "data"
        flatnote = FlatNote(
            "title",
            0,
            "Basic",
            "Default",
            ["tag"],
            False,
            {"Front": field, "Back": field},
        )

        assert kinote.n.tags == []
        ki.update_kinote(kinote, flatnote)
        assert kinote.n.tags == ["tag"]


def test_update_kinote_sets_deck():
    collection_path = get_collection_path()
    query = ""
    with Anki(path=collection_path) as a:
        i = set(a.col.find_notes(query)).pop()
        kinote = KiNote(a, a.col.get_note(i))
        field = "data"
        flatnote = FlatNote(
            "title", 0, "Basic", "deck", [], False, {"Front": field, "Back": field}
        )

        assert kinote.get_deck() == "Default"
        ki.update_kinote(kinote, flatnote)
        assert kinote.get_deck() == "deck"


def test_update_kinote_sets_field_contents():
    collection_path = get_collection_path()
    query = ""
    with Anki(path=collection_path) as a:
        i = set(a.col.find_notes(query)).pop()
        kinote = KiNote(a, a.col.get_note(i))
        field = "TITLE\ndata"
        flatnote = FlatNote(
            "title", 0, "Basic", "Default", [], True, {"Front": field, "Back": field}
        )

        assert "TITLE" not in kinote.n.fields[0]

        ki.update_kinote(kinote, flatnote)

        assert "TITLE" in kinote.n.fields[0]
        assert "</p>" in kinote.n.fields[0]


def test_update_kinote_removes_field_contents():
    collection_path = get_collection_path()
    query = ""
    with Anki(path=collection_path) as a:
        i = set(a.col.find_notes(query)).pop()
        kinote = KiNote(a, a.col.get_note(i))
        field = "c"
        flatnote = FlatNote(
            "title", 0, "Basic", "Default", [], False, {"Front": field, "Back": field}
        )

        assert "a" in kinote.n.fields[0]
        ki.update_kinote(kinote, flatnote)
        assert "a" not in kinote.n.fields[0]


def test_update_kinote_raises_error_on_nonexistent_notetype_name():
    collection_path = get_collection_path()
    query = ""
    with Anki(path=collection_path) as a:
        i = set(a.col.find_notes(query)).pop()
        kinote = KiNote(a, a.col.get_note(i))
        field = "c"
        flatnote = FlatNote(
            "title",
            0,
            "nonbasic",
            "Default",
            [],
            False,
            {"Front": field, "Back": field},
        )

        with pytest.raises(FileNotFoundError):
            ki.update_kinote(kinote, flatnote)


def test_display_fields_health_warning_catches_missing_clozes(capfd):
    collection_path = get_collection_path()
    query = ""
    with Anki(path=collection_path) as a:
        i = set(a.col.find_notes(query)).pop()
        KiNote(a, a.col.get_note(i))
        field = "c"
        flatnote = FlatNote(
            "title", 0, "Cloze", "Default", [], False, {"Text": field, "Back Extra": ""}
        )
        result = ki.add_note_from_flatnote(a, flatnote)
        captured = capfd.readouterr()
        assert result is None
        assert "unknown error code" in captured.err


def test_display_fields_health_warning_catches_empty_notes():
    collection_path = get_collection_path()
    query = ""
    with Anki(path=collection_path) as a:
        i = set(a.col.find_notes(query)).pop()
        note = a.col.get_note(i)
        note.fields = []
        health = ki.display_fields_health_warning(note)
        assert health == 1


def test_slugify():
    text = "\u1234"
    result = ki.slugify(text, allow_unicode=False)

    # Filter out Ethiopian syllable see.
    assert result == ""


def test_add_note_from_flatnote_returns_kinote():
    collection_path = get_collection_path()
    with Anki(path=collection_path) as a:
        field = "x"
        flatnote = FlatNote(
            "title", 0, "Basic", "Default", ["tag"], False, {"Front": field, "Back": field}
        )
        result = ki.add_note_from_flatnote(a, flatnote)
        assert isinstance(result, KiNote)


def test_add_note_from_flatnote_returns_markdown_parsed_kinote():
    collection_path = get_collection_path()
    with Anki(path=collection_path) as a:
        field = "*hello*"
        flatnote = FlatNote(
            "title", 0, "Basic", "Default", ["tag"], True, {"Front": field, "Back": field}
        )
        result = ki.add_note_from_flatnote(a, flatnote)
        assert isinstance(result, KiNote)
        assert "<em>hello</em>" in result.n.fields[0]


def test_get_note_files_changed_since_last_push(capfd):
    collection_path = get_collection_path()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, collection_path)
        repo = git.Repo(REPODIR)

        last_push_path = Path(repo.working_dir) / ".ki" / "last_push"
        last_push_path.write_text("")

        changed = list(map(str, ki.get_note_files_changed_since_last_push(repo)))
        captured = capfd.readouterr()
        assert changed == ['collection/Default/c.md', 'collection/Default/a.md']
        assert "last_push" not in captured.err


def test_get_note_files_changed_since_last_push_when_last_push_file_is_missing(capfd):
    collection_path = get_collection_path()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, collection_path)
        repo = git.Repo(REPODIR)

        last_push_path = Path(repo.working_dir) / ".ki" / "last_push"
        os.remove(last_push_path)

        changed = list(map(str, ki.get_note_files_changed_since_last_push(repo)))
        captured = capfd.readouterr()
        assert changed == ['collection/Default/c.md', 'collection/Default/a.md']
        assert "last_push" in captured.err


def test_backup_is_no_op_when_backup_already_exists(capfd):
    collection_path = get_collection_path()
    runner = CliRunner()
    with runner.isolated_filesystem():
        clone(runner, collection_path)
        os.chdir(REPODIR)

        ki.backup(Path(collection_path))
        ki.backup(Path(collection_path))
        captured = capfd.readouterr()
        assert "Backup already exists." in captured.out


def test_git_subprocess_pull():
    collection_path = get_collection_path()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, collection_path)

        # Edit collection.
        shutil.copyfile(EDITED_COLLECTION_PATH, collection_path)
        os.chdir(REPODIR)

        # Pull, poorly.
        with pytest.raises(ValueError):
            ki.git_subprocess_pull("anki", "main")


def test_get_notepath():
    collection_path = get_collection_path()
    query = ""
    runner = CliRunner()
    with runner.isolated_filesystem(), Anki(path=collection_path) as a:
        i = set(a.col.find_notes(query)).pop()
        kinote = KiNote(a, a.col.get_note(i))

        deckpath = Path(".")
        dupe_path = deckpath / "a.md"
        dupe_path.write_text("ay")
        notepath = ki.get_notepath(kinote, "Front", deckpath)
        assert str(notepath) == "a_1.md"


def test_tidy_html_recursively():
    """Does tidy wrapper print a nice error when tidy is missing?"""
    runner = CliRunner()
    with runner.isolated_filesystem():
        root = Path(".")
        file = root / "a.html"
        file.write_text("ay")
        old_path = os.environ["PATH"]
        try:
            os.environ["PATH"] = ""
            with pytest.raises(FileNotFoundError):
                ki.tidy_html_recursively(root, False)
        finally:
            os.environ["PATH"] = old_path
