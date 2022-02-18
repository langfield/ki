#!/usr/bin/env python3
"""Tests for ki command line interface (CLI)."""
import os
import random
import shutil
import tempfile
import subprocess
from distutils.dir_util import copy_tree
from importlib.metadata import version

import git
import pytest
import bitstring
import checksumdir
from loguru import logger
from apy.anki import Anki
from click.testing import CliRunner

from beartype import beartype
from beartype.typing import List

import ki
from ki.note import KiNote


# pylint:disable=unnecessary-pass, too-many-lines


TEST_DATA_PATH = "tests/data/"
COLLECTIONS_PATH = os.path.join(TEST_DATA_PATH, "collections/")
COLLECTION_FILENAME = "collection.anki2"
ORIG_COLLECTION_FILENAME = "original.anki2"
EDITED_COLLECTION_FILENAME = "edited.anki2"
COLLECTION_PATH = os.path.abspath(
    os.path.join(COLLECTIONS_PATH, ORIG_COLLECTION_FILENAME)
)
EDITED_COLLECTION_PATH = os.path.abspath(
    os.path.join(COLLECTIONS_PATH, EDITED_COLLECTION_FILENAME)
)
GITREPO_PATH = os.path.join(TEST_DATA_PATH, "gitrepo/")
REPODIR = os.path.splitext(COLLECTION_FILENAME)[0]

NOTE_0 = "note1645010162168.md"
NOTE_1 = "note1645222430007.md"
NOTE_2 = os.path.abspath("tests/data/note123412341234.md")
NOTE_3 = os.path.abspath("tests/data/note 3.md")

# HELPER FUNCTIONS


def invoke(*args, **kwargs):
    """Wrap click CliRunner invoke()."""
    return CliRunner().invoke(*args, **kwargs)


@beartype
def clone(runner: CliRunner, repository: str, directory: str = "") -> None:
    """Make a test `ki clone` call."""
    runner.invoke(
        ki.ki,
        ["clone", repository, directory],
        standalone_mode=False,
        catch_exceptions=False,
    )


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
            notes.append(KiNote(a, a.col.getNote(i)))
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

        # TODO: Update hash literals.
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


# CLONE


def test_clone_fails_if_collection_doesnt_exist():
    """Does ki clone only if `.anki2` file exists?"""
    collection_path = get_collection_path()
    os.remove(collection_path)
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        with pytest.raises(FileNotFoundError):
            clone(runner, collection_path)


def test_clone_creates_directory():
    """Does it create the directory?"""
    collection_path = get_collection_path()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, collection_path)

        assert os.path.isdir(REPODIR)


def test_clone_errors_when_directory_already_exists():
    """Does it disallow overwrites?"""
    collection_path = get_collection_path()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.

        # Create directory where we want to clone.
        os.mkdir(REPODIR)

        # Should error out because directory already exists.
        with pytest.raises(FileExistsError):
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

        # Make sure there are exactly 2 commits.
        commits = list(repo.iter_commits("HEAD"))
        assert len(commits) == 2


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


# PUSH


@pytest.mark.skip
def test_push_writes_changes_correctly():
    """If there are committed changes, does push change the collection file?"""
    collection_path = get_collection_path()
    old_notes = get_notes(collection_path)
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, collection_path)

        note = os.path.join(REPODIR, NOTE_0)
        with open(note, "a", encoding="UTF-8") as note_file:
            note_file.write("e\n")

        # Commit.
        repo = git.Repo(REPODIR)
        repo.git.add(all=True)
        repo.index.commit("Added 'e'.")

        # Push and check for changes.
        os.chdir(REPODIR)
        push(runner)
        new_notes = get_notes(collection_path)
        assert len(old_notes) == len(new_notes) == 1

        for old, new in zip(old_notes, new_notes):
            assert str(old) + "e\n" == str(new)


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


@pytest.mark.skip
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

        logger.debug(os.listdir())

        # Modify local file.
        note = os.path.join(NOTE_0)
        with open(note, "a", encoding="UTF-8") as note_file:
            note_file.write("e\n")

        # Add new file.
        shutil.copyfile(NOTE_2, os.path.basename(NOTE_2))
        # Add new file.
        shutil.copyfile(NOTE_3, os.path.basename(NOTE_3))

        # Commit.
        os.chdir("../")
        repo = git.Repo(REPODIR)
        repo.git.add(all=True)
        repo.index.commit("Added 'e'.")

        # Push changes.
        os.chdir(REPODIR)
        push(runner)


# UTILS


def test_get_default_clone_directory():
    """Does it generate the right path?"""
    path = ki.get_default_clone_directory("collection.anki2")
    assert path == os.path.abspath("./collection")

    path = ki.get_default_clone_directory("sensors.anki2")
    assert path == os.path.abspath("./sensors")
