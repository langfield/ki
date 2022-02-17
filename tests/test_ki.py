"""Tests for ki command line interface (CLI)."""
import os
import random
import shutil
import hashlib
import sqlite3
import difflib
import tempfile
import subprocess
from distutils.dir_util import copy_tree
from importlib.metadata import version

import git
import pytest
import bitstring
import checksumdir
from loguru import logger
from beartype import beartype
from click.testing import CliRunner

import ki


# pylint:disable=unnecessary-pass


TEST_DATA_PATH = "tests/data/"
COLLECTION_FILENAME = "collection.anki2"
ORIG_COLLECTION_FILENAME = "original.anki2"
UPDATED_COLLECTION_FILENAME = "updated.anki2"
COLLECTION_PATH = os.path.abspath(
    os.path.join(TEST_DATA_PATH, ORIG_COLLECTION_FILENAME)
)
UPDATED_COLLECTION_PATH = os.path.abspath(
    os.path.join(TEST_DATA_PATH, UPDATED_COLLECTION_FILENAME)
)
GITREPO_PATH = os.path.join(TEST_DATA_PATH, "gitrepo/")
REPODIR = os.path.splitext(COLLECTION_FILENAME)[0]


# HELPER FUNCTIONS


def invoke(*args, **kwargs):
    """Wrap click CliRunner invoke()."""
    return CliRunner().invoke(*args, **kwargs)


@beartype
def clone(runner: CliRunner, repository: str, directory: str = "") -> None:
    """Make a test `ki clone` call."""
    runner.invoke(ki.ki, ["clone", repository, directory], standalone_mode=False, catch_exceptions=False)


@beartype
def pull(runner: CliRunner) -> None:
    """Make a test `ki pull` call."""
    runner.invoke(ki.ki, ["pull"], standalone_mode=False, catch_exceptions=False)


@beartype
def push(runner: CliRunner) -> None:
    """Make a test `ki push` call."""
    runner.invoke(ki.ki, ["push"], standalone_mode=False, catch_exceptions=False)


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
def read_sqlite3(db_path: str) -> None:
    """Print all tables."""
    # Create a SQL connection to our SQLite database
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    tables = list(cur.execute("SELECT name FROM sqlite_master WHERE type = 'table'"))
    print(tables)
    con.close()


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


@pytest.mark.skip
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
            assert "f6945f2bb37aef63d57e76f915d3a97f  collection.anki2" in hashes
            assert "a68250f8ee3dc8302534f908bcbafc6a  collection.anki2" not in hashes

        # Update collection.
        shutil.copyfile(UPDATED_COLLECTION_PATH, collection_path)

        # Pull updated collection.
        os.chdir(REPODIR)
        pull(runner)
        os.chdir("../")

        # Check that updated hash is written and old hash is still there.
        with open(os.path.join(REPODIR, ".ki/hashes"), encoding="UTF-8") as hashes_file:
            hashes = hashes_file.read()
            assert "f6945f2bb37aef63d57e76f915d3a97f  collection.anki2" in hashes
            assert "a68250f8ee3dc8302534f908bcbafc6a  collection.anki2" in hashes


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
    true_note_path = os.path.abspath(os.path.join(GITREPO_PATH, "note0.md"))
    cloned_note_path = os.path.join(REPODIR, "note0.md")
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
        assert not os.path.isfile(os.path.join(REPODIR, "note1.md"))

        # Update collection.
        shutil.copyfile(UPDATED_COLLECTION_PATH, collection_path)

        # Pull updated collection.
        os.chdir(REPODIR)
        pull(runner)
        assert os.path.isfile("note1.md")


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


def test_push_writes_changes_correctly():
    """If there are committed changes, does push change the collection file?"""
    collection_path = get_collection_path()
    read_sqlite3(collection_path)
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, collection_path)

        note = os.path.join(REPODIR, "note0.md")
        with open(note, "a", encoding="UTF-8") as note_file:
            note_file.write("e\n")

        push(runner)
        read_sqlite3(collection_path)
        raise NotImplementedError


@pytest.mark.skip
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
        with pytest.raises(ValueError):
            push(runner)


@pytest.mark.skip
def test_push_generates_correct_backup():
    """Does push store a backup identical to old collection file?"""
    collection_path = get_collection_path()
    old_hash = ki.md5(collection_path)
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, collection_path)

        # Make change in repo.
        note = os.path.join(REPODIR, "note0.md")
        with open(note, "a", encoding="UTF-8") as note_file:
            note_file.write("e\n")

        push(runner)
        os.chdir(REPODIR)
        assert os.path.isdir(".ki/backups")

        os.chdir(".ki/backups")
        paths = os.listdir()

        backup = False
        for path in paths:
            if ki.md5(path) == old_hash:
                backup = True

        assert backup


# UTILS


def test_get_default_clone_directory():
    """Does it generate the right path?"""
    path = ki.get_default_clone_directory("collection.anki2")
    assert path == os.path.abspath("./collection")

    path = ki.get_default_clone_directory("sensors.anki2")
    assert path == os.path.abspath("./sensors")
