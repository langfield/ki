"""Tests for ki command line interface (CLI)."""
import os
import shutil
import hashlib
import tempfile
from distutils.dir_util import copy_tree
from importlib.metadata import version

import git
import pytest
from loguru import logger
from beartype import beartype
from click.testing import CliRunner
from cli_test_helpers import shell

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


# HELPER FUNCTIONS


@beartype
def invoke(*args, **kwargs) -> int:
    """Wrap click CliRunner invoke()."""
    return CliRunner().invoke(*args, **kwargs)


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
    try:
        _ = git.Repo(path).git_dir
        return True
    except git.exc.InvalidGitRepositoryError:
        return False


# CLI


def test_bad_command_is_bad():
    """Typos should result in errors."""
    result = invoke(ki.ki, ["clome"])
    assert result.exit_code == 2
    assert "Error: No such command 'clome'." in result.output


def test_runas_module():
    """Can this package be run as a Python module?"""
    result = shell("python -m ki --help")
    assert result.exit_code == 0


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


def test_pull_push_fails_without_ki_subdirectory():
    """Do pull and push know whether they're in a ki-generated git repo?"""
    gitrepo_path = os.path.abspath(GITREPO_PATH)
    runner = CliRunner()
    with runner.isolated_filesystem():
        tempdir = tempfile.mkdtemp()
        copy_tree(gitrepo_path, tempdir)
        os.chdir(tempdir)
        with pytest.raises(FileNotFoundError):
            ki.pull()
        with pytest.raises(FileNotFoundError):
            ki.push()


def test_clone_pull_compute_and_store_md5sum():
    """Does ki add new hash to `.ki/hashes`?"""
    collection_path = get_collection_path()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        ki.clone(collection_path)
        repodir = os.path.splittext(COLLECTION_FILENAME)[0]

        # Check that hash is written.
        with open(os.path.join(repodir, ".ki/hashes"), encoding="UTF-8") as hashes_file:
            hashes = hashes_file.read()
            assert "f6945f2bb37aef63d57e76f915d3a97f  collection.anki2" in hashes
            assert "a68250f8ee3dc8302534f908bcbafc6a  collection.anki2" not in hashes

        # Update collection.
        shutil.copyfile(UPDATED_COLLECTION_PATH, collection_path)

        # Pull updated collection.
        os.chdir(repodir)
        ki.pull()
        os.chdir("../")

        # Check that updated hash is written and old hash is still there.
        with open(os.path.join(repodir, ".ki/hashes"), encoding="UTF-8") as hashes_file:
            hashes = hashes_file.read()
            assert "f6945f2bb37aef63d57e76f915d3a97f  collection.anki2" in hashes
            assert "a68250f8ee3dc8302534f908bcbafc6a  collection.anki2" in hashes


# CLONE


def test_get_default_clone_directory():
    """Does it generate the right path?"""
    path = ki.get_default_clone_directory("collection.anki2")
    assert path == os.path.abspath("./collection")

    path = ki.get_default_clone_directory("sensors.anki2")
    assert path == os.path.abspath("./sensors")


def test_clone_creates_directory():
    """Does it create the directory?"""
    collection_path = get_collection_path()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        ki.clone(collection_path)
        repodir = os.path.splittext(COLLECTION_FILENAME)[0]

        assert os.path.isdir(repodir)


def test_clone_errors_when_directory_already_exists():
    """Does it disallow overwrites?"""
    collection_path = get_collection_path()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        repodir = os.path.splittext(COLLECTION_FILENAME)[0]

        # Create directory where we want to clone.
        os.mkdir(repodir)

        # Should error out because directory already exists.
        with pytest.raises(FileNotFoundError):
            ki.clone(collection_path)


def test_clone_generates_expected_notes():
    """Do generated note files match content of an example collection?"""
    collection_path = get_collection_path()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        ki.clone(collection_path)
        repodir = os.path.splittext(COLLECTION_FILENAME)[0]

        # Compute hashes.
        cloned_md5 = hashlib.md5(os.path.join(repodir, "note.md"))
        true_md5 = hashlib.md5(os.path.join(GITREPO_PATH, "note.md"))

        assert cloned_md5 == true_md5


def test_clone_generates_ki_subdirectory():
    """Does clone command generate .ki/ directory?"""
    collection_path = get_collection_path()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        ki.clone(collection_path)
        repodir = os.path.splittext(COLLECTION_FILENAME)[0]

        # Check kidir exists.
        kidir = os.path.join(repodir, ".ki/")
        assert os.path.isdir(kidir)


def test_cloned_collection_is_git_repository():
    """Does clone run `git init` and stuff?"""
    collection_path = get_collection_path()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        ki.clone(collection_path)
        repodir = os.path.splittext(COLLECTION_FILENAME)[0]

        assert is_git_repo(repodir)


def test_clone_commits_directory_contents():
    """Does clone leave user with an up-to-date repo?"""
    pass


def test_clone_leaves_collection_file_unchanged():
    """Does clone leave the collection alone?"""
    pass


# PULL


def test_pull_fails_if_collection_no_longer_exists():
    """Does ki pull only if `.anki2` file exists?"""
    pass


def test_pull_writes_changes_correctly():
    """Does ki get the changes from modified collection file?"""
    pass


def test_pull_deletes_ephemeral_repo_directory():
    """Does ki clean up `/tmp/ki/remote/`?"""
    pass


def test_pull_creates_ephemeral_repo_directory():
    """Does ki first clone to `/tmp/ki/remote/`?"""
    pass


def test_pull_removes_ephemeral_git_remote():
    """Does ki remove remote before quitting?"""
    pass


# PUSH


def test_push_writes_changes_correctly():
    """If there are committed changes, does push change the collection file?"""
    pass


def test_push_verifies_md5sum():
    """Does ki only push if md5sum matches last pull?"""
    pass


def test_push_generates_correct_backup():
    """Does push store a backup identical to old collection file?"""
    pass
