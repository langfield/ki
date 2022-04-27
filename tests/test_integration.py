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
import prettyprinter as pp
from loguru import logger
from result import Ok, Err, OkErr
from pytest_mock import MockerFixture
from click.testing import CliRunner
from anki.collection import Collection

from beartype import beartype
from beartype.typing import List

import ki
import ki.maybes as M
import ki.functional as F
from ki import BRANCH_NAME, get_colnote
from ki.types import (
    KiRepo,
    ColNote,
    RepoRef,
    Notetype,
    KiRepoRef,
    ExtantDir,
    ExtantFile,
    MissingFileError,
    TargetExistsError,
    NotKiRepoError,
    UpdatesRejectedError,
    SQLiteLockError,
    ExpectedNonexistentPathError,
    PathCreationCollisionError,
    GitRefNotFoundError,
    GitHeadRefNotFoundError,
    CollectionChecksumError,
    MissingFieldOrdinalError,
)
from tests.test_ki import open_collection


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
GITREPO_PATH = os.path.abspath(os.path.join(TEST_DATA_PATH, "repos/", "original/"))
MULTI_GITREPO_PATH = os.path.join(TEST_DATA_PATH, "repos/", "multideck/")
REPODIR = os.path.splitext(COLLECTION_FILENAME)[0]
MULTIDECK_REPODIR = os.path.splitext(MULTIDECK_COLLECTION_FILENAME)[0]
HTML_REPODIR = os.path.splitext(HTML_COLLECTION_FILENAME)[0]
MULTI_NOTE_PATH = "aa/bb/cc/cc.md"

NOTES_PATH = os.path.abspath(os.path.join(TEST_DATA_PATH, "notes/"))
SUBMODULE_DIRNAME = "submodule"

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

    return repo


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


@beartype
def test_fails_without_ki_subdirectory(tmp_path: Path):
    """Do pull and push know whether they're in a ki-generated git repo?"""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        tempdir = tempfile.mkdtemp()
        copy_tree(GITREPO_PATH, tempdir)
        os.chdir(tempdir)
        with pytest.raises(NotKiRepoError):
            pull(runner)
        with pytest.raises(NotKiRepoError):
            push(runner)


@beartype
def test_computes_and_stores_md5sum(tmp_path: Path):
    """Does ki add new hash to `.ki/hashes`?"""
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        # Clone collection in cwd.
        clone(runner, col_file)

        # Check that hash is written.
        with open(os.path.join(REPODIR, ".ki/hashes"), encoding="UTF-8") as hashes_file:
            hashes = hashes_file.read()
            assert "a68250f8ee3dc8302534f908bcbafc6a  collection.anki2" in hashes
            assert "199216c39eeabe23a1da016a99ffd3e2  collection.anki2" not in hashes

        # Edit collection.
        shutil.copyfile(EDITED_COLLECTION_PATH, col_file)

        logger.debug(f"CWD: {F.cwd()}")

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
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, col_file)
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
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():
        out = clone(runner, col_file)
        logger.debug(f"\nCLONE:\n{out}")

        # Edit collection.
        shutil.copyfile(EDITED_COLLECTION_PATH, col_file)

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
    col_file = get_col_file()
    os.remove(col_file)
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        with pytest.raises(FileNotFoundError):
            clone(runner, col_file)
        assert not os.path.isdir(REPODIR)


def test_clone_creates_directory():
    """Does it create the directory?"""
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, col_file)

        assert os.path.isdir(REPODIR)


@beartype
def test_clone_displays_errors_from_creation_of_staging_kirepo(mocker: MockerFixture):
    """Do errors get displayed nicely?"""
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        mocker.patch(
            "ki.get_ephemeral_kirepo",
            return_value=Err(ExpectedNonexistentPathError(Path("path-that-exists"))),
        )
        with pytest.raises(ExpectedNonexistentPathError):
            clone(runner, col_file)


@beartype
def test_clone_displays_errors_from_creation_of_kirepo_metadata(mocker: MockerFixture):
    """Do errors get displayed nicely?"""
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        directory = ExtantDir(Path("directory"))
        collision = PathCreationCollisionError(directory, "token")
        mocker.patch("ki.F.fmkleaves", return_value=Err(collision))

        with pytest.raises(PathCreationCollisionError):
            clone(runner, col_file)


@beartype
def test_clone_displays_errors_from_head_kirepo_ref(mocker: MockerFixture):
    """Do errors get displayed nicely?"""
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():
        directory = F.force_mkdir(Path("repo"))
        repo = git.Repo.init(directory)

        mocker.patch(
            "ki.M.head_kirepo_ref",
            return_value=Err(
                GitHeadRefNotFoundError(
                    repo, ValueError("<failed_to_find_HEAD_exception>")
                )
            ),
        )

        with pytest.raises(GitHeadRefNotFoundError):
            clone(runner, col_file)


@beartype
def test_clone_displays_errors_from_loading_kirepo_at_end(mocker: MockerFixture):
    """Do errors get propagated in the places we expect?"""
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # `M.kirepo()` is called four times in `clone()`, and we only want to
        # actually mock the last call. So we need the mocked function to behave
        # somewhat normally for the first three calls, which means returning a
        # valid `kirepo`. And we can't return a mock because beartype will
        # catch it. We need to return an *actual* kirepo or mess with the
        # `__class__` attr of a mock (seems dangerous).

        # So we actually clone three times in three separate directories, and
        # instantiate a kirepo from each. The `clone()` call will do some
        # copying between them, but since they're distinct locations, it will
        # think everything is working fine.

        # So we pass a iterable as the `side_effect` of our mock, and return
        # our three 'fake' kirepos, and then finally the `Err` object we
        # actually needed on the fourth call.
        os.mkdir("A")
        os.chdir("A")
        clone(runner, col_file)
        os.chdir("..")
        A_kirepo = M.kirepo(F.test(Path("A") / REPODIR))

        os.mkdir("B")
        os.chdir("B")
        clone(runner, col_file)
        os.chdir("..")
        B_kirepo = M.kirepo(F.test(Path("B") / REPODIR))

        os.mkdir("C")
        os.chdir("C")
        clone(runner, col_file)
        os.chdir("..")
        C_kirepo = M.kirepo(F.test(Path("C") / REPODIR))

        mocker.patch(
            "ki.M.kirepo",
            side_effect=[A_kirepo, B_kirepo, C_kirepo, Err(NotKiRepoError())],
        )
        with pytest.raises(NotKiRepoError):
            clone(runner, col_file)


def test_clone_handles_html():
    """Does it tidy html and stuff?"""
    col_file = get_html_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, col_file)
        assert os.path.isdir(HTML_REPODIR)

        path = Path(".") / "html" / "Default" / "あだ名.md"
        contents = path.read_text()
        assert "<!DOCTYPE html>" in contents


def test_clone_errors_when_directory_is_populated():
    """Does it disallow overwrites?"""
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Create directory where we want to clone.
        os.mkdir(REPODIR)
        with open(os.path.join(REPODIR, "hi"), "w", encoding="UTF-8") as hi_file:
            hi_file.write("hi\n")

        # Should error out because directory already exists.
        with pytest.raises(TargetExistsError):
            out = clone(runner, col_file)


def test_clone_cleans_up_on_error():
    """Does it clean up on nontrivial errors?"""
    col_file = get_html_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        clone(runner, col_file)
        assert os.path.isdir(HTML_REPODIR)
        shutil.rmtree(HTML_REPODIR)
        old_path = os.environ["PATH"]
        try:
            with pytest.raises(FileNotFoundError):
                os.environ["PATH"] = ""
                out = clone(runner, col_file)
            assert not os.path.isdir(HTML_REPODIR)
        finally:
            os.environ["PATH"] = old_path


# TODO: Consider writing new `Exception` subclasses that print a slightly
# prettier message, informing the user of how to install the relevant missing
# dependency.
def test_clone_displays_nice_errors_for_missing_dependencies():
    """Does it tell the user what to install?"""
    col_file = get_html_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        clone(runner, col_file)
        assert os.path.isdir(HTML_REPODIR)
        shutil.rmtree(HTML_REPODIR)
        old_path = os.environ["PATH"]

        # In case where nothing is installed, we expect to fail on `tidy`
        # first.
        try:
            with pytest.raises(FileNotFoundError) as raised:
                os.environ["PATH"] = ""
                out = clone(runner, col_file)
            error = raised.exconly()
            assert "tidy" in str(error)
        finally:
            os.environ["PATH"] = old_path

        # If `tidy` is on the PATH, but nothing else, then we expect a
        # `GitCommandNotFound` error.
        try:
            with pytest.raises(git.GitCommandNotFound) as raised:
                tmp = F.mkdtemp()
                os.symlink("/usr/bin/tidy", tmp / "tidy")
                os.environ["PATH"] = str(tmp)
                out = clone(runner, col_file)
            error = raised.exconly()
        finally:
            os.environ["PATH"] = old_path


def test_clone_succeeds_when_directory_exists_but_is_empty():
    """Does it clone into empty directories?"""
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Create directory where we want to clone.
        os.mkdir(REPODIR)
        clone(runner, col_file)


def test_clone_generates_expected_notes():
    """Do generated note files match content of an example collection?"""
    true_note_path = os.path.join(GITREPO_PATH, NOTE_0)
    cloned_note_path = os.path.join(REPODIR, NOTE_0)
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, col_file)

        # Check that deck directory is created.
        assert os.path.isdir(os.path.join(REPODIR, "Default"))

        # Compute hashes.
        cloned_md5 = F.md5(ExtantFile(cloned_note_path))
        true_md5 = F.md5(ExtantFile(true_note_path))

        assert cloned_md5 == true_md5


def test_clone_generates_deck_tree_correctly():
    """Does generated FS tree match example collection?"""
    true_note_path = os.path.abspath(os.path.join(MULTI_GITREPO_PATH, MULTI_NOTE_PATH))
    cloned_note_path = os.path.join(MULTIDECK_REPODIR, MULTI_NOTE_PATH)
    col_file = get_multideck_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        out = clone(runner, col_file)
        logger.debug(f"\n{out}")

        # Check that deck directory is created and all subdirectories.
        assert os.path.isdir(os.path.join(MULTIDECK_REPODIR, "Default"))
        assert os.path.isdir(os.path.join(MULTIDECK_REPODIR, "aa/bb/cc"))
        assert os.path.isdir(os.path.join(MULTIDECK_REPODIR, "aa/dd"))

        # Compute hashes.
        cloned_md5 = F.md5(ExtantFile(cloned_note_path))
        true_md5 = F.md5(ExtantFile(true_note_path))

        assert cloned_md5 == true_md5


def test_clone_generates_ki_subdirectory():
    """Does clone command generate .ki/ directory?"""
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, col_file)

        # Check kidir exists.
        kidir = os.path.join(REPODIR, ".ki/")
        assert os.path.isdir(kidir)


def test_cloned_collection_is_git_repository():
    """Does clone run `git init` and stuff?"""
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, col_file)

        assert is_git_repo(REPODIR)


def test_clone_commits_directory_contents():
    """Does clone leave user with an up-to-date repo?"""
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, col_file)

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
    col_file = get_col_file()
    original_md5 = F.md5(col_file)
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, col_file)

        updated_md5 = F.md5(col_file)
        assert original_md5 == updated_md5


def test_clone_directory_argument_works():
    """Does clone obey the target directory argument?"""
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        tempdir = tempfile.mkdtemp()
        target = os.path.join(tempdir, "TARGET")
        assert not os.path.isdir(target)
        assert not os.path.isfile(target)

        # Clone collection in cwd.
        clone(runner, col_file, target)
        assert os.path.isdir(target)


# PULL


def test_pull_fails_if_collection_no_longer_exists():
    """Does ki pull only if `.anki2` file exists?"""
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, col_file)

        # Delete collection and try to pull.
        os.remove(col_file)
        with pytest.raises(FileNotFoundError):
            os.chdir(REPODIR)
            pull(runner)


def test_pull_fails_if_collection_file_is_corrupted():
    """Does `pull()` fail gracefully when the collection file is bad?"""
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, col_file)

        # Overwrite collection and try to pull.
        col_file.write_text("bad_contents")

        os.chdir(REPODIR)
        with pytest.raises(SQLiteLockError):
            pull(runner)


def test_pull_writes_changes_correctly():
    """Does ki get the changes from modified collection file?"""
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, col_file)
        assert not os.path.isfile(os.path.join(REPODIR, NOTE_1))

        # Edit collection.
        shutil.copyfile(EDITED_COLLECTION_PATH, col_file)

        # Pull edited collection.
        os.chdir(REPODIR)
        pull(runner)
        assert os.path.isfile(NOTE_1)


def test_pull_unchanged_collection_is_no_op():
    """Does ki remove remote before quitting?"""
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, col_file)
        orig_hash = checksum_git_repository(REPODIR)

        # Pull updated collection.
        os.chdir(REPODIR)
        pull(runner)
        os.chdir("../")
        new_hash = checksum_git_repository(REPODIR)

        assert orig_hash == new_hash


def test_pull_avoids_unnecessary_merge_conflicts():
    """Does ki prevent gratuitous merge conflicts?"""
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, col_file)
        assert not os.path.isfile(os.path.join(REPODIR, NOTE_1))

        # Edit collection.
        shutil.copyfile(EDITED_COLLECTION_PATH, col_file)

        # Pull edited collection.
        os.chdir(REPODIR)
        out = pull(runner)
        assert "Automatic merge failed; fix" not in out


def test_pull_still_works_from_subdirectories():
    """Does pull still work if you're farther down in the directory tree than the repo route?"""
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, col_file)
        assert not os.path.isfile(os.path.join(REPODIR, NOTE_1))

        # Edit collection.
        shutil.copyfile(EDITED_COLLECTION_PATH, col_file)

        # Pull edited collection.
        os.chdir(os.path.join(REPODIR, "Default"))
        pull(runner)


def test_pull_displays_errors_from_repo_ref():
    """Does 'pull()' return early when the last push commit ref is bad?"""
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, col_file)

        kirepo: KiRepo = M.kirepo(F.test(Path(REPODIR))).unwrap()
        kirepo.last_push_file.write_text("gibberish")

        # Edit collection.
        shutil.copyfile(EDITED_COLLECTION_PATH, col_file)

        os.chdir(REPODIR)
        with pytest.raises(GitRefNotFoundError):
            pull(runner)


def test_pull_displays_errors_from_clone_helper(mocker: MockerFixture):
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, col_file)

        # Edit collection.
        shutil.copyfile(EDITED_COLLECTION_PATH, col_file)

        directory = F.force_mkdir(Path("directory"))
        collision = PathCreationCollisionError(directory, "token")
        mocker.patch("ki.F.fmkleaves", return_value=Err(collision))

        os.chdir(REPODIR)
        with pytest.raises(PathCreationCollisionError):
            pull(runner)


def test_pull_handles_unexpectedly_changed_checksums(mocker: MockerFixture):
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, col_file)

        # Edit collection.
        shutil.copyfile(EDITED_COLLECTION_PATH, col_file)

        directory = F.force_mkdir(Path("directory"))
        checksum = CollectionChecksumError(F.touch(directory, "file"))
        mocker.patch("ki.F.md5", side_effect=["good", "good", "bad"])

        os.chdir(REPODIR)
        with pytest.raises(CollectionChecksumError):
            pull(runner)


def test_pull_displays_errors_from_repo_initialization(mocker: MockerFixture):
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, col_file)

        # Edit collection.
        shutil.copyfile(EDITED_COLLECTION_PATH, col_file)

        directory = F.force_mkdir(Path("repo"))
        repo = git.Repo.init(Path(REPODIR))
        returns = [Ok(repo), Ok(repo), Err(git.InvalidGitRepositoryError())]
        mocker.patch("ki.M.repo", side_effect=returns)

        os.chdir(REPODIR)
        with pytest.raises(git.InvalidGitRepositoryError):
            pull(runner)


# PUSH


def test_push_writes_changes_correctly(tmp_path: Path):
    """If there are committed changes, does push change the collection file?"""
    col_file = get_col_file()
    old_notes = get_notes(col_file)
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):

        # Clone collection in cwd.
        out = clone(runner, col_file)
        logger.debug(f"\n{out}")

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
        out = push(runner)
        logger.debug(f"\n{out}")
        new_notes = get_notes(col_file)

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
        logger.debug(f"OLD:\n{old_notes}")
        logger.debug(f"NEW:\n{new_notes}")
        assert len(old_notes) == 2
        assert len(new_notes) == 2


def test_push_verifies_md5sum():
    """Does ki only push if md5sum matches last pull?"""
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, col_file)

        # Swap a bit.
        randomly_swap_1_bit(col_file)

        # Make sure ki complains.
        os.chdir(REPODIR)
        with pytest.raises(UpdatesRejectedError):
            push(runner)


def test_push_generates_correct_backup():
    """Does push store a backup identical to old collection file?"""
    col_file = get_col_file()
    old_hash = F.md5(col_file)
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, col_file)

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

        backup_exists = False
        for path in paths:
            if F.md5(F.test(Path(path))) == old_hash:
                backup_exists = True

        assert backup_exists


def test_push_doesnt_write_uncommitted_changes():
    """Does push only write changes that have been committed?"""
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, col_file)

        # Make change in repo.
        note = os.path.join(REPODIR, NOTE_0)
        with open(note, "a", encoding="UTF-8") as note_file:
            note_file.write("e\n")

        # DON'T COMMIT, push.
        os.chdir(REPODIR)
        out = push(runner)
        assert "ki push: up to date." in out
        assert len(os.listdir(".ki/backups")) == 0


def test_push_doesnt_fail_after_pull():
    """Does push work if we pull and then edit and then push?"""
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, col_file)
        assert not os.path.isfile(os.path.join(REPODIR, NOTE_1))

        # Edit collection.
        shutil.copyfile(EDITED_COLLECTION_PATH, col_file)

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
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, col_file)
        assert os.path.isdir(REPODIR)

        os.chdir(REPODIR)
        push(runner)
        push(runner)
        push(runner)
        push(runner)
        push(runner)
        push(runner)


def test_push_deletes_notes():
    """Does push remove deleted notes from collection?"""
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, col_file)

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
        clone(runner, col_file)
        assert not os.path.isfile(NOTE_0)


def test_push_still_works_from_subdirectories():
    """Does push still work if you're farther down in the directory tree than the repo route?"""
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, col_file)

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
        os.chdir(os.path.join(REPODIR, "Default"))
        push(runner)


def test_push_deletes_added_notes():
    """Does push remove deleted notes added with ki?"""
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, col_file)

        # Add new files.
        os.chdir(REPODIR)
        contents = os.listdir("Default")
        shutil.copyfile(NOTE_2_PATH, os.path.join("Default", NOTE_2))
        shutil.copyfile(NOTE_3_PATH, os.path.join("Default", NOTE_3))

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
        os.chdir("Default")
        post_push_contents = os.listdir()
        notes = [path for path in post_push_contents if path[-3:] == ".md"]
        assert len(notes) == 4

        # Delete added files.
        for file in post_push_contents:
            if file not in contents:
                logger.debug(f"Removing '{file}'")
                os.remove(file)

        logger.debug(f"Remaining files: {os.listdir()}")

        # Commit the deletions.
        os.chdir("../../")
        repo = git.Repo(REPODIR)
        repo.git.add(all=True)
        repo.index.commit("Added 'e'.")
        os.chdir(REPODIR)

        # Push changes.
        out = push(runner)
        logger.debug(f"\nPUSH:\n{out}")

    # Check that notes are gone.
    with runner.isolated_filesystem():
        clone(runner, col_file)
        contents = os.listdir(os.path.join(REPODIR, "Default"))
        notes = [path for path in contents if path[-3:] == ".md"]
        logger.debug(f"Notes: {notes}")
        assert len(notes) == 2


def test_push_generates_correct_title_for_notes():
    """Does push use the truncated sort field as a filename?"""
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, col_file)

        # Add new files.
        os.chdir(REPODIR)
        shutil.copyfile(NOTE_2_PATH, os.path.join("Default", NOTE_2))

        # Commit the additions.
        os.chdir("../")
        repo = git.Repo(REPODIR)
        repo.git.add(all=True)
        repo.index.commit("Added 'e'.")

        # Push changes.
        os.chdir(REPODIR)
        out = push(runner)
        logger.debug(f"\nPUSH:\n{out}")

        os.chdir("Default")
        post_push_contents = os.listdir()
        notes = [path for path in post_push_contents if path[-3:] == ".md"]
        assert "r.md" in notes


def test_push_displays_informative_error_when_last_push_file_is_missing(capfd):
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, col_file)
        repo = git.Repo(REPODIR)

        last_push_path = Path(repo.working_dir) / ".ki" / "last_push"
        os.remove(last_push_path)

        # We should get a missing file error.
        os.chdir(REPODIR)
        with pytest.raises(MissingFileError):
            push(runner)


def test_push_honors_ignore_patterns():
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, col_file)
        os.chdir(REPODIR)

        shutil.copyfile(NOTE_2_PATH, os.path.join("Default", NOTE_2))
        with open(".gitignore", "a") as ignore_f:
            ignore_f.write("\nelephant")

        repo = git.Repo(".")
        repo.git.add(all=True)
        repo.index.commit(".")

        out = push(runner)
        assert "push: Ignoring" in out
        assert "matching pattern '.gitignore'" in out

        # Add and commit a new file that is not a note.
        Path("dummy_file").touch()

        repo = git.Repo(".")
        repo.git.add(all=True)
        repo.index.commit(".")

        # Since the output is currently very verbose, we should print a warning
        # for every such file. In the future, these warnings should only be
        # displayed if a verbosity flag is set.
        # TODO: Implement this verbosity flag.
        out = push(runner)
        assert "push: Not Anki note" in out


def test_push_displays_errors_from_head_ref_maybes(mocker: MockerFixture):
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone, edit, and commit.
        clone(runner, col_file)
        os.chdir(REPODIR)
        shutil.copyfile(NOTE_2_PATH, os.path.join("Default", NOTE_2))
        repo = git.Repo(".")
        repo.git.add(all=True)
        repo.index.commit(".")

        mocker.patch(
            "ki.M.head_kirepo_ref",
            return_value=Err(GitHeadRefNotFoundError(repo, Exception("<exc>"))),
        )
        with pytest.raises(GitHeadRefNotFoundError):
            push(runner)


def test_push_displays_errors_from_head_repo_ref(mocker: MockerFixture):
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone, edit, and commit.
        clone(runner, col_file)
        os.chdir(REPODIR)
        shutil.copyfile(NOTE_2_PATH, os.path.join("Default", NOTE_2))
        repo = git.Repo(".")
        repo.git.add(all=True)
        repo.index.commit(".")

        mocker.patch(
            "ki.M.head_repo_ref",
            side_effect=[
                Err(GitHeadRefNotFoundError(repo, Exception("<exc>"))),
            ],
        )
        with pytest.raises(GitHeadRefNotFoundError):
            push(runner)


def test_push_displays_errors_from_head_repo_ref_in_push_deltas(mocker: MockerFixture):
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone, edit, and commit.
        clone(runner, col_file)
        os.chdir(REPODIR)

        repo = git.Repo(".")
        head_1_sha = repo.head.commit.hexsha

        shutil.copyfile(NOTE_2_PATH, os.path.join("Default", NOTE_2))
        repo = git.Repo(".")
        repo.git.add(all=True)
        repo.index.commit(".")

        mocker.patch(
            "ki.M.head_repo_ref",
            side_effect=[
                Ok(RepoRef(repo, head_1_sha)),
                Err(GitHeadRefNotFoundError(repo, Exception("<exc>"))),
            ],
        )
        with pytest.raises(GitHeadRefNotFoundError):
            out = push(runner)
            logger.debug(out)


def test_push_displays_errors_from_notetype_parsing_in_push_deltas(
    mocker: MockerFixture,
):
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone, edit, and commit.
        clone(runner, col_file)
        os.chdir(REPODIR)

        repo = git.Repo(".")
        head_1_sha = repo.head.commit.hexsha

        shutil.copyfile(NOTE_2_PATH, os.path.join("Default", NOTE_2))
        repo = git.Repo(".")
        repo.git.add(all=True)
        repo.index.commit(".")

        col = open_collection(col_file)
        note = col.get_note(set(col.find_notes("")).pop())
        notetype: Notetype = ki.parse_notetype_dict(note.note_type()).unwrap()
        col.close()

        mocker.patch(
            "ki.parse_notetype_dict",
            side_effect=[
                Ok(notetype),
                Ok(notetype),
                Ok(notetype),
                Ok(notetype),
                Ok(notetype),
                Ok(notetype),
                Ok(notetype),
                Ok(notetype),
                Ok(notetype),
                Ok(notetype),
                Ok(notetype),
                Ok(notetype),
                Ok(notetype),
                Ok(notetype),
                Ok(notetype),
                Ok(notetype),
                Err(MissingFieldOrdinalError(3, "<notetype>")),
            ],
        )
        with pytest.raises(MissingFieldOrdinalError):
            out = push(runner)
            logger.debug(out)


def test_push_displays_errors_from_stage_kirepo_instantiation(mocker: MockerFixture):
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone, edit, and commit.
        clone(runner, col_file)
        os.chdir(REPODIR)
        shutil.copyfile(NOTE_2_PATH, os.path.join("Default", NOTE_2))
        repo = git.Repo(".")
        repo.git.add(all=True)
        repo.index.commit(".")

        mocker.patch(
            "ki.flatten_staging_repo",
            return_value=Err(NotKiRepoError()),
        )
        with pytest.raises(NotKiRepoError):
            push(runner)
