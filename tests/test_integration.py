#!/usr/bin/env python3
"""Tests for ki command line interface (CLI)."""
import os
import shutil
import sqlite3
import tempfile
import subprocess
from pathlib import Path
from distutils.dir_util import copy_tree
from importlib.metadata import version

import git
import pytest
import prettyprinter as pp
from loguru import logger
from pytest_mock import MockerFixture
from click.testing import CliRunner
from anki.collection import Note

from beartype import beartype
from beartype.typing import List

import ki
import ki.maybes as M
import ki.functional as F
from ki import MEDIA
from ki.types import (
    KiRepo,
    RepoRef,
    Notetype,
    ColNote,
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
    AnkiAlreadyOpenError,
)
from ki.maybes import KI
from tests.test_ki import (
    open_collection,
    EDITED_COLLECTION_PATH,
    GITREPO_PATH,
    MULTI_GITREPO_PATH,
    REPODIR,
    MULTIDECK_REPODIR,
    HTML_REPODIR,
    MULTI_NOTE_PATH,
    MULTI_NOTE_ID,
    SUBMODULE_DIRNAME,
    NOTE_0,
    NOTE_1,
    NOTE_2,
    NOTE_3,
    NOTE_4,
    NOTE_2_PATH,
    NOTE_3_PATH,
    NOTE_0_ID,
    NOTE_4_ID,
    MEDIA_NOTE,
    MEDIA_NOTE_PATH,
    MEDIA_REPODIR,
    MEDIA_FILE_PATH,
    MEDIA_FILENAME,
    SPLIT_REPODIR,
    TEST_DATA_PATH,
    invoke,
    clone,
    pull,
    push,
    get_col_file,
    get_multideck_col_file,
    get_html_col_file,
    get_media_col_file,
    get_split_col_file,
    is_git_repo,
    randomly_swap_1_bit,
    checksum_git_repository,
    get_notes,
    get_repo_with_submodules,
)


PARSE_NOTETYPE_DICT_CALLS_PRIOR_TO_FLATNOTE_PUSH = 2

# pylint:disable=unnecessary-pass, too-many-lines


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

    assert result.stdout == f"ki, version {expected_version}{os.linesep}"
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


@beartype
@pytest.mark.skip
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
@pytest.mark.skip
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


@pytest.mark.skip
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


@pytest.mark.skip
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


@pytest.mark.skip
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


@pytest.mark.skip
def test_clone_fails_if_collection_is_already_open():
    """Does ki print a nice error message when Anki is accidentally left open?"""
    col_file = get_col_file()
    os.remove(col_file)
    runner = CliRunner()
    with runner.isolated_filesystem():
        col = open_collection(col_file)
        with pytest.raises(AnkiAlreadyOpenError):
            clone(runner, col_file)


@pytest.mark.skip
def test_clone_creates_directory():
    """Does it create the directory?"""
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, col_file)

        assert os.path.isdir(REPODIR)


@beartype
@pytest.mark.skip
def test_clone_displays_errors_from_creation_of_kirepo_metadata(mocker: MockerFixture):
    """Do errors get displayed nicely?"""
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        directory = ExtantDir(Path("directory"))
        collision = PathCreationCollisionError(directory, "token")
        mocker.patch("ki.F.fmkleaves", side_effect=collision)

        with pytest.raises(PathCreationCollisionError):
            clone(runner, col_file)


@beartype
@pytest.mark.skip
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
            side_effect=[NotKiRepoError()],
        )
        with pytest.raises(NotKiRepoError):
            clone(runner, col_file)


@pytest.mark.xfail
@pytest.mark.skip
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


@pytest.mark.skip
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
            clone(runner, col_file)


# TODO: This must be re-implemented.
@pytest.mark.xfail
@pytest.mark.skip
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
                clone(runner, col_file)
            assert not os.path.isdir(HTML_REPODIR)
        finally:
            os.environ["PATH"] = old_path


# TODO: Consider writing new `Exception` subclasses that print a slightly
# prettier message, informing the user of how to install the relevant missing
# dependency.
#
# TODO: This test now fails, which we expect, since the code that does
# auto-cleanup on clone fail has been removed. This must be added back in when
# we do error-handling.
@pytest.mark.xfail
@pytest.mark.skip
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
                clone(runner, col_file)
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
                clone(runner, col_file)
            error = raised.exconly()
        finally:
            os.environ["PATH"] = old_path


@pytest.mark.skip
def test_clone_succeeds_when_directory_exists_but_is_empty():
    """Does it clone into empty directories?"""
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Create directory where we want to clone.
        os.mkdir(REPODIR)
        clone(runner, col_file)


@pytest.mark.skip
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


@pytest.mark.skip
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


@pytest.mark.skip
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


@pytest.mark.skip
def test_cloned_collection_is_git_repository():
    """Does clone run `git init` and stuff?"""
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, col_file)

        assert is_git_repo(REPODIR)


@pytest.mark.skip
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


@pytest.mark.skip
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


@pytest.mark.skip
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


@pytest.mark.skip
def test_clone_writes_media_files():
    """Does clone copy media files from the media directory into 'MEDIA'?"""
    col_file = get_media_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():
        clone(runner, col_file)
        dot_media_path = Path("media") / MEDIA
        audio_path = dot_media_path / "1sec.mp3"
        assert dot_media_path.is_dir()
        assert audio_path.is_file()


@pytest.mark.skip
def test_clone_handles_cards_from_a_single_note_in_distinct_decks():
    col_file = get_split_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():
        clone(runner, col_file)
        assert os.path.islink(Path(SPLIT_REPODIR) / "top" / "b" / "a.md")
        assert os.path.isfile(Path(SPLIT_REPODIR) / "top" / "a" / "a.md")


# PULL


@pytest.mark.skip
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


@pytest.mark.skip
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


@pytest.mark.skip
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


@pytest.mark.skip
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


@pytest.mark.skip
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


@pytest.mark.skip
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


@pytest.mark.skip
def test_pull_displays_errors_from_repo_ref():
    """Does 'pull()' return early when the last push commit ref is bad?"""
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, col_file)

        kirepo: KiRepo = M.kirepo(F.test(Path(REPODIR)))
        kirepo.last_push_file.write_text("gibberish")

        # Edit collection.
        shutil.copyfile(EDITED_COLLECTION_PATH, col_file)

        os.chdir(REPODIR)
        with pytest.raises(GitRefNotFoundError):
            pull(runner)


@pytest.mark.skip
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
        mocker.patch("ki.F.fmkleaves", side_effect=collision)

        os.chdir(REPODIR)
        with pytest.raises(PathCreationCollisionError):
            pull(runner)


@pytest.mark.skip
def test_pull_handles_unexpectedly_changed_checksums(mocker: MockerFixture):
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, col_file)

        # Edit collection.
        shutil.copyfile(EDITED_COLLECTION_PATH, col_file)

        mocker.patch("ki.F.md5", side_effect=["good", "good", "good", "bad"])

        os.chdir(REPODIR)
        with pytest.raises(CollectionChecksumError):
            pull(runner)


@pytest.mark.skip
def test_pull_displays_errors_from_repo_initialization(mocker: MockerFixture):
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, col_file)

        # Edit collection.
        shutil.copyfile(EDITED_COLLECTION_PATH, col_file)

        repo = git.Repo.init(Path(REPODIR))
        effects = [git.InvalidGitRepositoryError()]
        mocker.patch("ki.M.repo", side_effect=effects)

        os.chdir(REPODIR)
        with pytest.raises(git.InvalidGitRepositoryError):
            pull(runner)


@pytest.mark.xfail
@pytest.mark.skip
def test_pull_preserves_reassigned_note_ids(tmp_path):
    """UNFINISHED!"""
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

        # Copy a new note into the submodule.
        shutil.copyfile(
            NOTE_2_PATH, Path(repo.working_dir) / SUBMODULE_DIRNAME / "Default" / NOTE_2
        )

        # Get a reference to the submodule repo.
        subrepo = git.Repo(Path(repo.working_dir) / SUBMODULE_DIRNAME)

        # Clone the submodule repo into another directory.
        sm_remote_repo = git.Repo.clone_from(subrepo.working_dir, "sm_remote")

        subrepo.git.add(all=True)
        subrepo.index.commit("Add `z` to `submodule/Default/a.md`")
        repo.git.add(all=True)
        repo.index.commit("Update submodule.")

        out = push(runner)
        logger.debug(out)

        colnotes = get_notes(col_file)
        notes: List[Note] = [colnote.n for colnote in colnotes]
        assert len(notes) == 3
        assert "<br />z<br />" in notes[0]["Back"]
        raise NotImplementedError


def test_pull_handles_uncommitted_submodule_commits(tmp_path):
    """UNFINISHED!"""
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

        # Copy a new note into the submodule.
        shutil.copyfile(
            NOTE_2_PATH, Path(repo.working_dir) / SUBMODULE_DIRNAME / "Default" / NOTE_2
        )

        # Get a reference to the submodule repo.
        subrepo = git.Repo(Path(repo.working_dir) / SUBMODULE_DIRNAME)

        # Clone the submodule repo into another directory.
        sm_remote_repo = git.Repo.clone_from(subrepo.working_dir, "sm_remote")

        subrepo.git.add(all=True)
        subrepo.index.commit("Add `z` to `submodule/Default/a.md`")
        repo.git.add(all=True)
        repo.index.commit("Update submodule.")

        # Edit a file within the submodule.
        file = Path(repo.working_dir) / SUBMODULE_DIRNAME / "Default" / "a.md"
        logger.debug(f"Adding 'y' to file '{file}'")
        with open(file, "a", encoding="UTF-8") as note_f:
            note_f.write("\ny\n")

        subrepo.git.add(all=True)
        subrepo.index.commit("Add `y` to `submodule/Default/a.md`")
        repo.git.add(all=True)
        repo.index.commit("Update submodule.")

        # Edit collection.
        shutil.copyfile(EDITED_COLLECTION_PATH, col_file)

        out = pull(runner)
        logger.debug(out)


# PUSH


@pytest.mark.skip
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


@pytest.mark.skip
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


@pytest.mark.skip
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


@pytest.mark.skip
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


@pytest.mark.skip
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


@pytest.mark.skip
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


@pytest.mark.skip
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


@pytest.mark.skip
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


@pytest.mark.skip
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


@pytest.mark.skip
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


@pytest.mark.skip
def test_push_displays_informative_error_when_last_push_file_is_missing():
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


@pytest.mark.xfail
@pytest.mark.skip
def test_push_honors_ignore_patterns():
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, col_file)
        os.chdir(REPODIR)

        shutil.copyfile(NOTE_2_PATH, os.path.join("Default", NOTE_2))
        with open(".gitignore", "a", encoding="UTF-8") as ignore_f:
            ignore_f.write("\nelephant")

        repo = git.Repo(".")
        repo.git.add(all=True)
        repo.index.commit(".")

        out = push(runner)
        assert "Warning: ignoring" in out
        assert "matching ignore pattern '.gitignore'" in out

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
        assert "Warning: not Anki note" in out


@pytest.mark.skip
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
            side_effect=GitHeadRefNotFoundError(repo, Exception("<exc>")),
        )
        with pytest.raises(GitHeadRefNotFoundError):
            push(runner)


@pytest.mark.skip
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
                GitHeadRefNotFoundError(repo, Exception("<exc>")),
            ],
        )
        with pytest.raises(GitHeadRefNotFoundError):
            push(runner)


@pytest.mark.skip
def test_push_displays_errors_from_notetype_parsing_in_push_deltas_during_model_adding(
    mocker: MockerFixture,
):
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone, edit, and commit.
        clone(runner, col_file)
        os.chdir(REPODIR)

        repo = git.Repo(".")

        shutil.copyfile(NOTE_2_PATH, os.path.join("Default", NOTE_2))
        repo = git.Repo(".")
        repo.git.add(all=True)
        repo.index.commit(".")

        col = open_collection(col_file)
        note = col.get_note(set(col.find_notes("")).pop())
        notetype: Notetype = ki.parse_notetype_dict(note.note_type())
        col.close()

        effects = [MissingFieldOrdinalError(3, "<notetype>")]

        mocker.patch("ki.parse_notetype_dict", side_effect=effects)

        with pytest.raises(MissingFieldOrdinalError):
            push(runner)


@pytest.mark.skip
def test_push_displays_errors_from_notetype_parsing_in_push_deltas_during_push_flatnote_to_anki(
    mocker: MockerFixture,
):
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone, edit, and commit.
        clone(runner, col_file)
        os.chdir(REPODIR)

        repo = git.Repo(".")

        shutil.copyfile(NOTE_2_PATH, os.path.join("Default", NOTE_2))
        repo = git.Repo(".")
        repo.git.add(all=True)
        repo.index.commit(".")

        col = open_collection(col_file)
        note = col.get_note(set(col.find_notes("")).pop())
        notetype: Notetype = ki.parse_notetype_dict(note.note_type())
        col.close()

        effects = [notetype] * PARSE_NOTETYPE_DICT_CALLS_PRIOR_TO_FLATNOTE_PUSH
        effects += [MissingFieldOrdinalError(3, "<notetype>")]

        mocker.patch("ki.parse_notetype_dict", side_effect=effects)

        with pytest.raises(MissingFieldOrdinalError):
            out = push(runner)
            logger.debug(out)


@pytest.mark.skip
def test_push_handles_submodules(tmp_path):
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

        # Copy a new note into the submodule.
        shutil.copyfile(
            NOTE_2_PATH, Path(repo.working_dir) / SUBMODULE_DIRNAME / "Default" / NOTE_2
        )

        subrepo = git.Repo(Path(repo.working_dir) / SUBMODULE_DIRNAME)
        subrepo.git.add(all=True)
        subrepo.index.commit(".")
        repo.git.add(all=True)
        repo.index.commit(".")

        out = push(runner)
        logger.debug(out)

        colnotes = get_notes(col_file)
        notes: List[Note] = [colnote.n for colnote in colnotes]
        assert len(notes) == 3
        assert "<br />z<br />" in notes[0]["Back"]


@pytest.mark.skip
def test_push_writes_media(tmp_path):
    col_file = get_media_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        clone(runner, col_file)
        root = F.cwd()
        media_note_path = root / MEDIA_REPODIR / "Default" / MEDIA_NOTE
        media_file_path = root / MEDIA_REPODIR / "Default" / MEDIA / MEDIA_FILENAME
        shutil.copyfile(MEDIA_NOTE_PATH, media_note_path)
        shutil.copyfile(MEDIA_FILE_PATH, media_file_path)
        os.chdir(MEDIA_REPODIR)
        repo = git.Repo(F.cwd())
        repo.git.add(all=True)
        repo.index.commit("Add air.md")
        out = push(runner)
        logger.debug(out)
        os.chdir("../")
        shutil.rmtree(MEDIA_REPODIR)
        out = clone(runner, col_file)
        logger.debug(out)

        col = open_collection(col_file)
        check = col.media.check()
        logger.debug(os.listdir(Path(MEDIA_REPODIR) / "Default"))
        assert os.path.isfile(Path(MEDIA_REPODIR) / "Default" / MEDIA_NOTE)
        assert col.media.have(MEDIA_FILENAME)
        assert len(check.missing) == 0
        assert len(check.unused) == 0


@pytest.mark.skip
def test_push_handles_foreign_models(tmp_path):
    """Just check that we don't return an exception from `push()`."""
    col_file = get_col_file()
    runner = CliRunner()
    japan_path = (Path(TEST_DATA_PATH) / "repos" / "japanese-core-2000").resolve()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        clone(runner, col_file)
        shutil.copytree(japan_path, Path(REPODIR) / "Default" / "japan")
        logger.debug(F.cwd())
        os.chdir(REPODIR)
        repo = git.Repo(F.cwd())
        repo.git.add(all=True)
        repo.index.commit("japan")
        out = push(runner)
        logger.debug(out)


@pytest.mark.skip
def test_push_fails_if_database_is_locked():
    """Does ki print a nice error message when Anki is accidentally left open?"""
    col_file = get_col_file()
    runner = CliRunner()
    japan_path = (Path(TEST_DATA_PATH) / "repos" / "japanese-core-2000").resolve()
    with runner.isolated_filesystem():
        clone(runner, col_file)
        shutil.copytree(japan_path, Path(REPODIR) / "Default" / "japan")
        os.chdir(REPODIR)
        repo = git.Repo(F.cwd())
        repo.git.add(all=True)
        repo.index.commit("japan")
        con = sqlite3.connect(col_file)
        con.isolation_level = "EXCLUSIVE"
        con.execute("BEGIN EXCLUSIVE")
        with pytest.raises(SQLiteLockError):
            out = push(runner)


@pytest.mark.skip
def test_push_is_nontrivial_when_pulled_changes_are_reverted(tmp_path):
    """
    If you push, make changes in Anki, then pull those changes, then undo them
    within the ki repo, then push again, the push should *not* be a no-op. The
    changes are currently applied in Anki, and the push should undo them.
    """
    col_file = get_col_file()
    col_copy_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):

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
        repo.index.commit("Deleted.")

        # Push changes.
        os.chdir(REPODIR)
        out = push(runner)
        logger.debug(f"\nPUSH:\n{out}")
        notes = get_notes(col_file)
        notes = [colnote.n["Front"] for colnote in notes]
        assert notes == ["c"]

        # Revert the collection.
        os.remove(col_file)
        shutil.copyfile(col_copy_file, col_file)

        # Pull again.
        out = pull(runner)
        logger.debug(f"\nPULL:\n{out}")

        # Remove again.
        assert os.path.isfile(NOTE_0)
        os.remove(NOTE_0)
        repo = git.Repo(F.cwd())
        repo.git.add(all=True)
        repo.index.commit("Deleted.")

        # Push changes.
        out = push(runner)
        logger.debug(f"\nPUSH2:\n{out}")
        notes = get_notes(col_file)
        notes = [colnote.n["Front"] for colnote in notes]
        logger.debug(pp.pformat(notes))
        assert "a" not in notes
        assert notes == ["c"]
        assert "ki push: up to date." not in out


@pytest.mark.skip
def test_push_doesnt_unnecessarily_deduplicate_notetypes():
    """
    Does push refrain from adding a new notetype if the requested notetype
    already exists in the collection?
    """
    col_file = get_col_file()
    col_copy_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, col_file)

        col = open_collection(col_file)
        orig_models = col.models.all_names_and_ids()
        col.close(save=False)

        # Remove a note file.
        os.chdir(REPODIR)
        logger.debug(os.listdir("Default"))
        assert os.path.isfile(NOTE_0)
        os.remove(NOTE_0)
        logger.debug(os.listdir("Default"))

        # Commit the deletion.
        os.chdir("../")
        repo = git.Repo(REPODIR)
        repo.git.add(all=True)
        repo.index.commit("Deleted.")

        # Push changes.
        os.chdir(REPODIR)
        out = push(runner)
        logger.debug(f"\nPUSH:\n{out}")
        logger.debug(os.listdir("Default"))

        # Revert the collection.
        os.remove(col_file)
        shutil.copyfile(col_copy_file, col_file)
        logger.debug(os.listdir("Default"))

        # Pull again.
        out = pull(runner)
        logger.debug(f"\nPULL:\n{out}")
        logger.debug(os.listdir("Default"))

        # Remove again.
        assert os.path.isfile(NOTE_0)
        os.remove(NOTE_0)
        repo = git.Repo(F.cwd())
        repo.git.add(all=True)
        repo.index.commit("Deleted.")

        # Push changes.
        out = push(runner)
        logger.debug(f"\nPUSH:\n{out}")
        logger.debug(os.listdir("Default"))

        col = open_collection(col_file)
        models = col.models.all_names_and_ids()
        assert len(orig_models) == len(models)
        col.close(save=False)


@pytest.mark.skip
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
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, col_file)

        # Remove a note file.
        os.chdir(REPODIR)
        logger.debug(os.listdir("Default"))
        assert os.path.isfile(NOTE_0)
        temp_note_0_file = F.mkdtemp() / "NOTE_0"
        shutil.move(NOTE_0, temp_note_0_file)
        assert not os.path.isfile(NOTE_0)
        logger.debug(os.listdir("Default"))

        # Commit the deletion.
        os.chdir("../")
        repo = git.Repo(REPODIR)
        repo.git.add(all=True)
        repo.index.commit("Deleted.")

        # Push changes.
        os.chdir(REPODIR)
        out = push(runner)
        logger.debug(f"\nPUSH:\n{out}")
        logger.debug(os.listdir("Default"))

        # Put file back.
        shutil.move(temp_note_0_file, NOTE_0)
        repo.git.add(all=True)
        repo.index.commit("Added.")

        # Push again.
        out = push(runner)
        logger.debug(f"\nPUSH:\n{out}")
        logger.debug(os.listdir("Default"))
        assert "ki push: up to date." not in out


@pytest.mark.skip
def test_push_changes_deck_for_moved_notes():
    col_file = get_multideck_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, col_file)

        # Move a note.
        os.chdir(MULTIDECK_REPODIR)
        target = "aa/dd/cc.md"
        assert os.path.isfile(MULTI_NOTE_PATH)
        shutil.move(MULTI_NOTE_PATH, target)
        assert not os.path.isfile(MULTI_NOTE_PATH)

        # Commit the move.
        os.chdir("../")
        repo = git.Repo(MULTIDECK_REPODIR)
        repo.git.add(all=True)
        repo.index.commit("Move.")

        # Push changes.
        os.chdir(MULTIDECK_REPODIR)
        out = push(runner)
        logger.debug(f"\nPUSH:\n{out}")

        # Check that deck has changed.
        notes: List[ColNote] = get_notes(col_file)
        notes = filter(lambda colnote: colnote.n.id == MULTI_NOTE_ID, notes)
        notes = list(notes)
        assert len(notes) == 1
        colnote = notes.pop()
        assert colnote.deck == "aa::dd"
