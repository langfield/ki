#!/usr/bin/env python3
"""Tests for ki command line interface (CLI)."""
import os
import random
import shutil
import sqlite3
import tempfile
import functools
import subprocess
from pathlib import Path
from dataclasses import dataclass
from distutils.dir_util import copy_tree
from importlib.metadata import version

import git
import click
import pytest
import bitstring
import checksumdir
import prettyprinter as pp
from lark import Lark
from lark.exceptions import UnexpectedToken
from loguru import logger
from result import Result, Err, Ok, OkErr
from click.testing import CliRunner
from anki.collection import Collection

from beartype import beartype
from beartype.typing import List, Callable

import ki
from ki import (
    BRANCH_NAME,
    LOCAL_SUFFIX,
    STAGE_SUFFIX,
    DELETED_SUFFIX,
    IGNORE,
    NotetypeDict,
    GitChangeType,
    Notetype,
    ColNote,
    Delta,
    ExtantDir,
    ExtantFile,
    MissingFileError,
    TargetExistsError,
    NotKiRepoError,
    UpdatesRejectedError,
    NotetypeMismatchError,
    UnhealthyNoteWarning,
    NoteFieldValidationWarning,
    KiRepo,
    KiRepoRef,
    RepoRef,
    M_kirepo,
    M_head_kirepo_ref,
    M_head_repo_ref,
    M_repo_ref,
    fftest,
    ffmkdir,
    ffcwd,
    ffforce_mkdir,
    ffchdir,
    md5,
    write_decks,
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
    slugify,
    display_fields_health_warning,
    is_anki_note,
    ftouch,
    get_batches,
    parse_markdown_note,
    flatten_staging_repo,
    filter_note_path,
    lock,
    unsubmodule_repo,
)
from ki.transformer import FlatNote, NoteTransformer


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
    return fftest(Path(col_file))


@beartype
def get_multideck_col_file() -> ExtantFile:
    """Put `multideck.anki2` in a tempdir and return its abspath."""
    # Copy collection to tempdir.
    tempdir = tempfile.mkdtemp()
    col_file = os.path.abspath(os.path.join(tempdir, MULTIDECK_COLLECTION_FILENAME))
    shutil.copyfile(MULTIDECK_COLLECTION_PATH, col_file)
    return fftest(Path(col_file))


@beartype
def get_html_col_file() -> ExtantFile:
    """Put `html.anki2` in a tempdir and return its abspath."""
    # Copy collection to tempdir.
    tempdir = tempfile.mkdtemp()
    col_file = os.path.abspath(os.path.join(tempdir, HTML_COLLECTION_FILENAME))
    shutil.copyfile(HTML_COLLECTION_PATH, col_file)
    return fftest(Path(col_file))


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
    cwd: ExtantDir = ffcwd()
    col = Collection(collection)
    ffchdir(cwd)

    notes: List[ColNote] = []
    for nid in set(col.find_notes("")):
        colnote: OkErr = get_colnote(col, nid)
        if colnote.is_err():
            raise colnote
        colnote: ColNote = colnote.unwrap()
        notes.append(colnote)

    return notes


@beartype
def get_staging_repo(repo: git.Repo) -> git.Repo:
    """Get deltas from ephemeral staging repo."""
    # Get ephemeral repo (submodules converted to ordinary directories).
    sha = str(repo.head.commit)
    staging_repo = get_ephemeral_repo(Path("ki/local"), repo, "AAA", sha)

    # Copy `.ki/` directory into the staging repo.
    staging_repo_kidir = Path(staging_repo.working_dir) / ".ki"
    shutil.copytree(Path(repo.working_dir) / ".ki", staging_repo_kidir)

    return staging_repo


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

        logger.debug(f"CWD: {ffcwd()}")

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
def test_clone_creates_directory():
    """Does it create the directory?"""
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, col_file)

        assert os.path.isdir(REPODIR)


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
            out = clone(runner, col_file)


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
            with pytest.raises(git.InvalidGitRepositoryError):
                os.environ["PATH"] = ""
                out = clone(runner, col_file)
            assert not os.path.isdir(HTML_REPODIR)
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
        cloned_md5 = md5(ExtantFile(cloned_note_path))
        true_md5 = md5(ExtantFile(true_note_path))

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
        cloned_md5 = md5(ExtantFile(cloned_note_path))
        true_md5 = md5(ExtantFile(true_note_path))

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
    original_md5 = md5(col_file)
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, col_file)

        updated_md5 = md5(col_file)
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
    old_hash = md5(col_file)
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
            if md5(fftest(Path(path))) == old_hash:
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
        parse_markdown_note(parser, transformer, fftest(Path(NOTE_5_PATH)))
    with pytest.raises(UnexpectedToken):
        parse_markdown_note(parser, transformer, fftest(Path(NOTE_6_PATH)))


def test_get_batches():
    """Does it get batches from a list of strings?"""
    runner = CliRunner()
    with runner.isolated_filesystem():
        root = ffcwd()
        one = ftouch(root, "note1.md")
        two = ftouch(root, "note2.md")
        three = ftouch(root, "note3.md")
        four = ftouch(root, "note4.md")
        batches = list(get_batches([one, two, three, four], n=2))
        assert batches == [[one, two], [three, four]]


def test_is_anki_note():
    """Do the checks in ``is_anki_note()`` actually do anything?"""
    runner = CliRunner()
    with runner.isolated_filesystem():
        root = ffcwd()
        mda = ftouch(root, "note.mda")
        amd = ftouch(root, "note.amd")
        mdtxt = ftouch(root, "note.mdtxt")
        nd = ftouch(root, "note.nd")

        assert is_anki_note(mda) is False
        assert is_anki_note(amd) is False
        assert is_anki_note(mdtxt) is False
        assert is_anki_note(nd) is False

        note_file: ExtantFile = ftouch(root, "note.md")

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
    cwd: ExtantDir = ffcwd()
    col = Collection(col_file)
    ffchdir(cwd)
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
    res: OkErr = update_note(note, flatnote, notetype, notetype)
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
    res: OkErr = update_note(note, flatnote, notetype, notetype)
    res.unwrap()
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
    res: OkErr = update_note(note, flatnote, notetype, notetype)

    assert "TITLE" in note.fields[0]
    assert "</p>" in note.fields[0]


def test_update_note_removes_field_contents():
    col = open_collection(get_col_file())
    note = col.get_note(set(col.find_notes("")).pop())

    field = "c"
    fields = {"Front": field, "Back": field}
    flatnote = FlatNote("title", 0, "Basic", "Default", [], False, fields)

    assert "a" in note.fields[0]
    notetype: Notetype = parse_notetype_dict(note.note_type())
    res: OkErr = update_note(note, flatnote, notetype, notetype)
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


def test_update_note_changes_notetype(capfd):
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
    result = slugify(text, allow_unicode=False)

    # Filter out Ethiopian syllable see.
    assert result == ""


def test_slugify_handles_unicode():
    """Test that slugify handles unicode alphanumerics."""
    # Hiragana should be okay.
    text = "ゅ"
    result = slugify(text, allow_unicode=True)
    assert result == text

    # Emojis as well.
    text = "😶"
    result = slugify(text, allow_unicode=True)
    assert result == text


def test_slugify_handles_html_tags():
    text = '<img src="card11front.jpg" />'
    result = slugify(text, allow_unicode=True)

    assert result == "img-srccard11frontjpg"


def test_get_note_path_produces_nonempty_filenames():
    field_text = '<img src="card11front.jpg" />'
    runner = CliRunner()
    with runner.isolated_filesystem():
        deck_dir: ExtantDir = ffforce_mkdir(Path("a"))

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
    res: OkErr = update_note(note, flatnote, notetype, notetype)
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
    cwd: ExtantDir = ffcwd()
    kirepo: KiRepo = M_kirepo(cwd).unwrap()
    con: sqlite3.Connection = lock(kirepo)
    md5sum: str = md5(kirepo.col_file)

    # Get reference to HEAD of current repo.
    head: KiRepoRef = M_head_kirepo_ref(kirepo).unwrap()

    # Copy current kirepo into a temp directory (the STAGE), hard reset to HEAD.
    stage_kirepo: KiRepo = get_ephemeral_kirepo(STAGE_SUFFIX, head, md5sum).unwrap()
    stage_kirepo = flatten_staging_repo(stage_kirepo, kirepo).unwrap()

    # This statement cannot be any farther down because we must get a reference
    # to HEAD *before* we commit, and then after the following line, the
    # reference we got will be HEAD~1, hence the variable name.
    head_1: RepoRef = M_head_repo_ref(stage_kirepo.repo).unwrap()

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
    a_repo: git.Repo = get_ephemeral_repo(DELETED_SUFFIX, head_1, md5sum).unwrap()
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


def test_unsubmodule_repo_removes_gitmodules():
    """
    When you have a ki repo with submodules, does calling
    `unsubmodule_repo()`on it remove them? We test this by checking if the
    `.gitmodules` file exists.
    """
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():
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


@pytest.mark.skip
def test_backup_is_no_op_when_backup_already_exists(capfd):
    col_file = get_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():
        clone(runner, col_file)
        os.chdir(REPODIR)

        backup(col_file)
        backup(col_file)
        captured = capfd.readouterr()
        assert "Backup already exists." in captured.out


@pytest.mark.skip
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


@pytest.mark.skip
def test_get_note_path():
    col_file = get_col_file()
    query = ""
    runner = CliRunner()
    with runner.isolated_filesystem(), Anki(path=col_file) as a:
        i = set(a.col.find_notes(query)).pop()
        _ = KiNote(a, a.col.get_note(i))

        deck_dir = ffcwd()
        dupe_path = deck_dir / "a.md"
        dupe_path.write_text("ay")
        note_path = get_note_path("a", deck_dir)
        assert str(note_path.name) == "a_1.md"


@pytest.mark.skip
def test_tidy_html_recursively():
    """Does tidy wrapper print a nice error when tidy is missing?"""
    runner = CliRunner()
    with runner.isolated_filesystem():
        root = ffcwd()
        file = root / "a.html"
        file.write_text("ay")
        old_path = os.environ["PATH"]
        try:
            os.environ["PATH"] = ""
            with pytest.raises(FileNotFoundError):
                tidy_html_recursively(root, False)
        finally:
            os.environ["PATH"] = old_path


@pytest.mark.skip
def test_create_deck_dir():
    deckname = "aa::bb::cc"
    runner = CliRunner()
    with runner.isolated_filesystem():
        root = ffcwd()
        path = create_deck_dir(deckname, root)
        assert path.is_dir()
        assert os.path.isdir("aa/bb/cc")


@pytest.mark.skip
def test_create_deck_dir_strips_leading_periods():
    deckname = ".aa::bb::.cc"
    runner = CliRunner()
    with runner.isolated_filesystem():
        root = ffcwd()
        path = create_deck_dir(deckname, root)
        assert path.is_dir()
        assert os.path.isdir("aa/bb/cc")


@pytest.mark.skip
def test_get_note_payload():
    col_file = get_col_file()
    query = ""
    runner = CliRunner()
    with runner.isolated_filesystem(), Anki(path=col_file) as a:
        i = set(a.col.find_notes(query)).pop()
        kinote = KiNote(a, a.col.get_note(i))
        fid = "1645010162168front"
        path = Path(fid)
        heyoo = "HEYOOOOO"
        path.write_text(heyoo, encoding="UTF-8")
        result = get_note_payload(kinote, {fid: path})
        assert heyoo in result
        assert "\nb\n" in result


@pytest.mark.skip
def test_write_notes_generates_deck_tree_correctly():
    """Does generated FS tree match example collection?"""
    true_note_path = os.path.abspath(os.path.join(MULTI_GITREPO_PATH, MULTI_NOTE_PATH))
    cloned_note_path = os.path.join(MULTIDECK_REPODIR, MULTI_NOTE_PATH)
    col_file = get_multideck_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        targetdir = fftest(Path(MULTIDECK_REPODIR))
        targetdir = ffmkdir(targetdir)
        write_notes(col_file, targetdir, silent=False)

        # Check that deck directory is created and all subdirectories.
        assert os.path.isdir(os.path.join(MULTIDECK_REPODIR, "Default"))
        assert os.path.isdir(os.path.join(MULTIDECK_REPODIR, "aa/bb/cc"))
        assert os.path.isdir(os.path.join(MULTIDECK_REPODIR, "aa/dd"))

        # Compute hashes.
        cloned_md5 = md5(ExtantFile(cloned_note_path))
        true_md5 = md5(ExtantFile(true_note_path))

        assert cloned_md5 == true_md5


@pytest.mark.skip
def test_write_notes_handles_html():
    """Does generated repo handle html okay?"""
    col_file = get_html_col_file()
    runner = CliRunner()
    with runner.isolated_filesystem():

        targetdir = ffmkdir(fftest(Path(HTML_REPODIR)))
        write_notes(col_file, targetdir, silent=False)
