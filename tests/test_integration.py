#!/usr/bin/env python3
"""Tests for ki command line interface (CLI)."""
import os
import gc
import sys
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
from anki.collection import Note, Collection

from beartype import beartype
from beartype.typing import List

import ki
import ki.maybes as M
import ki.functional as F
from ki import MEDIA
from ki.types import (
    KiRepo,
    Notetype,
    ColNote,
    ExtantDir,
    ExtantFile,
    MissingFileError,
    TargetExistsError,
    NotKiRepoError,
    UpdatesRejectedError,
    SQLiteLockError,
    PathCreationCollisionError,
    GitRefNotFoundError,
    GitHeadRefNotFoundError,
    CollectionChecksumError,
    MissingFieldOrdinalError,
    AnkiAlreadyOpenError,
)
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
    JAPANESE_GITREPO_PATH,
    BRANCH_NAME,
    get_test_collection,
    SampleCollection,
)


PARSE_NOTETYPE_DICT_CALLS_PRIOR_TO_FLATNOTE_PUSH = 2

# pylint: disable=unnecessary-pass, too-many-lines, invalid-name
# pylint: disable=missing-function-docstring, too-many-locals


EDITED: SampleCollection = get_test_collection("edited")


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
    ORIGINAL: SampleCollection = get_test_collection("original")
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        # Clone collection in cwd.
        clone(runner, ORIGINAL.col_file)

        # Check that hash is written.
        with open(
            os.path.join(ORIGINAL.repodir, ".ki/hashes"), encoding="UTF-8"
        ) as hashes_file:
            hashes = hashes_file.read()
            assert f"a68250f8ee3dc8302534f908bcbafc6a  {ORIGINAL.filename}" in hashes
            assert (
                f"199216c39eeabe23a1da016a99ffd3e2  {ORIGINAL.filename}" not in hashes
            )

        # Edit collection.
        shutil.copyfile(EDITED.path, ORIGINAL.col_file)

        # Pull edited collection.
        os.chdir(ORIGINAL.repodir)
        pull(runner)
        os.chdir("../")

        # Check that edited hash is written and old hash is still there.
        with open(
            os.path.join(ORIGINAL.repodir, ".ki/hashes"), encoding="UTF-8"
        ) as hashes_file:
            hashes = hashes_file.read()
            assert f"a68250f8ee3dc8302534f908bcbafc6a  {ORIGINAL.filename}" in hashes
            assert f"199216c39eeabe23a1da016a99ffd3e2  {ORIGINAL.filename}" in hashes


@pytest.mark.skip
def test_no_op_pull_push_cycle_is_idempotent():
    """Do pull/push not misbehave if you keep doing both?"""
    ORIGINAL: SampleCollection = get_test_collection("original")
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, ORIGINAL.col_file)
        assert os.path.isdir(ORIGINAL.repodir)

        os.chdir(ORIGINAL.repodir)
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
def test_output(tmp_path: Path):
    """Does it print nice things?"""
    ORIGINAL: SampleCollection = get_test_collection("original")
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        out = clone(runner, ORIGINAL.col_file)

        # Edit collection.
        shutil.copyfile(EDITED.path, ORIGINAL.col_file)

        # Pull edited collection.
        os.chdir(ORIGINAL.repodir)
        out = pull(runner)

        # Modify local repository.
        assert os.path.isfile(NOTE_7)
        with open(NOTE_7, "a", encoding="UTF-8") as note_file:
            note_file.write("e\n")
        shutil.copyfile(NOTE_2_PATH, NOTE_2)
        shutil.copyfile(NOTE_3_PATH, NOTE_3)

        # Commit.
        os.chdir("../")
        repo = git.Repo(ORIGINAL.repodir)
        repo.git.add(all=True)
        repo.index.commit("Added 'e'.")

        # Push changes.
        os.chdir(ORIGINAL.repodir)
        out = push(runner)
        assert "Overwrote" in out


# CLONE


@pytest.mark.skip
def test_clone_fails_if_collection_doesnt_exist():
    """Does ki clone only if `.anki2` file exists?"""
    ORIGINAL: SampleCollection = get_test_collection("original")
    os.remove(ORIGINAL.col_file)
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        with pytest.raises(FileNotFoundError):
            clone(runner, ORIGINAL.col_file)
        assert not os.path.isdir(ORIGINAL.repodir)


@pytest.mark.skip
def test_clone_fails_if_collection_is_already_open():
    """Does ki print a nice error message when Anki is accidentally left open?"""
    ORIGINAL: SampleCollection = get_test_collection("original")
    os.remove(ORIGINAL.col_file)
    runner = CliRunner()
    with runner.isolated_filesystem():
        _ = open_collection(ORIGINAL.col_file)
        with pytest.raises(AnkiAlreadyOpenError):
            clone(runner, ORIGINAL.col_file)


@pytest.mark.skip
def test_clone_creates_directory():
    """Does it create the directory?"""
    ORIGINAL: SampleCollection = get_test_collection("original")
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, ORIGINAL.col_file)

        assert os.path.isdir(ORIGINAL.repodir)


@beartype
@pytest.mark.skip
def test_clone_displays_errors_from_creation_of_kirepo_metadata(mocker: MockerFixture):
    """Do errors get displayed nicely?"""
    ORIGINAL: SampleCollection = get_test_collection("original")
    runner = CliRunner()
    with runner.isolated_filesystem():

        directory = ExtantDir(Path("directory"))
        collision = PathCreationCollisionError(directory, "token")
        mocker.patch("ki.F.fmkleaves", side_effect=collision)

        with pytest.raises(PathCreationCollisionError):
            clone(runner, ORIGINAL.col_file)


@pytest.mark.skip
def test_clone_handles_html():
    """Does it tidy html and stuff?"""
    HTML: SampleCollection = get_test_collection("html")
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, HTML.col_file)
        assert os.path.isdir(HTML.repodir)

        path = Path(".") / HTML.repodir / "Default" / "あだ名.md"
        contents = path.read_text(encoding="UTF-8")
        snippet = """<table class="kanji-match">\n    """
        snippet += """<tbody>\n      """
        snippet += """<tr class="match-row-kanji" lang="ja">\n"""
        assert snippet in contents


@pytest.mark.skip
def test_clone_tidying_only_breaks_lines_for_fields_containing_html():
    """Does it tidy html and stuff?"""
    HTML: SampleCollection = get_test_collection("html")
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, HTML.col_file)
        assert os.path.isdir(HTML.repodir)

        path = Path(".") / HTML.repodir / "Default" / "on-evil.md"
        contents = path.read_text(encoding="UTF-8")

        # This line should not be broken.
        assert (
            "and I doubt that punishment should be relevant to criminal justice."
            in contents
        )


@pytest.mark.skip
def test_clone_errors_when_directory_is_populated():
    """Does it disallow overwrites?"""
    ORIGINAL: SampleCollection = get_test_collection("original")
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Create directory where we want to clone.
        os.mkdir(ORIGINAL.repodir)
        with open(
            os.path.join(ORIGINAL.repodir, "hi"), "w", encoding="UTF-8"
        ) as hi_file:
            hi_file.write("hi\n")

        # Should error out because directory already exists.
        with pytest.raises(TargetExistsError):
            clone(runner, ORIGINAL.col_file)


@pytest.mark.skip
def test_clone_cleans_up_on_error():
    """Does it clean up on nontrivial errors?"""
    HTML: SampleCollection = get_test_collection("html")
    runner = CliRunner()
    with runner.isolated_filesystem():

        clone(runner, HTML.col_file)
        assert os.path.isdir(HTML.repodir)
        F.rmtree(F.test(Path(HTML.repodir)))
        old_path = os.environ["PATH"]
        try:
            with pytest.raises(FileNotFoundError) as err:
                os.environ["PATH"] = ""
                clone(runner, HTML.col_file)
            assert not os.path.isdir(HTML.repodir)
        finally:
            os.environ["PATH"] = old_path


@pytest.mark.skip
def test_clone_cleans_up_preserves_directories_that_exist_a_priori():
    """Does clone not delete targetdirs that already existed?"""
    HTML: SampleCollection = get_test_collection("html")
    runner = CliRunner()
    with runner.isolated_filesystem():

        os.mkdir(HTML.repodir)
        assert os.path.isdir(HTML.repodir)
        old_path = os.environ["PATH"]
        try:
            with pytest.raises(FileNotFoundError) as err:
                os.environ["PATH"] = ""
                clone(runner, HTML.col_file)
            assert os.path.isdir(HTML.repodir)
            assert len(os.listdir(HTML.repodir)) == 0
        finally:
            os.environ["PATH"] = old_path


@pytest.mark.skip
def test_clone_displays_nice_errors_for_missing_dependencies():
    """Does it tell the user what to install?"""
    HTML: SampleCollection = get_test_collection("html")
    runner = CliRunner()
    with runner.isolated_filesystem():

        clone(runner, HTML.col_file)
        assert os.path.isdir(HTML.repodir)
        F.rmtree(F.test(Path(HTML.repodir)))
        old_path = os.environ["PATH"]

        # In case where nothing is installed, we expect to fail on `tidy`
        # first.
        try:
            with pytest.raises(FileNotFoundError) as raised:
                os.environ["PATH"] = ""
                clone(runner, HTML.col_file)
            error = raised.exconly()
            assert "tidy" in str(error)
        finally:
            os.environ["PATH"] = old_path

        # If `tidy` is on the PATH, but nothing else, then we expect a
        # `GitCommandNotFound` error.
        try:
            with pytest.raises(git.GitCommandNotFound) as raised:
                if sys.platform == "win32":
                    gits = [
                        r"C:\Program Files\Git\bin;",
                        r"C:\Program Files\Git\cmd;",
                        r"C:\Program Files\Git\mingw64\bin;",
                        r"C:\Program Files\Git\usr\bin;",
                    ]
                    path = os.environ["PATH"]
                    for gitpath in gits:
                        path = path.replace(gitpath, "")
                    os.environ["PATH"] = path
                else:
                    tmp = F.mkdtemp()
                    tgt = tmp / "tidy"
                    shutil.copyfile(shutil.which("tidy"), tgt)
                    st = os.stat(tgt)
                    os.chmod(tgt, st.st_mode | 0o111)
                    path = str(tgt.parent)
                    os.environ["PATH"] = path

                clone(runner, HTML.col_file)
            error = raised.exconly()
        finally:
            os.environ["PATH"] = old_path


@pytest.mark.skip
def test_clone_succeeds_when_directory_exists_but_is_empty():
    """Does it clone into empty directories?"""
    ORIGINAL: SampleCollection = get_test_collection("original")
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Create directory where we want to clone.
        os.mkdir(ORIGINAL.repodir)
        clone(runner, ORIGINAL.col_file)


@pytest.mark.skip
def test_clone_generates_expected_notes():
    """Do generated note files match content of an example collection?"""
    ORIGINAL: SampleCollection = get_test_collection("original")
    true_note_path = os.path.join(GITREPO_PATH, NOTE_0)
    cloned_note_path = os.path.join(ORIGINAL.repodir, NOTE_0)
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, ORIGINAL.col_file)

        # Check that deck directory is created.
        assert os.path.isdir(os.path.join(ORIGINAL.repodir, "Default"))

        # Compute hashes.
        cloned_md5 = F.md5(ExtantFile(cloned_note_path))
        true_md5 = F.md5(ExtantFile(true_note_path))

        assert cloned_md5 == true_md5


@pytest.mark.skip
def test_clone_generates_deck_tree_correctly():
    """Does generated FS tree match example collection?"""
    MULTIDECK: SampleCollection = get_test_collection("multideck")
    true_note_path = os.path.abspath(os.path.join(MULTI_GITREPO_PATH, MULTI_NOTE_PATH))
    cloned_note_path = os.path.join(MULTIDECK.repodir, MULTI_NOTE_PATH)
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        out = clone(runner, MULTIDECK.col_file)

        # Check that deck directory is created and all subdirectories.
        assert os.path.isdir(os.path.join(MULTIDECK.repodir, "Default"))
        assert os.path.isdir(os.path.join(MULTIDECK.repodir, "aa/bb/cc"))
        assert os.path.isdir(os.path.join(MULTIDECK.repodir, "aa/dd"))

        # Compute hashes.
        cloned_md5 = F.md5(ExtantFile(cloned_note_path))
        true_md5 = F.md5(ExtantFile(true_note_path))

        assert cloned_md5 == true_md5


@pytest.mark.skip
def test_clone_generates_ki_subdirectory():
    """Does clone command generate .ki/ directory?"""
    ORIGINAL: SampleCollection = get_test_collection("original")
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, ORIGINAL.col_file)

        # Check kidir exists.
        kidir = os.path.join(ORIGINAL.repodir, ".ki/")
        assert os.path.isdir(kidir)


@pytest.mark.skip
def test_cloned_collection_is_git_repository():
    """Does clone run `git init` and stuff?"""
    ORIGINAL: SampleCollection = get_test_collection("original")
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, ORIGINAL.col_file)

        assert is_git_repo(ORIGINAL.repodir)


@pytest.mark.skip
def test_clone_commits_directory_contents():
    """Does clone leave user with an up-to-date repo?"""
    ORIGINAL: SampleCollection = get_test_collection("original")
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, ORIGINAL.col_file)

        # Construct repo object.
        repo = git.Repo(ORIGINAL.repodir)

        # Make sure there are no changes.
        changes = repo.head.commit.diff()
        assert len(changes) == 0

        # Make sure there is exactly 1 commit.
        commits = list(repo.iter_commits("HEAD"))
        assert len(commits) == 1


@pytest.mark.skip
def test_clone_leaves_collection_file_unchanged():
    """Does clone leave the collection alone?"""
    ORIGINAL: SampleCollection = get_test_collection("original")
    original_md5 = F.md5(ORIGINAL.col_file)
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, ORIGINAL.col_file)

        updated_md5 = F.md5(ORIGINAL.col_file)
        assert original_md5 == updated_md5


@pytest.mark.skip
def test_clone_directory_argument_works():
    """Does clone obey the target directory argument?"""
    ORIGINAL: SampleCollection = get_test_collection("original")
    runner = CliRunner()
    with runner.isolated_filesystem():

        tempdir = tempfile.mkdtemp()
        target = os.path.join(tempdir, "TARGET")
        assert not os.path.isdir(target)
        assert not os.path.isfile(target)

        # Clone collection in cwd.
        clone(runner, ORIGINAL.col_file, target)
        assert os.path.isdir(target)


@pytest.mark.skip
def test_clone_writes_media_files():
    """Does clone copy media files from the media directory into 'MEDIA'?"""
    MEDIACOL: SampleCollection = get_test_collection("media")
    runner = CliRunner()
    with runner.isolated_filesystem():
        clone(runner, MEDIACOL.col_file)
        dot_media_path = Path("media") / MEDIA
        audio_path = dot_media_path / "1sec.mp3"
        assert dot_media_path.is_dir()
        assert audio_path.is_file()


@pytest.mark.skip
def test_clone_handles_cards_from_a_single_note_in_distinct_decks(tmp_path: Path):
    SPLIT: SampleCollection = get_test_collection("split")
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        clone(runner, SPLIT.col_file)
        two = Path(SPLIT.repodir) / "top" / "b" / "a_Card 2.md"
        orig = Path(SPLIT.repodir) / "top" / "a" / "a.md"

        if sys.platform == "win32":
            assert two.read_text(encoding="UTF-8") == r"..\..\top\a\a.md"
        else:
            assert os.path.islink(two)
        assert os.path.isfile(orig)


@pytest.mark.skip
def test_clone_writes_plaintext_posix_symlinks_on_windows(
    tmp_path, mocker: MockerFixture
):
    SYMLINKS: SampleCollection = get_test_collection("symlinks")
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        mocker.patch("sys.platform", "win32")
        out = clone(runner, SYMLINKS.col_file)

        # Verify that there are no symlinks in the cloned sample repo.
        for root, _, files in os.walk(SYMLINKS.repodir):
            for file in files:
                path = os.path.join(root, file)
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
        winlinks = set([str(link) for link in winlinks])

        # Check that each latent symlink has the correct file mode.
        repo = git.Repo(SYMLINKS.repodir)
        for entry in repo.commit().tree.traverse():
            path = entry.path
            if isinstance(entry, git.Blob) and path in winlinks:
                mode = oct(entry.mode)
                assert mode == "0o120000"


@pytest.mark.skip
def test_clone_url_decodes_media_src_attributes(tmp_path: Path):
    DOUBLE: SampleCollection = get_test_collection("no_double_encodings")
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        clone(runner, DOUBLE.col_file)

        os.chdir(DOUBLE.repodir)
        path = Path("DeepLearning for CV") / "list-some-pros-and-cons-of-dl.md"
        with open(path, "r", encoding="UTF-8") as f:
            contents: str = f.read()
        assert '<img src="Screenshot 2019-05-01 at 14.40.56.png">' in contents


# PULL


@pytest.mark.skip
def test_pull_fails_if_collection_no_longer_exists():
    """Does ki pull only if `.anki2` file exists?"""
    ORIGINAL: SampleCollection = get_test_collection("original")
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, ORIGINAL.col_file)

        # Delete collection and try to pull.
        os.remove(ORIGINAL.col_file)
        with pytest.raises(FileNotFoundError):
            os.chdir(ORIGINAL.repodir)
            pull(runner)


@pytest.mark.skip
def test_pull_fails_if_collection_file_is_corrupted():
    """Does `pull()` fail gracefully when the collection file is bad?"""
    ORIGINAL: SampleCollection = get_test_collection("original")
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, ORIGINAL.col_file)

        # Overwrite collection and try to pull.
        ORIGINAL.col_file.write_text("bad_contents")

        os.chdir(ORIGINAL.repodir)
        with pytest.raises(SQLiteLockError):
            pull(runner)


@pytest.mark.skip
def test_pull_writes_changes_correctly():
    """Does ki get the changes from modified collection file?"""
    ORIGINAL: SampleCollection = get_test_collection("original")
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, ORIGINAL.col_file)
        assert not os.path.isfile(os.path.join(ORIGINAL.repodir, NOTE_1))

        # Edit collection.
        shutil.copyfile(EDITED.path, ORIGINAL.col_file)

        # Pull edited collection.
        os.chdir(ORIGINAL.repodir)
        pull(runner)
        assert os.path.isfile(NOTE_1)


@pytest.mark.skip
def test_pull_unchanged_collection_is_no_op():
    """Does ki remove remote before quitting?"""
    ORIGINAL: SampleCollection = get_test_collection("original")
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, ORIGINAL.col_file)
        orig_hash = checksum_git_repository(ORIGINAL.repodir)

        # Pull updated collection.
        os.chdir(ORIGINAL.repodir)
        pull(runner)
        os.chdir("../")
        new_hash = checksum_git_repository(ORIGINAL.repodir)

        assert orig_hash == new_hash


@pytest.mark.skip
def test_pull_avoids_unnecessary_merge_conflicts():
    """Does ki prevent gratuitous merge conflicts?"""
    ORIGINAL: SampleCollection = get_test_collection("original")
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, ORIGINAL.col_file)
        assert not os.path.isfile(os.path.join(ORIGINAL.repodir, NOTE_1))

        # Edit collection.
        shutil.copyfile(EDITED.path, ORIGINAL.col_file)

        # Pull edited collection.
        os.chdir(ORIGINAL.repodir)
        out = pull(runner)
        assert "Automatic merge failed; fix" not in out


@pytest.mark.skip
def test_pull_still_works_from_subdirectories():
    """Does pull still work if you're farther down in the directory tree than the repo route?"""
    ORIGINAL: SampleCollection = get_test_collection("original")
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, ORIGINAL.col_file)
        assert not os.path.isfile(os.path.join(ORIGINAL.repodir, NOTE_1))

        # Edit collection.
        shutil.copyfile(EDITED.path, ORIGINAL.col_file)

        # Pull edited collection.
        os.chdir(os.path.join(ORIGINAL.repodir, "Default"))
        pull(runner)


@pytest.mark.skip
def test_pull_displays_errors_from_repo_ref():
    """Does 'pull()' return early when the last push commit ref is bad?"""
    ORIGINAL: SampleCollection = get_test_collection("original")
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, ORIGINAL.col_file)

        kirepo: KiRepo = M.kirepo(F.test(Path(ORIGINAL.repodir)))
        kirepo.last_push_file.write_text("gibberish")

        # Edit collection.
        shutil.copyfile(EDITED.path, ORIGINAL.col_file)

        os.chdir(ORIGINAL.repodir)
        with pytest.raises(GitRefNotFoundError):
            pull(runner)


@pytest.mark.skip
def test_pull_displays_errors_from_clone_helper(mocker: MockerFixture):
    ORIGINAL: SampleCollection = get_test_collection("original")
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, ORIGINAL.col_file)

        # Edit collection.
        shutil.copyfile(EDITED.path, ORIGINAL.col_file)

        directory = F.force_mkdir(Path("directory"))
        collision = PathCreationCollisionError(directory, "token")
        mocker.patch("ki.F.fmkleaves", side_effect=collision)

        os.chdir(ORIGINAL.repodir)
        with pytest.raises(PathCreationCollisionError):
            pull(runner)


@pytest.mark.skip
def test_pull_handles_unexpectedly_changed_checksums(mocker: MockerFixture):
    ORIGINAL: SampleCollection = get_test_collection("original")
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, ORIGINAL.col_file)

        # Edit collection.
        shutil.copyfile(EDITED.path, ORIGINAL.col_file)

        mocker.patch("ki.F.md5", side_effect=["good", "good", "good", "bad"])

        os.chdir(ORIGINAL.repodir)
        with pytest.raises(CollectionChecksumError):
            pull(runner)


@pytest.mark.skip
def test_pull_displays_errors_from_repo_initialization(mocker: MockerFixture):
    ORIGINAL: SampleCollection = get_test_collection("original")
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, ORIGINAL.col_file)

        # Edit collection.
        shutil.copyfile(EDITED.path, ORIGINAL.col_file)

        git.Repo.init(Path(ORIGINAL.repodir))
        effects = [git.InvalidGitRepositoryError()]
        mocker.patch("ki.M.repo", side_effect=effects)

        os.chdir(ORIGINAL.repodir)
        with pytest.raises(git.InvalidGitRepositoryError):
            pull(runner)


@pytest.mark.skip
def test_pull_handles_non_standard_submodule_branch_names(tmp_path: Path):
    ORIGINAL: SampleCollection = get_test_collection("original")
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        repo: git.Repo = get_repo_with_submodules(runner, ORIGINAL.col_file)
        os.chdir(repo.working_dir)

        # Copy a new note into the submodule.
        note_path = Path(repo.working_dir) / SUBMODULE_DIRNAME / "Default" / NOTE_2
        shutil.copyfile(NOTE_2_PATH, note_path)

        # Get a reference to the submodule repo.
        subrepo = git.Repo(Path(repo.working_dir) / SUBMODULE_DIRNAME)
        subrepo.git.branch(["-m", "main", "brain"])

        # Commit changes in submodule and parent repo.
        subrepo.git.add(all=True)
        subrepo.index.commit("Add a new note.")
        repo.git.add(all=True)
        repo.index.commit("Update submodule.")

        out = push(runner, verbose=True)

        # Edit collection (implicitly removes submodule).
        shutil.copyfile(EDITED.path, ORIGINAL.col_file)

        out = pull(runner)


@pytest.mark.skip
def test_pull_handles_uncommitted_submodule_commits(tmp_path: Path):
    UNCOMMITTED_SM: SampleCollection = get_test_collection(
        "uncommitted_submodule_commits"
    )
    UNCOMMITTED_SM_EDITED: SampleCollection = get_test_collection(
        "uncommitted_submodule_commits_edited"
    )
    runner = CliRunner()
    japanese_gitrepo_path = Path(JAPANESE_GITREPO_PATH).resolve()
    with runner.isolated_filesystem(temp_dir=tmp_path):

        JAPANESE_SUBMODULE_DIRNAME = "japanese-core-2000"

        # Clone collection.
        out = clone(runner, UNCOMMITTED_SM.col_file)

        # Check that the content of a note in the collection is correct.
        os.chdir(UNCOMMITTED_SM.repodir)
        with open(
            Path(JAPANESE_SUBMODULE_DIRNAME) / "それ.md", "r", encoding="UTF-8"
        ) as f:
            note_text = f.read()
            expected = "that, that one\nthat, that one\nthis, this one"
            assert expected in note_text
        os.chdir("../")

        # Delete `japanese-core-2000/` subdirectory, and commit.
        sm_dir = Path(UNCOMMITTED_SM.repodir) / JAPANESE_SUBMODULE_DIRNAME
        F.rmtree(F.test(sm_dir))
        repo = git.Repo(UNCOMMITTED_SM.repodir)
        repo.git.add(all=True)
        repo.index.commit("Delete cloned `japanese-core-2000` folder.")
        repo.close()

        # Push the deletion.
        os.chdir(UNCOMMITTED_SM.repodir)
        out = push(runner)

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
        out = push(runner)

        # Add a new line to a note, and commit the addition in the submodule.
        with open(
            Path(JAPANESE_SUBMODULE_DIRNAME) / "それ.md", "a", encoding="UTF-8"
        ) as f:
            f.write("A new line at the bottom.")
        sm.git.add(all=True)
        _ = sm.index.commit("Added a new line.")
        sm.close()

        # Edit collection.
        shutil.copyfile(UNCOMMITTED_SM_EDITED.col_file, UNCOMMITTED_SM.col_file)

        # Pull changes from collection to root ki repository.
        out = pull(runner)
        assert "fatal: remote error: " not in out
        assert "CONFLICT" not in out

        with open(
            Path(JAPANESE_SUBMODULE_DIRNAME) / "それ.md", "r", encoding="UTF-8"
        ) as f:
            note_text = f.read()
        expected_mackerel = "\nholy mackerel\n"
        expected_this = "\nthis, this one\n"
        assert expected_mackerel in note_text
        assert expected_this in note_text


@pytest.mark.skip
def test_pull_removes_files_deleted_in_remote(tmp_path: Path):
    ORIGINAL: SampleCollection = get_test_collection("original")
    DELETED: SampleCollection = get_test_collection("deleted")
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):

        # Clone collection in cwd.
        clone(runner, ORIGINAL.col_file)

        # Edit collection.
        shutil.copyfile(DELETED.path, ORIGINAL.col_file)

        os.chdir(ORIGINAL.repodir)
        out = pull(runner)


@pytest.mark.skip
def test_pull_does_not_duplicate_decks_converted_to_subdecks_of_new_top_level_decks(
    tmp_path: Path,
):
    BEFORE: SampleCollection = get_test_collection("duplicated_subdeck_before")
    AFTER: SampleCollection = get_test_collection("duplicated_subdeck_after")
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):

        # Clone.
        clone(runner, BEFORE.col_file)

        # Edit collection.
        shutil.copyfile(AFTER.path, BEFORE.col_file)

        # Pull.
        os.chdir(BEFORE.repodir)
        out = pull(runner)

        # Check.
        if os.path.isdir("onlydeck"):
            for _, _, filenames in os.walk("onlydeck"):
                assert len(filenames) == 0


# PUSH


@pytest.mark.skip
def test_push_writes_changes_correctly(tmp_path: Path):
    """If there are committed changes, does push change the collection file?"""
    ORIGINAL: SampleCollection = get_test_collection("original")
    old_notes = get_notes(ORIGINAL.col_file)
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):

        # Clone collection in cwd.
        out = clone(runner, ORIGINAL.col_file)

        # Edit a note.
        note = os.path.join(ORIGINAL.repodir, NOTE_0)
        with open(note, "a", encoding="UTF-8") as note_file:
            note_file.write("e\n")

        # Delete a note.
        note = os.path.join(ORIGINAL.repodir, NOTE_4)
        os.remove(note)

        # Add a note.
        shutil.copyfile(NOTE_2_PATH, os.path.join(ORIGINAL.repodir, NOTE_2))

        # Commit.
        repo = git.Repo(ORIGINAL.repodir)
        repo.git.add(all=True)
        repo.index.commit("Added 'e'.")

        # Push and check for changes.
        os.chdir(ORIGINAL.repodir)
        out = push(runner)
        new_notes = get_notes(ORIGINAL.col_file)

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
        assert len(old_notes) == 2
        assert len(new_notes) == 2


@pytest.mark.skip
def test_push_verifies_md5sum():
    """Does ki only push if md5sum matches last pull?"""
    ORIGINAL: SampleCollection = get_test_collection("original")
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, ORIGINAL.col_file)

        # Swap a bit.
        randomly_swap_1_bit(ORIGINAL.col_file)

        # Make sure ki complains.
        os.chdir(ORIGINAL.repodir)
        with pytest.raises(UpdatesRejectedError):
            push(runner)


@pytest.mark.skip
def test_push_generates_correct_backup():
    """Does push store a backup identical to old collection file?"""
    ORIGINAL: SampleCollection = get_test_collection("original")
    old_hash = F.md5(ORIGINAL.col_file)
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, ORIGINAL.col_file)

        # Make change in repo.
        note = os.path.join(ORIGINAL.repodir, NOTE_0)
        with open(note, "a", encoding="UTF-8") as note_file:
            note_file.write("e\n")

        # Commit.
        repo = git.Repo(ORIGINAL.repodir)
        repo.git.add(all=True)
        repo.index.commit("Added 'e'.")

        os.chdir(ORIGINAL.repodir)
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
    ORIGINAL: SampleCollection = get_test_collection("original")
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, ORIGINAL.col_file)

        # Make change in repo.
        note = os.path.join(ORIGINAL.repodir, NOTE_0)
        with open(note, "a", encoding="UTF-8") as note_file:
            note_file.write("e\n")

        # DON'T COMMIT, push.
        os.chdir(ORIGINAL.repodir)
        out = push(runner)
        assert "ki push: up to date." in out
        assert len(os.listdir(".ki/backups")) == 0


@pytest.mark.skip
def test_push_doesnt_fail_after_pull():
    """Does push work if we pull and then edit and then push?"""
    ORIGINAL: SampleCollection = get_test_collection("original")
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, ORIGINAL.col_file)
        assert not os.path.isfile(os.path.join(ORIGINAL.repodir, NOTE_1))

        # Edit collection.
        shutil.copyfile(EDITED.path, ORIGINAL.col_file)

        # Pull edited collection.
        os.chdir(ORIGINAL.repodir)
        pull(runner)
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
        os.chdir("../")
        repo = git.Repo(ORIGINAL.repodir)
        repo.git.add(all=True)
        repo.index.commit("Added 'e'.")
        repo.close()
        del repo
        gc.collect()

        # Push changes.
        os.chdir(ORIGINAL.repodir)
        push(runner)


@pytest.mark.skip
def test_no_op_push_is_idempotent():
    """Does push not misbehave if you keep pushing?"""
    ORIGINAL: SampleCollection = get_test_collection("original")
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, ORIGINAL.col_file)
        assert os.path.isdir(ORIGINAL.repodir)

        os.chdir(ORIGINAL.repodir)
        push(runner)
        push(runner)
        push(runner)
        push(runner)
        push(runner)
        push(runner)


@pytest.mark.skip
def test_push_deletes_notes():
    """Does push remove deleted notes from collection?"""
    ORIGINAL: SampleCollection = get_test_collection("original")
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, ORIGINAL.col_file)

        # Remove a note file.
        os.chdir(ORIGINAL.repodir)
        assert os.path.isfile(NOTE_0)
        os.remove(NOTE_0)

        # Commit the deletion.
        os.chdir("../")
        repo = git.Repo(ORIGINAL.repodir)
        repo.git.add(all=True)
        repo.index.commit("Added 'e'.")

        # Push changes.
        os.chdir(ORIGINAL.repodir)
        out = push(runner)

    # Check that note is gone.
    with runner.isolated_filesystem():
        clone(runner, ORIGINAL.col_file)
        assert not os.path.isfile(NOTE_0)


@pytest.mark.skip
def test_push_still_works_from_subdirectories():
    """Does push still work if you're farther down in the directory tree than the repo route?"""
    ORIGINAL: SampleCollection = get_test_collection("original")
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, ORIGINAL.col_file)

        # Remove a note file.
        os.chdir(ORIGINAL.repodir)
        assert os.path.isfile(NOTE_0)
        os.remove(NOTE_0)

        # Commit the deletion.
        os.chdir("../")
        repo = git.Repo(ORIGINAL.repodir)
        repo.git.add(all=True)
        repo.index.commit("Added 'e'.")

        # Push changes.
        os.chdir(os.path.join(ORIGINAL.repodir, "Default"))
        push(runner)


@pytest.mark.skip
def test_push_deletes_added_notes():
    """Does push remove deleted notes added with ki?"""
    ORIGINAL: SampleCollection = get_test_collection("original")
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, ORIGINAL.col_file)

        # Add new files.
        os.chdir(ORIGINAL.repodir)
        contents = os.listdir("Default")
        shutil.copyfile(NOTE_2_PATH, os.path.join("Default", NOTE_2))
        shutil.copyfile(NOTE_3_PATH, os.path.join("Default", NOTE_3))

        # Commit the additions.
        os.chdir("../")
        repo = git.Repo(ORIGINAL.repodir)
        repo.git.add(all=True)
        repo.index.commit("Added 'e'.")

        # Push changes.
        os.chdir(ORIGINAL.repodir)
        out = push(runner)

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
        os.chdir("../../")
        repo = git.Repo(ORIGINAL.repodir)
        repo.git.add(all=True)
        repo.index.commit("Added 'e'.")
        os.chdir(ORIGINAL.repodir)

        # Push changes.
        out = push(runner)

    # Check that notes are gone.
    with runner.isolated_filesystem():
        clone(runner, ORIGINAL.col_file)
        contents = os.listdir(os.path.join(ORIGINAL.repodir, "Default"))
        notes = [path for path in contents if path[-3:] == ".md"]
        assert len(notes) == 2


@pytest.mark.skip
def test_push_displays_informative_error_when_last_push_file_is_missing():
    ORIGINAL: SampleCollection = get_test_collection("original")
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, ORIGINAL.col_file)
        repo = git.Repo(ORIGINAL.repodir)

        last_push_path = Path(repo.working_dir) / ".ki" / "last_push"
        os.remove(last_push_path)

        # We should get a missing file error.
        os.chdir(ORIGINAL.repodir)
        with pytest.raises(MissingFileError):
            push(runner)


@pytest.mark.skip
def test_push_honors_ignore_patterns():
    ORIGINAL: SampleCollection = get_test_collection("original")
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, ORIGINAL.col_file)
        os.chdir(ORIGINAL.repodir)

        shutil.copyfile(NOTE_2_PATH, os.path.join("Default", NOTE_2))
        with open(".gitignore", "a", encoding="UTF-8") as ignore_f:
            ignore_f.write("\nelephant")

        repo = git.Repo(".")
        repo.git.add(all=True)
        repo.index.commit(".")

        out = push(runner, verbose=True)
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
        out = push(runner, verbose=True)
        assert "Warning: not Anki note" in out


@pytest.mark.skip
def test_push_displays_errors_from_head_ref_maybes(mocker: MockerFixture):
    ORIGINAL: SampleCollection = get_test_collection("original")
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone, edit, and commit.
        clone(runner, ORIGINAL.col_file)
        os.chdir(ORIGINAL.repodir)
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
    ORIGINAL: SampleCollection = get_test_collection("original")
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone, edit, and commit.
        clone(runner, ORIGINAL.col_file)
        os.chdir(ORIGINAL.repodir)
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
    ORIGINAL: SampleCollection = get_test_collection("original")
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone, edit, and commit.
        clone(runner, ORIGINAL.col_file)
        os.chdir(ORIGINAL.repodir)

        repo = git.Repo(".")

        shutil.copyfile(NOTE_2_PATH, os.path.join("Default", NOTE_2))
        repo = git.Repo(".")
        repo.git.add(all=True)
        repo.index.commit(".")

        col = open_collection(ORIGINAL.col_file)
        note = col.get_note(set(col.find_notes("")).pop())
        _: Notetype = ki.parse_notetype_dict(note.note_type())
        col.close()

        effects = [MissingFieldOrdinalError(3, "<notetype>")]

        mocker.patch("ki.parse_notetype_dict", side_effect=effects)

        with pytest.raises(MissingFieldOrdinalError):
            push(runner)


@pytest.mark.skip
def test_push_displays_errors_from_notetype_parsing_in_push_deltas_during_push_flatnote_to_anki(
    mocker: MockerFixture,
):
    ORIGINAL: SampleCollection = get_test_collection("original")
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone, edit, and commit.
        clone(runner, ORIGINAL.col_file)
        os.chdir(ORIGINAL.repodir)

        repo = git.Repo(".")

        shutil.copyfile(NOTE_2_PATH, os.path.join("Default", NOTE_2))
        repo = git.Repo(".")
        repo.git.add(all=True)
        repo.index.commit(".")

        col = open_collection(ORIGINAL.col_file)
        note = col.get_note(set(col.find_notes("")).pop())
        notetype: Notetype = ki.parse_notetype_dict(note.note_type())
        col.close()

        effects = [notetype] * PARSE_NOTETYPE_DICT_CALLS_PRIOR_TO_FLATNOTE_PUSH
        effects += [MissingFieldOrdinalError(3, "<notetype>")]

        mocker.patch("ki.parse_notetype_dict", side_effect=effects)

        with pytest.raises(MissingFieldOrdinalError):
            out = push(runner)


@pytest.mark.skip
def test_push_handles_submodules(tmp_path: Path):
    ORIGINAL: SampleCollection = get_test_collection("original")
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        repo: git.Repo = get_repo_with_submodules(runner, ORIGINAL.col_file)
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

        out = push(runner)

        colnotes = get_notes(ORIGINAL.col_file)
        notes: List[Note] = [colnote.n for colnote in colnotes]
        assert len(notes) == 3
        assert "<br>z<br>" in notes[0]["Back"]


def test_push_writes_media(tmp_path: Path):
    MEDIACOL: SampleCollection = get_test_collection("media")
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):

        # Clone.
        logger.debug(f"First clone...")
        clone(runner, MEDIACOL.col_file)

        # Add a new note file containing media, and the corresponding media file.
        root = F.cwd()
        media_note_path = root / MEDIACOL.repodir / "Default" / MEDIA_NOTE
        media_file_path = root / MEDIACOL.repodir / "Default" / MEDIA / MEDIA_FILENAME
        logger.debug(f"Copying media file to '{media_file_path}'")
        shutil.copyfile(MEDIA_NOTE_PATH, media_note_path)
        shutil.copyfile(MEDIA_FILE_PATH, media_file_path)
        os.chdir(MEDIACOL.repodir)

        mode: int = ki.filemode(media_file_path)
        logger.warning(f"{mode = }")

        # Commit the additions.
        repo = git.Repo(F.cwd())
        repo.git.add(all=True)
        repo.index.commit("Add air.md")
        repo.close()

        mode: int = ki.filemode(media_file_path)
        logger.warning(f"{mode = }")

        # Push the commit.
        out = push(runner)

        # Annihilate the repo root.
        os.chdir("../")
        F.rmtree(F.test(Path(MEDIACOL.repodir)))

        # Re-clone the pushed collection.
        logger.debug(f"Second clone...")
        out = clone(runner, MEDIACOL.col_file)

        # Check that added note and media file exist.
        col = open_collection(MEDIACOL.col_file)
        check = col.media.check()
        assert os.path.isfile(Path(MEDIACOL.repodir) / "Default" / MEDIA_NOTE)
        assert col.media.have(MEDIA_FILENAME)
        assert len(check.missing) == 0
        assert len(check.unused) == 0


@pytest.mark.skip
def test_push_handles_foreign_models(tmp_path: Path):
    """Just check that we don't return an exception from `push()`."""
    ORIGINAL: SampleCollection = get_test_collection("original")
    runner = CliRunner()
    japan_path = (Path(TEST_DATA_PATH) / "repos" / "japanese-core-2000").resolve()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        clone(runner, ORIGINAL.col_file)
        shutil.copytree(japan_path, Path(ORIGINAL.repodir) / "Default" / "japan")
        os.chdir(ORIGINAL.repodir)
        repo = git.Repo(F.cwd())
        repo.git.add(all=True)
        repo.index.commit("japan")
        out = push(runner)


@pytest.mark.skip
def test_push_fails_if_database_is_locked():
    """Does ki print a nice error message when Anki is accidentally left open?"""
    ORIGINAL: SampleCollection = get_test_collection("original")
    runner = CliRunner()
    japan_path = (Path(TEST_DATA_PATH) / "repos" / "japanese-core-2000").resolve()
    with runner.isolated_filesystem():
        clone(runner, ORIGINAL.col_file)
        shutil.copytree(japan_path, Path(ORIGINAL.repodir) / "Default" / "japan")
        os.chdir(ORIGINAL.repodir)
        repo = git.Repo(F.cwd())
        repo.git.add(all=True)
        repo.index.commit("japan")
        con = sqlite3.connect(ORIGINAL.col_file)
        con.isolation_level = "EXCLUSIVE"
        con.execute("BEGIN EXCLUSIVE")
        with pytest.raises(SQLiteLockError):
            push(runner)


@pytest.mark.skip
def test_push_is_nontrivial_when_pulled_changes_are_reverted(tmp_path: Path):
    """
    If you push, make changes in Anki, then pull those changes, then undo them
    within the ki repo, then push again, the push should *not* be a no-op. The
    changes are currently applied in Anki, and the push should undo them.
    """
    ORIGINAL: SampleCollection = get_test_collection("original")
    COPY: SampleCollection = get_test_collection("original")
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):

        # Clone collection in cwd.
        clone(runner, ORIGINAL.col_file)

        # Remove a note file.
        os.chdir(ORIGINAL.repodir)
        assert os.path.isfile(NOTE_0)
        os.remove(NOTE_0)

        # Commit the deletion.
        os.chdir("../")
        repo = git.Repo(ORIGINAL.repodir)
        repo.git.add(all=True)
        repo.index.commit("Deleted.")

        # Push changes.
        os.chdir(ORIGINAL.repodir)
        out = push(runner)
        notes = get_notes(ORIGINAL.col_file)
        notes = [colnote.n["Front"] for colnote in notes]
        assert notes == ["c"]

        # Revert the collection.
        os.remove(ORIGINAL.col_file)
        shutil.copyfile(COPY.col_file, ORIGINAL.col_file)

        # Pull again.
        out = pull(runner)

        # Remove again.
        assert os.path.isfile(NOTE_0)
        os.remove(NOTE_0)
        repo = git.Repo(F.cwd())
        repo.git.add(all=True)
        repo.index.commit("Deleted.")

        # Push changes.
        out = push(runner)
        notes = get_notes(ORIGINAL.col_file)
        notes = [colnote.n["Front"] for colnote in notes]
        assert "a" not in notes
        assert notes == ["c"]
        assert "ki push: up to date." not in out


@pytest.mark.skip
def test_push_doesnt_unnecessarily_deduplicate_notetypes():
    """
    Does push refrain from adding a new notetype if the requested notetype
    already exists in the collection?
    """
    ORIGINAL: SampleCollection = get_test_collection("original")
    COPY: SampleCollection = get_test_collection("original")
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, ORIGINAL.col_file)

        col = open_collection(ORIGINAL.col_file)
        orig_models = col.models.all_names_and_ids()
        col.close(save=False)

        # Remove a note file.
        os.chdir(ORIGINAL.repodir)
        assert os.path.isfile(NOTE_0)
        os.remove(NOTE_0)

        # Commit the deletion.
        os.chdir("../")
        repo = git.Repo(ORIGINAL.repodir)
        repo.git.add(all=True)
        repo.index.commit("Deleted.")

        # Push changes.
        os.chdir(ORIGINAL.repodir)
        out = push(runner)

        # Revert the collection.
        os.remove(ORIGINAL.col_file)
        shutil.copyfile(COPY.col_file, ORIGINAL.col_file)

        # Pull again.
        out = pull(runner)

        # Remove again.
        assert os.path.isfile(NOTE_0)
        os.remove(NOTE_0)
        repo = git.Repo(F.cwd())
        repo.git.add(all=True)
        repo.index.commit("Deleted.")

        # Push changes.
        out = push(runner)

        col = open_collection(ORIGINAL.col_file)
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
    ORIGINAL: SampleCollection = get_test_collection("original")
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, ORIGINAL.col_file)

        # Remove a note file.
        os.chdir(ORIGINAL.repodir)
        assert os.path.isfile(NOTE_0)
        temp_note_0_file = F.mkdtemp() / "NOTE_0"
        shutil.move(NOTE_0, temp_note_0_file)
        assert not os.path.isfile(NOTE_0)

        # Commit the deletion.
        os.chdir("../")
        repo = git.Repo(ORIGINAL.repodir)
        repo.git.add(all=True)
        repo.index.commit("Deleted.")

        # Push changes.
        os.chdir(ORIGINAL.repodir)
        out = push(runner)

        # Put file back.
        shutil.move(temp_note_0_file, NOTE_0)
        repo.git.add(all=True)
        repo.index.commit("Added.")

        # Push again.
        out = push(runner)
        assert "ki push: up to date." not in out


@pytest.mark.skip
def test_push_changes_deck_for_moved_notes():
    MULTIDECK: SampleCollection = get_test_collection("multideck")
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, MULTIDECK.col_file)

        # Move a note.
        os.chdir(MULTIDECK.repodir)
        target = "aa/dd/cc.md"
        assert os.path.isfile(MULTI_NOTE_PATH)
        shutil.move(MULTI_NOTE_PATH, target)
        assert not os.path.isfile(MULTI_NOTE_PATH)

        # Commit the move.
        os.chdir("../")
        repo = git.Repo(MULTIDECK.repodir)
        repo.git.add(all=True)
        repo.index.commit("Move.")

        # Push changes.
        os.chdir(MULTIDECK.repodir)
        out = push(runner)

        # Check that deck has changed.
        notes: List[ColNote] = get_notes(MULTIDECK.col_file)
        notes = filter(lambda colnote: colnote.n.id == MULTI_NOTE_ID, notes)
        notes = list(notes)
        assert len(notes) == 1
        colnote = notes.pop()
        assert colnote.deck == "aa::dd"


@pytest.mark.skip
def test_push_is_trivial_for_committed_submodule_contents(tmp_path: Path):
    UNCOMMITTED_SM: SampleCollection = get_test_collection(
        "uncommitted_submodule_commits"
    )
    runner = CliRunner()
    japanese_gitrepo_path = Path(JAPANESE_GITREPO_PATH).resolve()
    with runner.isolated_filesystem(temp_dir=tmp_path):

        JAPANESE_SUBMODULE_DIRNAME = "japanese-core-2000"

        # Clone collection in cwd.
        out = clone(runner, UNCOMMITTED_SM.col_file)

        # Delete a directory.
        sm_dir = Path(UNCOMMITTED_SM.repodir) / JAPANESE_SUBMODULE_DIRNAME
        F.rmtree(F.test(sm_dir))
        repo = git.Repo(UNCOMMITTED_SM.repodir)
        repo.git.add(all=True)
        repo.index.commit("Delete cloned `japanese-core-2000` folder.")

        # Push deletion.
        os.chdir(UNCOMMITTED_SM.repodir)
        out = push(runner)

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

        out = push(runner)
        out = push(runner)
        assert "ki push: up to date." in out


@pytest.mark.skip
def test_push_prints_informative_warning_on_push_when_subrepo_was_added_instead_of_submodule(
    tmp_path: Path,
):
    ORIGINAL: SampleCollection = get_test_collection("original")
    runner = CliRunner()
    japanese_gitrepo_path = Path(JAPANESE_GITREPO_PATH).resolve()
    with runner.isolated_filesystem(temp_dir=tmp_path):

        JAPANESE_SUBMODULE_DIRNAME = "japanese-core-2000"

        # Clone collection in cwd.
        clone(runner, ORIGINAL.col_file)
        os.chdir(ORIGINAL.repodir)

        # Add a *subrepo* (not submodule).
        submodule_name = JAPANESE_SUBMODULE_DIRNAME
        shutil.copytree(japanese_gitrepo_path, submodule_name)

        repo = git.Repo(".")
        p = subprocess.run(
            ["git", "add", "--all"], capture_output=True, encoding="UTF-8"
        )
        if "warning" in p.stderr:
            repo.index.commit("Add subrepo.")
            repo.close()
            out = push(runner)
            assert "'git submodule add'" in out


@pytest.mark.skip
def test_push_handles_tags_containing_trailing_commas():
    COMMAS: SampleCollection = get_test_collection("commas")
    runner = CliRunner()
    with runner.isolated_filesystem():

        # Clone collection in cwd.
        clone(runner, COMMAS.col_file)
        os.chdir(COMMAS.repodir)

        c_file = Path("Default") / "c.md"
        with open(c_file, "r", encoding="UTF-8") as read_f:
            contents = read_f.read().replace("tag2", "tag3")
            with open(c_file, "w", encoding="UTF-8") as write_f:
                write_f.write(contents)

        repo = git.Repo(".")
        repo.git.add(all=True)
        repo.index.commit("e")
        repo.close()

        push(runner)


@pytest.mark.skip
def test_push_correctly_encodes_quotes_in_html_tags(tmp_path: Path):
    BROKEN: SampleCollection = get_test_collection("broken_media_links")
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):

        # Clone collection in cwd.
        clone(runner, BROKEN.col_file)
        os.chdir(BROKEN.repodir)

        note_file = (
            Path("🧙‍Recommendersysteme") / "wie-sieht-die-linkstruktur-von-einem-hub.md"
        )
        with open(note_file, "r", encoding="UTF-8") as read_f:
            contents = read_f.read().replace("guter", "guuter")
            with open(note_file, "w", encoding="UTF-8") as write_f:
                write_f.write(contents)

        repo = git.Repo(".")
        repo.git.add(all=True)
        repo.index.commit("e")
        repo.close()

        push(runner)

        notes = get_notes(BROKEN.col_file)
        colnote = notes.pop()
        back: str = colnote.n["Back"]
        col = Collection(BROKEN.col_file)
        escaped: str = col.media.escape_media_filenames(back)
        col.close()
        assert (
            '<img src="paste-64c7a314b90f3e9ef1b2d94edb396e07a121afdf.jpg">' in escaped
        )
