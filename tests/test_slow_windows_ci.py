#!/usr/bin/env python3
"""Tests for ki command line interface (CLI)."""
import os
from importlib.metadata import version

from beartype import beartype
from beartype.typing import List

import ki
from tests.test_ki import invoke


# CLI

import subprocess


# POTENTIALLY SLOW STUFF


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
    DELETED_COLLECTION_PATH,
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
    JAPANESE_GITREPO_PATH,
    UNCOMMITTED_SM_ERROR_REPODIR,
    UNCOMMITTED_SM_ERROR_EDITED_PATH,
    get_uncommitted_sm_pull_exception_col_file,
    BRANCH_NAME,
)


PARSE_NOTETYPE_DICT_CALLS_PRIOR_TO_FLATNOTE_PUSH = 2


@beartype
def capture(args: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        text=True,
        capture_output=True,
    )


def test_version():
    """Does --version display information as expected?"""
    expected_version = version("ki")
    result = invoke(ki.ki, ["--version"])

    assert result.stdout.rstrip() == f"ki, version {expected_version}"
    assert result.exit_code == 0


def test_version_clickless():
    """Does --version display information as expected?"""
    expected_version = version("ki")
    p = capture(["ki", "--version"])

    assert p.stdout.rstrip() == f"ki, version {expected_version}"
    assert p.returncode == 0
