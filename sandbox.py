import os
import random
import shutil
import hashlib
import sqlite3
import tempfile
import subprocess
from distutils.dir_util import copy_tree
from importlib.metadata import version

import git
import anki
import pytest
import bitstring
import checksumdir
from beartype import beartype
from loguru import logger
from click.testing import CliRunner

import ki

from tests.test_ki import get_collection_path, clone


def test_clone_fails_if_collection_doesnt_exist():
    """Does ki clone only if `.anki2` file exists?"""
    collection_path = get_collection_path()

    runner = CliRunner()
    with runner.isolated_filesystem():
        clone(runner, collection_path)


if __name__ == "__main__":
    test_clone_fails_if_collection_doesnt_exist()
