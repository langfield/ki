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

from tests.test_ki import get_collection_path, clone, REPODIR, UPDATED_COLLECTION_PATH, pull

import logging
logging.basicConfig(level=logging.INFO)

def main():
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


if __name__ == "__main__":
    main()
