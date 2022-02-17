#!/usr/bin/env python3
"""
Python package `ki` is a command-line interface for the version control and
editing of ``.anki2`` collections as git repositories of markdown files.
Rather than providing an interactive UI like the Anki desktop client, `ki` aims
to allow natural editing *in the filesystem*.

In general, the purpose of `ki` is to allow users to work on large, complex
Anki decks in exactly the same way they work on large, complex software
projects.
.. include:: ./DOCUMENTATION.md
"""

# pylint:disable=unnecessary-pass

__author__ = ""
__email__ = ""
__license__ = "AGPLv3"
__url__ = ""
__version__ = "0.0.1a"

import os
import logging
import hashlib
import tempfile
import subprocess
import configparser
from typing import List

import git
import anki
import click
from apy.anki import Anki, Note
from loguru import logger
from beartype import beartype

from ki.note import KiNote


logging.basicConfig(level=logging.INFO)


@click.group()
@click.version_option()
@beartype
def ki() -> None:
    """
    The universal CLI entry point for `ki`.

    Takes no arguments, only has three subcommands (clone, pull, push).
    """
    return


@ki.command()
@click.argument("collection")
@click.argument("directory", required=False, default="")
def clone(collection: str, directory: str = "") -> None:
    """
    Clone an Anki collection into a directory.

    Parameters
    ----------
    collection : str
        The path to a `.anki2` collection file.
    directory : str, default=""
        An optional path to a directory to clone the collection into.
        Note: we check that this directory does not yet exist.
    """
    _clone(collection, directory)


def _clone(collection: str, directory: str = "") -> None:
    """Clone an Anki collection into a directory."""
    collection = os.path.abspath(collection)
    if not os.path.isfile(collection):
        raise FileNotFoundError

    logger.debug(f"clone was called with args: {collection}, {directory}")

    # Generate default target directory.
    if directory == "":
        directory = get_default_clone_directory(collection)
    directory = os.path.abspath(directory)

    # Create target directory.
    os.mkdir(directory)

    # Create .ki subdirectory.
    kidir = os.path.join(directory, ".ki/")
    os.mkdir(kidir)

    # Create config file.
    config_path = os.path.join(kidir, "config")
    config = configparser.ConfigParser()
    config["remote"] = {"path": collection}
    with open(config_path, "w", encoding="UTF-8") as config_file:
        config.write(config_file)

    # Create hashes file.
    hashes_path = os.path.join(kidir, "hashes")
    with open(hashes_path, "a", encoding="UTF-8") as hashes_file:
        hashes_file.write(md5(collection))

    # Run git init.
    repo = git.Repo.init(directory)

    # Add `.ki/hashes` to gitignore.
    ignore_path = os.path.join(directory, ".gitignore")
    with open(ignore_path, "w", encoding="UTF-8") as ignore_file:
        ignore_file.write(".ki/hashes\n")

    # Import with apy.
    query = ""
    with Anki(path=collection) as a:
        notes: List[KiNote] = []
        for i in set(a.col.find_notes(query)):
            notes.append(KiNote(a, a.col.getNote(i)))

    # Dump notes.
    for i, note in enumerate(notes):
        note_path = os.path.join(directory, f"note{i}.md")
        with open(note_path, "w", encoding="UTF-8") as note_file:
            note_file.write(str(note))

    # Add and commit all contents.
    repo.git.add(all=True)
    repo.index.commit("Initial commit")


@ki.command()
@beartype
def pull() -> None:
    """
    Pull from a preconfigured remote Anki collection into an existing ki
    repository.
    """
    # Check that config file exists.
    config_path = os.path.join(os.getcwd(), ".ki/", "config")
    if not os.path.isfile(config_path):
        raise FileNotFoundError

    # Parse config file.
    config = configparser.ConfigParser()
    config.read(config_path)
    collection = config["remote"]["path"]

    if not os.path.isfile(collection):
        raise FileNotFoundError

    # Create a temp directory root.
    tempdir = tempfile.mkdtemp()
    root = os.path.join(tempdir, "ki/", "remote/")
    os.makedirs(root)

    # Clone into an ephemeral repository.
    ephem = os.path.join(root, md5(collection))
    logger.debug(f"calling clone with args: {collection}, {ephem}")
    _clone(collection, ephem)

    # Create remote pointing to ephemeral repository and pull.
    repo = git.Repo(os.getcwd())
    origin = repo.create_remote("origin", os.path.join(ephem, ".git"))
    _git = repo.git
    _git.config("pull.rebase", "false")

    # Won't use this because we can't print stdout/stderr for the user.
    # origin.pull("main", allow_unrelated_histories=True)

    p = subprocess.run(["git", "pull", "-v", "--allow-unrelated-histories", "origin", "main"], check=False, capture_output=True)
    logger.info(f"\n{p.stdout.decode()}")
    logger.info(f"\n{p.stderr.decode()}")


@ki.command()
@beartype
def push() -> None:
    """
    Pack a ki repository into a .anki2 file and push to collection location.
    """
    pass


# UTILS


@beartype
def get_default_clone_directory(collection_path: str) -> str:
    """ "
    Get the default clone directory path.

    This should just be the name of the collection (which is usually a file
    called `collection.anki2`) so this will usually be `./collection/`.

    Parameters
    ----------
    collection_path : str
        The path to a `.anki2` collection file.

    Returns
    -------
    str
        The path to clone into.
    """
    basename = os.path.basename(collection_path)
    sections = os.path.splitext(basename)
    assert len(sections) == 2
    return os.path.abspath(sections[0])


@beartype
def md5(path: str) -> str:
    """Compute md5sum of file at `path`."""
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
