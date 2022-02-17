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
import anki
import click
from apy.anki import Anki
from loguru import logger
from beartype import beartype


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
    if not os.path.isfile(collection):
        raise FileNotFoundError

    # Import with apy.
    with Anki(path=collection) as a:
        a.col.find_notes("")
        notes = list(a.find_notes(query=""))
        logger.info(notes)


@ki.command()
@beartype
def pull() -> None:
    """
    Pull from a preconfigured remote Anki collection into an existing ki
    repository.
    """
    pass


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
