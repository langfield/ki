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

__author__ = ''
__email__ = ''
__license__ = 'AGPLv3'
__url__ = ''
__version__ = '0.0.1a'

import click
from beartype import beartype


@click.group()
@beartype
def ki() -> None:
    """
    The universal CLI entry point for `ki`.

    Takes no arguments, only has three subcommands (clone, pull, push).
    """
    pass


@ki.command()
@click.argument("collection")
@click.argument("directory", required=False, default="")
@beartype
def clone(collection_path: str, directory_path: str = "") -> None:
    """
    Clone an Anki collection into a directory.

    Parameters
    ----------
    collection_path : str
        The path to a `.anki2` collection file.
    directory_path : str, default=""
        An optional path to a directory to clone the collection into.
        Note: we check that this directory does not yet exist.
    """
    pass


@beartype
def get_default_clone_directory(collection_path: str) -> str:
    """"
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
    pass
