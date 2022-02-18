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
import re
import shutil
import logging
import tarfile
import hashlib
import tempfile
import subprocess
import configparser
from typing import List, Dict, Any

import git
import anki
import click
from loguru import logger
from beartype import beartype
from apy.anki import Anki, Note
from apy.convert import markdown_to_html, plain_to_html, markdown_file_to_notes

from ki.note import KiNote


logging.basicConfig(level=logging.INFO)


# pylint: disable=invalid-name
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
    basename = os.path.basename(collection)
    hashes_path = os.path.join(kidir, "hashes")
    with open(hashes_path, "a", encoding="UTF-8") as hashes_file:
        hashes_file.write(f"{md5(collection)}  {basename}")

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

    # Lock DB and get hash.
    # TODO: lock DB.
    md5sum = md5(collection)

    # TODO: Everything above this line could be in a function called
    # `open_repo()` that is called at the beginning of `pull` and `push`.

    # TODO: The following block could be in a function called
    # `get_latest_collection_hash()`.
    # Quit if hash matches last pull.
    kidir = os.path.join(os.getcwd(), ".ki/")
    hashes_path = os.path.join(kidir, "hashes")
    with open(hashes_path, "r", encoding="UTF-8") as hashes_file:
        if md5sum in hashes_file.readlines()[-1]:
            logger.info("Up to date.")
            # TODO: unlock DB.
            return

    # Create a temp directory root.
    tempdir = tempfile.mkdtemp()
    root = os.path.join(tempdir, "ki/", "remote/")
    os.makedirs(root)

    # Clone into an ephemeral repository.
    ephem = os.path.join(root, md5sum)
    _clone(collection, ephem)

    # TODO: unlock DB.

    # Create remote pointing to ephemeral repository and pull.
    repo = git.Repo(os.getcwd())
    _ = repo.create_remote("origin", os.path.join(ephem, ".git"))
    _git = repo.git
    _git.config("pull.rebase", "false")

    # Won't use this because we can't print stdout/stderr for the user.
    # origin.pull("main", allow_unrelated_histories=True)

    p = subprocess.run(
        ["git", "pull", "-v", "--allow-unrelated-histories", "origin", "main"],
        check=False,
        capture_output=True,
    )
    logger.info(f"\n{p.stdout.decode()}")
    logger.info(f"\n{p.stderr.decode()}")

    # Append to hashes file.
    basename = os.path.basename(collection)
    kidir = os.path.join(os.getcwd(), ".ki/")
    hashes_path = os.path.join(kidir, "hashes")
    with open(hashes_path, "a", encoding="UTF-8") as hashes_file:
        hashes_file.write(f"{md5sum}  {basename}")


@ki.command()
@beartype
def push() -> None:
    """
    Pack a ki repository into a .anki2 file and push to collection location.
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

    # Lock DB and get hash.
    # TODO: lock DB.
    md5sum = md5(collection)

    # Quit if hash doesn't match last pull.
    kidir = os.path.join(os.getcwd(), ".ki/")
    hashes_path = os.path.join(kidir, "hashes")
    with open(hashes_path, "r", encoding="UTF-8") as hashes_file:
        if md5sum not in hashes_file.readlines()[-1]:
            click.echo(f"Failed to push some refs to '{collection}'")
            click.echo(
                "hint: Updates were rejected because the tip of your current branch is behind"
            )
            click.echo(
                "hint: the Anki remote collection. Integrate the remote changes (e.g."
            )
            click.echo("hint: 'ki pull ...') before pushing again.")
            # TODO: unlock DB.

    # Create a temp directory root.
    tempdir = tempfile.mkdtemp()
    root = os.path.join(tempdir, "ki/", "local/")
    os.makedirs(root)

    # Clone into an ephemeral repository.
    ephem = os.path.join(root, md5sum)

    # Clone repository at latest commit in `/tmp/.../ki/local/`.
    repo = git.Repo(os.getcwd())
    git.Repo.clone_from(os.getcwd(), ephem, branch=repo.active_branch)

    # Get path to new collection.
    coll_root = os.path.join(root, "coll")
    os.makedirs(coll_root)
    basename = os.path.basename(collection)
    new_collection = os.path.join(coll_root, basename)
    assert os.path.isdir(coll_root)
    assert not os.path.isfile(new_collection)
    assert not os.path.isdir(new_collection)

    # Copy collection to new collection.
    shutil.copyfile(collection, new_collection)

    # Get all files in checked-out ephemeral repository.
    files = [path for path in os.listdir(ephem) if os.path.isfile(path)]
    note_paths = [file for file in files if is_anki_note(file)]

    # Generate `.anki2` file.
    # We must distinguish between new notes and old, edited notes.
    # So if they have an nid, they are old notes.
    # If an `nid` is provided that is not already extant in the deck, we should
    # raise an error and tell the user not to make up `nid`s.
    with Anki(path=new_collection) as a:

        # List of note dictionaries as defined in `apy.convert`.
        notemaps: List[Dict[str, Any]] = []
        for note_path in note_paths:

            # `apy` support multiple notes per-markdown file. Should we?
            notemaps.extend(markdown_file_to_notes(note_path))

            # This generates a new `nid`, so we don't want to use it.
            # notes = a.add_notes_from_file(note_path)

        for notemap in notemaps:
            nid = int(notemap["nid"])
            note: Note = Note(a, a.col.get_note(nid))
            update_apy_note(note, notemap)

    assert os.path.isfile(new_collection)

    # Backup collection.
    backupsdir = os.path.join(os.getcwd(), ".ki/", "backups")
    assert not os.path.isfile(backupsdir)
    if not os.path.isdir(backupsdir):
        os.mkdir(backupsdir)
    backup_path = os.path.join(backupsdir, f"{md5sum}.anki2")
    assert not os.path.isfile(backup_path)
    shutil.copyfile(collection, backup_path)

    # Overwrite remote collection.
    shutil.copyfile(new_collection, collection)

    # TODO: unlock DB.


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


@beartype
def is_anki_note(path: str) -> bool:
    """Check if file is an `apy`-style markdown anki note."""
    # Ought to have markdown file extension.
    if path[-3:] != ".md":
        return False
    with open(path, "r", encoding="UTF-8") as md_file:
        lines = md_file.readlines()
    if len(lines) < 2:
        return False
    if lines[0] != "# Note\n":
        return False
    if not re.match(r"^nid: [0-9]+$", lines[1]):
        return False
    return True


@beartype
def update_apy_note(note: Note, notemap: Dict[str, Any]) -> None:
    """Update an `apy` Note in a collection."""
    new_tags = notemap["tags"].split()
    if new_tags != note.n.tags:
        note.n.tags = new_tags

    new_deck = notemap.get("deck", None)
    if new_deck is not None and new_deck != note.get_deck():
        note.set_deck(new_deck)

    for i, value in enumerate(notemap["fields"].values()):
        if notemap["markdown"]:
            note.n.fields[i] = markdown_to_html(value)
        else:
            note.n.fields[i] = plain_to_html(value)

    note.n.flush()
    note.a.modified = True
    if note.n.dupeOrEmpty():
        click.confirm(
            "The updated note is now a dupe!", prompt_suffix="", show_default=False
        )
