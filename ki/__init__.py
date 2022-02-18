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

# pylint:disable=unnecessary-pass, too-many-lines

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
import sqlite3
import tempfile
import subprocess
import configparser

import git
import anki
import click
import gitdb
from tqdm import tqdm
from loguru import logger

from apy.anki import Anki, Note
from apy.convert import markdown_to_html, plain_to_html, markdown_file_to_notes

from beartype import beartype
from beartype.typing import List, Dict, Any, Iterator

from ki.note import KiNote


logging.basicConfig(level=logging.INFO)


REMOTE_NAME = "anki"
HINT = (
    "hint: Updates were rejected because the tip of your current branch is behind\n"
    + "hint: the Anki remote collection. Integrate the remote changes (e.g.\n"
    + "hint: 'ki pull ...') before pushing again."
)


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
    sha = _clone(collection, directory)

    # Stuff below this line should not happen in a `pull()`.

    # Create initial commit SHA file.
    if directory == "":
        directory = get_default_clone_directory(collection)
    kidir = os.path.join(directory, ".ki/")
    initial_path = os.path.join(kidir, "initial")
    with open(initial_path, "a", encoding="UTF-8") as initial_file:
        initial_file.write(f"{sha}")

    # Commit `.ki/` directory with initial commit SHA file.
    repo = git.Repo.init(directory)
    repo.git.add([".ki/"])
    repo.index.commit("Initial SHA")


@beartype
def _clone(collection: str, directory: str = "") -> str:
    """Clone an Anki collection into a directory."""
    collection = os.path.abspath(collection)
    if not os.path.isfile(collection):
        raise FileNotFoundError

    # Create default target directory.
    if directory == "":
        directory = get_default_clone_directory(collection)
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

    # Add `.ki/hashes` and `.ki/backups` to gitignore.
    ignore_path = os.path.join(directory, ".gitignore")
    with open(ignore_path, "w", encoding="UTF-8") as ignore_file:
        ignore_file.write(".ki/hashes\n")
        ignore_file.write(".ki/backups\n")

    # Open deck with `apy`, and dump notes and markdown files.
    query = ""
    with Anki(path=collection) as a:
        for i in a.col.find_notes(query):
            note = KiNote(a, a.col.getNote(i))
            note_path = os.path.join(directory, f"note{note.n.id}.md")
            with open(note_path, "w", encoding="UTF-8") as note_file:
                note_file.write(str(note))

    # Initialize git repo and commit contents.
    repo = git.Repo.init(directory)
    repo.git.add(all=True)
    commit = repo.index.commit("Initial commit")

    # Return SHA of initial commit.
    return str(commit)


@ki.command()
@beartype
def pull() -> None:
    """
    Pull from a preconfigured remote Anki collection into an existing ki
    repository.
    """
    # Lock DB and get hash.
    collection = open_repository()
    con = lock(collection)
    md5sum = md5(collection)

    # Quit if hash matches last pull.
    if md5sum in get_latest_collection_hash():
        logger.info("Up to date.")
        unlock(con)
        return

    # Git clone local repository at SHA of last FETCH in `/tmp/.../ki/local/<md5sum>`.
    root = os.path.join(tempfile.mkdtemp(), "ki/", "local/")
    os.makedirs(root)
    cwd = os.getcwd()
    fetch_head_dir = os.path.join(root, md5sum)
    repo = git.Repo(cwd)
    git.Repo.clone_from(cwd, fetch_head_dir, branch=repo.active_branch)
    logger.debug(f"fetch_head_dir: {os.listdir(fetch_head_dir)}")

    # Do a reset --hard to the SHA of last FETCH.
    fetch_head_sha = get_fetch_head_sha(repo)
    fetch_head_repo = git.Repo(fetch_head_dir)
    fetch_head_repo.index.reset(fetch_head_sha, hard=True)

    # Ki clone into ephemeral repository and unlock DB.
    root = os.path.join(tempfile.mkdtemp(), "ki/", "remote/")
    os.makedirs(root)
    anki_remote_dir = os.path.join(root, md5sum)
    _clone(collection, anki_remote_dir)
    unlock(con)
    logger.debug(f"anki_remote_dir: {os.listdir(anki_remote_dir)}")

    # Create remote pointing to anki repository and pull into ``fetch_head_repo``.
    os.chdir(fetch_head_dir)
    anki_remote_path = os.path.join(anki_remote_dir, ".git")
    anki_remote = fetch_head_repo.create_remote(REMOTE_NAME, anki_remote_path)

    # .ki/initial is deleted here, why?
    diff = fetch_head_repo.git.diff(repo.head.commit.tree)
    logger.debug(f"Diff: {diff}")

    # Commit the deletion of .ki/initial because it doesn't matter.
    fetch_head_repo.git.add(all=True)
    fetch_head_repo.index.commit("Deleted '.ki/initial'")

    # Actually pull.
    fetch_head_repo.git.config("pull.rebase", "false")
    p = subprocess.run(
        [
            "git",
            "pull",
            "-v",
            "--allow-unrelated-histories",
            "--strategy-option",
            "theirs",
            REMOTE_NAME,
            "main",
        ],
        check=False,
        capture_output=True,
    )
    logger.info(f"\n{p.stdout.decode()}")
    logger.info(f"\n{p.stderr.decode()}")
    click.secho(f"\n{p.stdout.decode()}", blink=True)
    click.secho(f"\n{p.stderr.decode()}", bold=True)
    assert p.returncode == 0

    # Delete the remote we added.
    fetch_head_repo.delete_remote(anki_remote)

    # Create remote pointing to ``fetch_head`` repository and pull into ``repo``.
    os.chdir(cwd)
    fetch_head_remote_path = os.path.join(fetch_head_dir, ".git")
    fetch_head_remote = repo.create_remote(REMOTE_NAME, fetch_head_remote_path)
    repo.git.config("pull.rebase", "false")
    p = subprocess.run(
        ["git", "pull", "-v", REMOTE_NAME, "main"],
        check=False,
        capture_output=True,
    )
    logger.info(f"\n{p.stdout.decode()}")
    logger.info(f"\n{p.stderr.decode()}")
    click.secho(f"\n{p.stdout.decode()}", blink=True)
    click.secho(f"\n{p.stderr.decode()}", bold=True)
    assert p.returncode == 0

    # Delete the remote we added.
    repo.delete_remote(fetch_head_remote)

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
    # Lock DB, get path to collection, and compute hash.
    collection = open_repository()
    con = lock(collection)
    md5sum = md5(collection)

    # Quit if hash doesn't match last pull.
    if md5sum not in get_latest_collection_hash():
        click.echo(f"Failed to push some refs to '{collection}'\n{HINT}")
        unlock(con)
        return

    # Git clone repository at latest commit in `/tmp/.../ki/local/<md5sum>`.
    root = os.path.join(tempfile.mkdtemp(), "ki/", "local/")
    os.makedirs(root)
    cwd = os.getcwd()
    ephem = os.path.join(root, md5sum)
    repo = git.Repo(cwd)
    git.Repo.clone_from(cwd, ephem, branch=repo.active_branch)

    # Get path to new collection.
    new_collection = os.path.join(root, os.path.basename(collection))
    assert not os.path.isfile(new_collection)
    assert not os.path.isdir(new_collection)

    # Get all changed notes in checked-out ephemeral repository.
    ephem_repo = git.Repo(ephem)
    notepaths: Iterator[str] = get_files_changed_since_last_fetch(ephem_repo)

    # Copy collection to new collection and modify in-place.
    shutil.copyfile(collection, new_collection)
    with Anki(path=new_collection) as a:

        # List of note dictionaries as defined in `apy.convert`.
        notemaps: List[Dict[str, Any]] = []
        for notepath in tqdm(notepaths):

            # Support multiple notes-per-file.
            notemaps = markdown_file_to_notes(notepath)
            for notemap in notemaps:

                # Read `nid` from notemap, raise error if not found.
                try:
                    nid = int(notemap["nid"])
                except KeyError as err:
                    logger.debug(f"notemap: {notemap}")
                    logger.debug(f"path: {notepath}")
                    raise err

                # Look for `nid` and update existing note if found.
                try:
                    note: Note = Note(a, a.col.get_note(nid))
                    update_apy_note(note, notemap)

                # Otherwise, add a new note.
                except anki.errors.NotFoundError:
                    note: Note = add_note_from_notemap(a, notemap)
                    logger.info(f"Couldn't find note with nid: '{nid}'")
                    logger.info(f"Assigned new nid: '{note.n.id}'")

    assert os.path.isfile(new_collection)

    backup(collection)
    shutil.copyfile(new_collection, collection)
    unlock(con)


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
    fields_health_check = note.n.fields_check()

    if fields_health_check == 1:
        logger.warning(f"Found empty note:\n {note}")
        return
    if fields_health_check == 2:
        # logger.warning(f"Found duplicate note:\n {note}")
        return

    if fields_health_check:
        logger.warning(f"Found duplicate or empty note:\n {note}")
        logger.debug(f"Fields health check: {fields_health_check}")
        logger.debug(f"Fields health check (type): {type(fields_health_check)}")


@beartype
def open_repository() -> str:
    """Get collection path from `.ki/` directory."""
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

    return collection


@beartype
def get_latest_collection_hash() -> str:
    """Get the last collection hash stored in `.ki/hashes`."""
    kidir = os.path.join(os.getcwd(), ".ki/")
    hashes_path = os.path.join(kidir, "hashes")
    with open(hashes_path, "r", encoding="UTF-8") as hashes_file:
        return hashes_file.readlines()[-1]


@beartype
def backup(collection: str) -> None:
    """Backup collection to `.ki/backups`."""
    md5sum = md5(collection)
    backupsdir = os.path.join(os.getcwd(), ".ki/", "backups")
    assert not os.path.isfile(backupsdir)
    if not os.path.isdir(backupsdir):
        os.mkdir(backupsdir)
    backup_path = os.path.join(backupsdir, f"{md5sum}.anki2")
    assert not os.path.isfile(backup_path)
    shutil.copyfile(collection, backup_path)
    assert os.path.isfile(backup_path)


@beartype
def lock(collection: str) -> sqlite3.Connection:
    """Acquire a lock on a SQLite3 database given a path."""
    con = sqlite3.connect(collection)
    con.isolation_level = "EXCLUSIVE"
    con.execute("BEGIN EXCLUSIVE")
    return con


@beartype
def unlock(con: sqlite3.Connection) -> None:
    """Unlock a SQLite3 database."""
    con.commit()
    con.close()


@beartype
def get_fetch_head_sha(repo: git.Repo) -> str:
    """
    Get FETCH_HEAD SHA.

    Returns empty string if we've never `git fetch`-ed
    """
    try:
        return repo.rev_parse("FETCH_HEAD").binsha.hex()
    except gitdb.exc.BadName:
        initial_path = os.path.join(repo.working_dir, ".ki/", "initial")
        with open(initial_path, "r", encoding="UTF-8") as initial_file:
            sha = initial_file.read()
        logger.debug(f"SHA: {sha}")
        return sha


@beartype
def get_files_changed_since_last_fetch(repo: git.Repo) -> Iterator[str]:
    """Gets a list of paths to modified/new/deleted files since last fetch."""
    fetch_head_sha = get_fetch_head_sha(repo)

    # Treat case where there is no last fetch.
    if fetch_head_sha == "":
        dir_entries: Iterator[os.DirEntry] = os.scandir(repo.working_dir)
        paths: Iterator[str] = map(lambda entry: entry.path, dir_entries)

    else:
        diff: str = repo.git.diff(fetch_head_sha, "HEAD", name_only=True)
        paths: List[str] = diff.split("\n")

    return filter(is_anki_note, paths)


@beartype
def add_note_from_notemap(apyanki: Anki, notemap: Dict[str, Any]) -> Note:
    """Add a note given its `apy` parsed notemap."""
    model_name = notemap["model"]

    # Set current notetype for collection to `model_name`.
    model = apyanki.set_model(model_name)

    model_field_names = [field["name"] for field in model["flds"]]

    field_names = notemap["fields"].keys()
    field_values = notemap["fields"].values()

    if len(field_names) != len(model_field_names):
        click.echo(f"Error: Not enough fields for model {model_name}!")
        apyanki.modified = False
        raise click.Abort()

    for x, y in zip(model_field_names, field_names):
        if x != y:
            click.echo("Warning: Inconsistent field names " f"({x} != {y})")

    # pylint: disable=protected-access
    note = apyanki._add_note(
        field_values,
        f"{notemap['tags']}",
        notemap["markdown"],
        notemap.get("deck"),
    )

    return note
