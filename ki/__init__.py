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
import warnings
import subprocess
import configparser

import git
import anki
import click
import gitdb
from bs4 import MarkupResemblesLocatorWarning
from tqdm import tqdm
from loguru import logger

from apy.anki import Anki, Note
from apy.convert import markdown_to_html, plain_to_html, markdown_file_to_notes

from beartype import beartype
from beartype.typing import List, Dict, Any, Iterator, Sequence

from ki.note import KiNote


logging.basicConfig(level=logging.INFO)


TQDM_NUM_COLS = 70
CHANGE_TYPES = "A D R M T".split()
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
    warnings.filterwarnings(action="ignore", category=MarkupResemblesLocatorWarning)
    repo = _clone(collection, directory, msg="Initial commit", silent=False)
    update_last_push_commit_sha(repo)


@beartype
def echo(string: str, silent: bool = False) -> None:
    """Call `click.secho()` with formatting."""
    if not silent:
        click.secho(string, bold=True)


@beartype
def _clone(collection: str, directory: str, msg: str, silent: bool) -> git.Repo:
    """
    Clone an Anki collection into a directory.

    Parameters
    ----------
    collection : str
        The path to a `.anki2` collection file.
    directory : str
        A path to a directory to clone the collection into.
        Note: we check that this directory does not yet exist.
    msg : str
        Message for initial commit.
    silent : bool
        Indicates whether we are calling `_clone()` from `pull()`.

    Returns
    -------
    git.Repo
        The cloned repository.
    """
    collection = os.path.abspath(collection)
    if not os.path.isfile(collection):
        raise FileNotFoundError
    echo(f"Found .anki2 file at '{collection}'", silent=silent)

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
    md5sum = md5(collection)
    echo(f"Computed md5sum: {md5sum}")
    basename = os.path.basename(collection)
    hashes_path = os.path.join(kidir, "hashes")
    with open(hashes_path, "a", encoding="UTF-8") as hashes_file:
        hashes_file.write(f"{md5sum}  {basename}")
    echo(f"Wrote md5sum to '{hashes_path}'", silent=silent)
    echo(f"Cloning into '{directory}'...", silent=silent)

    # Add `.ki/` to gitignore.
    ignore_path = os.path.join(directory, ".gitignore")
    with open(ignore_path, "w", encoding="UTF-8") as ignore_file:
        ignore_file.write(".ki/\n")

    # Open deck with `apy`, and dump notes and markdown files.
    query = ""
    with Anki(path=collection) as a:
        nids = list(a.col.find_notes(query))
        for i in tqdm(nids, ncols=TQDM_NUM_COLS, disable=silent):
            note = KiNote(a, a.col.getNote(i))
            note_path = os.path.join(directory, f"note{note.n.id}.md")
            with open(note_path, "w", encoding="UTF-8") as note_file:
                note_file.write(str(note))

    # Initialize git repo and commit contents.
    repo = git.Repo.init(directory)
    repo.git.add(all=True)
    _ = repo.index.commit(msg)

    return repo


@ki.command()
@beartype
def pull() -> None:
    """
    Pull from a preconfigured remote Anki collection into an existing ki
    repository.
    """
    # Suppress `bs4` warnings.
    warnings.filterwarnings(action="ignore", category=MarkupResemblesLocatorWarning)

    # Lock DB and get hash.
    collection = open_repo()
    con = lock(collection)
    md5sum = md5(collection)

    # Quit if hash matches last pull.
    if md5sum in get_latest_collection_hash():
        click.secho("ki pull: up to date.", bold=True)
        unlock(con)
        return

    # Clone `repo` into an ephemeral repo at commit SHA of last successful `push()`.
    cwd = os.getcwd()
    repo = git.Repo(cwd)
    last_push_sha = get_last_push_sha(repo)
    last_push_repo = get_ephemeral_repo("ki/local/", repo, md5sum, last_push_sha)

    # Ki clone into another ephemeral repo and unlock DB.
    msg = f"Fetch changes from DB at '{collection}' with md5sum '{md5sum}'"
    root = os.path.join(tempfile.mkdtemp(), "ki/", "remote/")
    os.makedirs(root)
    anki_remote_dir = os.path.join(root, md5sum)
    _clone(collection, anki_remote_dir, msg, silent=True)
    unlock(con)

    # Create remote pointing to anki repo.
    anki_remote_path = os.path.join(anki_remote_dir, ".git")
    anki_remote = last_push_repo.create_remote(REMOTE_NAME, anki_remote_path)

    # Pull anki remote ephemeral repo into ``last_push_repo``.
    os.chdir(last_push_repo.working_dir)
    last_push_repo.git.config("pull.rebase", "false")
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
        check=True,
        capture_output=True,
    )
    last_push_repo.delete_remote(anki_remote)

    # Create remote pointing to ``last_push`` repo and pull into ``repo``.
    os.chdir(cwd)
    last_push_remote_path = os.path.join(last_push_repo.working_dir, ".git")
    last_push_remote = repo.create_remote(REMOTE_NAME, last_push_remote_path)
    repo.git.config("pull.rebase", "false")
    p = subprocess.run(
        ["git", "pull", "-v", REMOTE_NAME, "main"],
        check=False,
        capture_output=True,
    )
    click.secho(f"{p.stdout.decode()}", bold=True)
    click.secho(f"{p.stderr.decode()}", bold=True)
    repo.delete_remote(last_push_remote)

    # Append to hashes file.
    basename = os.path.basename(collection)
    kidir = os.path.join(os.getcwd(), ".ki/")
    hashes_path = os.path.join(kidir, "hashes")
    with open(hashes_path, "a", encoding="UTF-8") as hashes_file:
        hashes_file.write(f"{md5sum}  {basename}")

    # Check that md5sum hasn't changed.
    assert md5(collection) == md5sum


@ki.command()
@beartype
def push() -> None:
    """
    Pack a ki repository into a .anki2 file and push to collection location.

    1. Clone the repository at the latest commit in a staging repo.
    2. Get all notes that have changed since the last successful push.
    3. Clone repository at SHA of last successful push to get nids of deleted files.
    4. Add/edit/delete notes using `apy`.
    """
    # Lock DB, get path to collection, and compute hash.
    collection = open_repo()
    con = lock(collection)
    md5sum = md5(collection)

    # Quit if hash doesn't match last pull.
    if md5sum not in get_latest_collection_hash():
        failed: str = f"Failed to push some refs to '{collection}'\n{HINT}"
        click.secho(failed, fg="yellow", bold=True)
        unlock(con)
        return

    cwd = os.getcwd()
    repo = git.Repo(cwd)
    sha = str(repo.head.commit)
    staging_repo = get_ephemeral_repo("ki/local/", repo, md5sum, sha)

    # Copy `.ki/` directory into the staging repo.
    repo_kidir = os.path.join(cwd, ".ki/")
    staging_repo_kidir = os.path.join(staging_repo.working_dir, ".ki/")
    shutil.copytree(repo_kidir, staging_repo_kidir)

    # Get path to new collection.
    new_collection = os.path.join(tempfile.mkdtemp(), os.path.basename(collection))
    assert not os.path.isfile(new_collection)
    assert not os.path.isdir(new_collection)

    # Get all changed notes in checked-out staging repo.
    notepaths: Iterator[str] = get_note_files_changed_since_last_push(staging_repo)

    # If there are no changes, update LAST_PUSH commit and quit.
    if len(set(notepaths)) == 0:
        click.secho("ki push: up to date.", bold=True)
        update_last_push_commit_sha(repo)
        return

    click.secho("ki push: nontrivial push.", bold=True)

    # Copy collection to new collection and modify in-place.
    shutil.copyfile(collection, new_collection)
    with Anki(path=new_collection) as a:

        last_push_sha = get_last_push_sha(staging_repo)
        deletions_repo = get_ephemeral_repo("ki/deleted/", repo, md5sum, last_push_sha)

        for notepath in tqdm(notepaths, ncols=TQDM_NUM_COLS):

            # If the file doesn't exist, parse its `nid` from its counterpart
            # in `deletions_repo`, and then delete using `apy`.
            if not os.path.isfile(notepath):
                deleted_file = os.path.basename(notepath)
                deleted_path = os.path.join(deletions_repo.working_dir, deleted_file)

                assert os.path.isfile(deleted_path)
                nids = get_nids(deleted_path)
                a.delete_notes(nids)
                continue

            # Loop over nids and update/add notes.
            for notemap in parse_markdown_notes(notepath):
                nid = notemap["nid"]
                try:
                    note: Note = Note(a, a.col.get_note(nid))
                    update_apy_note(note, notemap)
                except anki.errors.NotFoundError:
                    note: Note = add_note_from_notemap(a, notemap)
                    logger.warning(f"Couldn't find note with nid: '{nid}'")
                    logger.warning(f"Assigned new nid: '{note.n.id}'")

    assert os.path.isfile(new_collection)

    # Backup collection file, overwrite collection, and unlock DB.
    backup(collection)
    shutil.copyfile(new_collection, collection)
    unlock(con)

    # Append to hashes file.
    new_md5sum = md5(new_collection)
    basename = os.path.basename(collection)
    kidir = os.path.join(os.getcwd(), ".ki/")
    hashes_path = os.path.join(kidir, "hashes")
    with open(hashes_path, "a", encoding="UTF-8") as hashes_file:
        hashes_file.write(f"{new_md5sum}  {basename}")

    # Update LAST_PUSH commit SHA file.
    update_last_push_commit_sha(repo)


# UTILS


@beartype
def parse_markdown_notes(path: str) -> List[Dict[str, Any]]:
    """Parse nids from markdown file of notes."""
    # Support multiple notes-per-file.
    notemaps: List[Dict[str, Any]] = markdown_file_to_notes(path)
    casted_notemaps = []
    for notemap in notemaps:
        try:
            nid = int(notemap["nid"])
            notemap["nid"] = nid
            casted_notemaps.append(notemap)
        except KeyError as err:
            logger.error("Failed to parse nid.")
            logger.error(f"notemap: {notemap}")
            logger.error(f"path: {path}")
            raise err
    return casted_notemaps


@beartype
def get_nids(path: str) -> List[int]:
    """Get just nids from a markdown note."""
    notemaps = parse_markdown_notes(path)
    return [notemap["nid"] for notemap in notemaps]


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
def open_repo() -> str:
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
    if os.path.isfile(backup_path):
        click.secho("Backup already exists.", bold=True)
        return
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
def get_last_push_sha(repo: git.Repo) -> str:
    """Get LAST_PUSH SHA."""
    last_push_path = os.path.join(repo.working_dir, ".ki/", "last_push")
    with open(last_push_path, "r", encoding="UTF-8") as last_push_file:
        sha = last_push_file.read()
    return sha


@beartype
def get_note_files_changed_since_last_push(repo: git.Repo) -> Sequence[str]:
    """Gets a list of paths to modified/new/deleted note md files since last push."""
    paths: Iterator[str]
    last_push_sha = get_last_push_sha(repo)

    # Treat case where there is no last push.
    if last_push_sha == "":
        dir_entries: Iterator[os.DirEntry] = os.scandir(repo.working_dir)
        paths = map(lambda entry: entry.path, dir_entries)

    else:
        # Use a `DiffIndex` to get the changed files.
        files = []
        hcommit = repo.head.commit
        diff_index = hcommit.diff(last_push_sha)
        for change_type in CHANGE_TYPES:
            for diff in diff_index.iter_change_type(change_type):
                files.append(diff.a_path)
                files.append(diff.b_path)
        paths = [os.path.join(repo.working_dir, file) for file in files]

    changed = []
    for path in paths:
        if os.path.isfile(path) and not is_anki_note(path):
            continue
        changed.append(path)

    return changed


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


@beartype
def get_ephemeral_repo(suffix: str, repo: git.Repo, md5sum: str, sha: str) -> git.Repo:
    """
    Clone the git repo at `repo` into an ephemeral repo.

    Parameters
    ----------
    suffix : str
        /tmp/.../ path suffix, e.g. `ki/local/`.
    repo : git.Repo
        The git repository to clone.
    md5sum : str
        The md5sum of the associated anki collection.
    sha : str
        The commit SHA to reset --hard to.

    Returns
    -------
    git.Repo
        The cloned repository.
    """
    # Git clone `repo` at latest commit in `/tmp/.../<suffix>/<md5sum>`.
    assert os.path.isdir(repo.working_dir)
    root = os.path.join(tempfile.mkdtemp(), suffix)
    os.makedirs(root)
    target = os.path.join(root, md5sum)
    git.Repo.clone_from(repo.working_dir, target, branch=repo.active_branch)

    # Do a reset --hard to the given SHA.
    ephem: git.Repo = git.Repo(target)
    ephem.git.reset(sha, hard=True)
    return ephem


@beartype
def update_last_push_commit_sha(repo: git.Repo) -> None:
    """Dump the SHA of current HEAD commit to ``last_push file``."""
    last_push_path = os.path.join(repo.working_dir, ".ki/", "last_push")
    with open(last_push_path, "w", encoding="UTF-8") as last_push_file:
        last_push_file.write(f"{str(repo.head.commit)}")
