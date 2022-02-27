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
import pprint
import shutil
import logging
import tarfile
import hashlib
import sqlite3
import tempfile
import warnings
import subprocess
import collections
import configparser
from pathlib import Path

import git
import anki
import click
import gitdb
from bs4 import MarkupResemblesLocatorWarning
from tqdm import tqdm
from loguru import logger

from lark import Lark, Transformer
from lark.lexer import Token

from apy.anki import Anki
from apy.convert import markdown_to_html, plain_to_html

from beartype import beartype
from beartype.typing import (
    List,
    Dict,
    Any,
    Iterator,
    Sequence,
    Iterable,
    Optional,
    Union,
)

from ki.note import KiNote

Header = collections.namedtuple(
    "Header", ["title", "nid", "model", "deck", "tags", "markdown"]
)
Field = collections.namedtuple("Field", ["title", "content"])
FlatNote = collections.namedtuple("Note", ["title", "nid", "model", "deck", "tags", "markdown", "fields"])

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
    colpath = Path(collection)
    targetdir = Path(directory) if directory != "" else None
    repo = _clone(colpath, targetdir, msg="Initial commit", silent=False)
    update_last_push_commit_sha(repo)


@beartype
def _clone(
    colpath: Path, targetdir: Optional[Path], msg: str, silent: bool
) -> git.Repo:
    """
    Clone an Anki collection into a directory.

    Parameters
    ----------
    colpath : pathlib.Path
        The path to a `.anki2` collection file.
    targetdir : Optional[pathlib.Path]
        A path to a directory to clone the collection into.
        Note: we check that this directory is empty.
    msg : str
        Message for initial commit.
    silent : bool
        Indicates whether we are calling `_clone()` from `pull()`.

    Returns
    -------
    git.Repo
        The cloned repository.
    """
    colpath = colpath.resolve()
    if not colpath.is_file():
        raise FileNotFoundError
    echo(f"Found .anki2 file at '{colpath}'", silent=silent)

    # Create default target directory.
    if targetdir is None:
        targetdir = Path.cwd() / colpath.stem
    if targetdir.is_dir():
        if len(set(targetdir.iterdir())) > 0:
            raise FileExistsError
    else:
        logger.debug(f"Trying to create target directory at: {targetdir}")
        targetdir.mkdir()
        assert targetdir.is_dir()

    # Create .ki subdirectory.
    kidir = targetdir / ".ki/"
    kidir.mkdir()

    # Create config file.
    config_path = kidir / "config"
    config = configparser.ConfigParser()
    config["remote"] = {"path": colpath}
    with open(config_path, "w", encoding="UTF-8") as config_file:
        config.write(config_file)

    # Append to hashes file.
    md5sum = md5(colpath)
    echo(f"Computed md5sum: {md5sum}", silent)
    append_md5sum(kidir, colpath, md5sum, silent)
    echo(f"Cloning into '{targetdir}'...", silent=silent)

    # Add `.ki/` to gitignore.
    ignore_path = targetdir / ".gitignore"
    ignore_path.write_text(".ki/\n")

    # Open deck with `apy`, and dump notes and markdown files.
    with Anki(path=colpath) as a:
        nids = list(a.col.find_notes(query=""))
        for i in tqdm(nids, ncols=TQDM_NUM_COLS, disable=silent):

            # TODO: Support multiple notes per-file.
            note = KiNote(a, a.col.getNote(i))
            note_path = targetdir / f"note{note.n.id}.md"
            note_path.write_text(str(note))

    # Initialize git repo and commit contents.
    repo = git.Repo.init(targetdir)
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
    colpath = open_repo()
    con = lock(colpath)
    md5sum = md5(colpath)

    # Quit if hash matches last pull.
    if md5sum in (Path.cwd() / ".ki/hashes").read_text().split("\n")[-1]:
        click.secho("ki pull: up to date.", bold=True)
        unlock(con)
        return

    echo(f"Pulling from '{colpath}'")
    echo(f"Computed md5sum: {md5sum}")

    # Git clone `repo` at commit SHA of last successful `push()`.
    cwd = Path.cwd()
    repo = git.Repo(cwd)
    last_push_sha = get_last_push_sha(repo)
    last_push_repo = get_ephemeral_repo(Path("ki/local"), repo, md5sum, last_push_sha)

    # Ki clone collection into an ephemeral ki repository at `anki_remote_dir`.
    msg = f"Fetch changes from DB at '{colpath}' with md5sum '{md5sum}'"
    root = Path(tempfile.mkdtemp()) / "ki" / "remote"
    root.mkdir(parents=True)
    anki_remote_dir = root / md5sum
    _clone(colpath, anki_remote_dir, msg, silent=True)

    # Create git remote pointing to anki remote repo.
    anki_remote = last_push_repo.create_remote(REMOTE_NAME, anki_remote_dir / ".git")

    # Pull anki remote repo into ``last_push_repo``.
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
    last_push_remote_path = Path(last_push_repo.working_dir) / ".git"
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
    append_md5sum(cwd / ".ki", colpath, md5sum)

    # Check that md5sum hasn't changed.
    assert md5(colpath) == md5sum
    unlock(con)


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
    colpath = open_repo()
    con = lock(colpath)
    md5sum = md5(colpath)

    # Quit if hash doesn't match last pull.
    if md5sum not in (Path.cwd() / ".ki/hashes").read_text().split("\n")[-1]:
        failed: str = f"Failed to push some refs to '{colpath}'\n{HINT}"
        click.secho(failed, fg="yellow", bold=True)
        unlock(con)
        return

    # Clone latest commit into a staging repo.
    cwd = Path.cwd()
    repo = git.Repo(cwd)
    sha = str(repo.head.commit)
    staging_repo = get_ephemeral_repo(Path("ki/local"), repo, md5sum, sha)

    # Copy `.ki/` directory into the staging repo.
    staging_repo_kidir = Path(staging_repo.working_dir) / ".ki"
    shutil.copytree(cwd / ".ki", staging_repo_kidir)

    # Get all notes changed between LAST_PUSH and HEAD.
    notepaths: Iterator[Path] = get_note_files_changed_since_last_push(staging_repo)

    # If there are no changes, update LAST_PUSH commit and quit.
    if len(set(notepaths)) == 0:
        click.secho("ki push: up to date.", bold=True)
        update_last_push_commit_sha(repo)
        return

    echo(f"Pushing to '{colpath}'")
    echo(f"Computed md5sum: {md5sum}")
    echo(f"Verified md5sum matches latest hash in '{cwd / '.ki' / 'hashes'}'")

    # Copy collection to a temp directory.
    new_colpath = Path(tempfile.mkdtemp()) / colpath.name
    assert not new_colpath.exists()
    shutil.copyfile(colpath, new_colpath)
    echo(f"Generating local .anki2 file from latest commit: {sha}")
    echo(f"Writing changes to '{new_colpath}'...")

    # Edit the copy with `apy`.
    with Anki(path=new_colpath) as a:

        # DEBUG
        nids = list(a.col.find_notes(query=""))

        # Clone repository state at commit SHA of LAST_PUSH to parse deleted notes.
        last_push_sha = get_last_push_sha(staging_repo)
        deletions_repo = get_ephemeral_repo(
            Path("ki/deleted"), repo, md5sum, last_push_sha
        )

        # Gather logging statements to display.
        log: List[str] = []

        new_nid_path_map = {}

        p = subprocess.run(
            ["git", "rev-parse", "-q", "--verify", "refs/stash"],
            check=False,
            capture_output=True,
        )
        stash_ref = p.stdout.decode()

        # Stash both unstaged and staged files (including untracked).
        repo.git.stash(include_untracked=True, keep_index=True)
        repo.git.reset("HEAD", hard=True)

        # TODO: All this logic can be abstracted away from the process of
        # actually parsing notes and constructing Anki-specific objects. This
        # is just a series of filesystem ops. They should be put in a
        # standalone function and tested without anything related to Anki.
        for notepath in tqdm(notepaths, ncols=TQDM_NUM_COLS):

            # If the file doesn't exist, parse its `nid` from its counterpart
            # in `deletions_repo`, and then delete using `apy`.
            if not notepath.is_file():
                deleted_path = Path(deletions_repo.working_dir) / notepath.name

                assert deleted_path.is_file()
                nids = get_nids(deleted_path)
                a.col.remove_notes(nids)
                a.modified = True
                continue

            # Track whether Anki reassigned any nids.
            reassigned = False
            notes: List[KiNote] = []
            new_nids = set()

            # Loop over nids and edit/add/delete notes.
            for notemap in parse_markdown_notes(notepath):
                nid = notemap["nid"]
                try:
                    note: KiNote = KiNote(a, a.col.get_note(nid))
                    update_apy_note(note, notemap)
                    notes.append(note)
                except anki.errors.NotFoundError:
                    log.append(f"Couldn't find note with nid: '{nid}'")
                    note: KiNote = add_note_from_notemap(a, notemap)
                    log.append(f"Couldn't find note with nid: '{nid}'")
                    log.append(f"Assigned new nid: '{note.n.id}'")
                    new_nids.add(note.n.id)
                    notes.append(note)
                    reassigned = True

            # If we reassigned any nids, we must regenerate the whole file.
            if reassigned:
                assert len(notes) > 0
                if len(notes) > 1:
                    logger.warning("MULTIPLE NOTES IN A SINGLE FILE!")

                # Get paths to note in local repo, as distinct from staging repo.
                note_relpath = os.path.relpath(notepath, staging_repo.working_dir)
                repo_notepath = Path(repo.working_dir) / note_relpath

                # If this is not an entirely new file, remove it.
                if repo_notepath.is_file():
                    repo_notepath.unlink()

                # Construct markdown file contents and write.
                content: str = ""
                for note in notes:
                    content += f"{str(note)}\n\n"
                first_nid = notes[0].n.id
                new_notepath = repo_notepath.parent / f"note{first_nid}.md"
                new_notepath.write_text(content)
                for nid in new_nids:
                    new_note_relpath = os.path.relpath(new_notepath, repo.working_dir)
                    new_nid_path_map[nid] = new_note_relpath

        if len(new_nid_path_map) > 0:
            msg = "Generated new nid(s).\n\n"
            for new_nid, path in new_nid_path_map.items():
                msg += f"Wrote new '{new_nid}' in file {path}\n"
            repo.git.add(all=True)
            _ = repo.index.commit(msg)

        p = subprocess.run(
            ["git", "rev-parse", "-q", "--verify", "refs/stash"],
            check=False,
            capture_output=True,
        )
        new_stash_ref = p.stdout.decode()

        # Display warnings.
        for line in log:
            click.secho(line, bold=True, fg="yellow")

        # DEBUG
        nids = list(a.col.find_notes(query=""))

    assert new_colpath.is_file()

    # Backup collection file and overwrite collection.
    backup(colpath)
    shutil.copyfile(new_colpath, colpath)
    echo(f"Overwrote '{colpath}'")

    # Append to hashes file.
    new_md5sum = md5(new_colpath)
    append_md5sum(cwd / ".ki", new_colpath, new_md5sum, silent=True)

    # Update LAST_PUSH commit SHA file and unlock DB.
    update_last_push_commit_sha(repo)
    unlock(con)


# UTILS


@beartype
def parse_markdown_notes(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Parse nids from markdown file of notes.

    Parameters
    ----------
    path : pathlib.Path
        Path to a markdown file containing `ki`-style Anki notes.

    Returns
    -------
    List[Dict[str, Any]]
        List of notemaps with the nids casted to integers.

    Raises
    ------
    KeyError
        When there's no `nid` field.
    ValueError
        When the `nid` is not coercable to an integer.
    """
    path = Path(path)

    # Support multiple notes-per-file.
    notemaps: List[Dict[str, Any]] = markdown_file_to_notes(path)
    casted_notemaps = []
    for notemap in notemaps:
        try:
            nid = int(notemap["nid"])
            notemap["nid"] = nid
            casted_notemaps.append(notemap)
        except (KeyError, ValueError) as err:
            if isinstance(err, KeyError):
                logger.warning("Failed to parse nid.")
                logger.warning(f"notemap: {notemap}")
                logger.warning(f"path: {path}")
            else:
                logger.warning("Parsed nid is not an integer.")
                logger.warning(f"notemap: {notemap}")
                logger.warning(f"path: {path}")
                logger.warning(f"nid: {notemap['nid']}")
            raise err
    return casted_notemaps


@beartype
def get_nids(path: Path) -> List[int]:
    """Get nids from a markdown file."""
    notemaps = parse_markdown_notes(path)
    return [notemap["nid"] for notemap in notemaps]


@beartype
def md5(path: Union[str, Path]) -> str:
    """Compute md5sum of file at `path`."""
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


@beartype
def is_anki_note(path: Path) -> bool:
    """Check if file is an `apy`-style markdown anki note."""
    path = str(path)

    # Ought to have markdown file extension.
    if path[-3:] != ".md":
        return False
    with open(path, "r", encoding="UTF-8") as md_file:
        lines = md_file.readlines()
    if len(lines) < 2:
        return False
    if lines[0] != "## Note\n":
        return False
    if not re.match(r"^nid: [0-9]+$", lines[1]):
        return False
    return True


@beartype
def update_apy_note(note: KiNote, notemap: Dict[str, Any]) -> None:
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
        logger.warning(f"Found duplicate note:\n {note}")
        return

    if fields_health_check:
        logger.warning(f"Found duplicate or empty note:\n {note}")
        logger.warning(f"Fields health check: {fields_health_check}")
        logger.warning(f"Fields health check (type): {type(fields_health_check)}")


@beartype
def open_repo() -> Path:
    """Get collection path from `.ki/` directory."""
    # Check that config file exists.
    config_path = Path.cwd() / ".ki/" / "config"
    if not config_path.is_file():
        raise FileNotFoundError

    # Parse config file.
    config = configparser.ConfigParser()
    config.read(config_path)
    colpath = Path(config["remote"]["path"])

    if not colpath.is_file():
        raise FileNotFoundError

    return colpath


@beartype
def backup(colpath: Path) -> None:
    """Backup collection to `.ki/backups`."""
    md5sum = md5(colpath)
    backupsdir = Path.cwd() / ".ki" / "backups"
    assert not backupsdir.is_file()
    if not backupsdir.is_dir():
        backupsdir.mkdir()
    backup_path = backupsdir / f"{md5sum}.anki2"
    if backup_path.is_file():
        click.secho("Backup already exists.", bold=True)
        return
    assert not backup_path.is_file()
    echo(f"Writing backup of .anki2 file to '{backupsdir}'")
    shutil.copyfile(colpath, backup_path)
    assert backup_path.is_file()


@beartype
def lock(colpath: os.PathLike) -> sqlite3.Connection:
    """Acquire a lock on a SQLite3 database given a path."""
    con = sqlite3.connect(colpath)
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
    last_push_path = Path(repo.working_dir) / ".ki" / "last_push"
    return last_push_path.read_text()


@beartype
def get_note_files_changed_since_last_push(repo: git.Repo) -> Sequence[Path]:
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
        paths = [Path(repo.working_dir) / file for file in files]
        paths = set(paths)

    changed = []
    for path in paths:
        if path.is_file() and not is_anki_note(path):
            continue
        changed.append(path)

    return changed


@beartype
def get_ephemeral_repo(suffix: Path, repo: git.Repo, md5sum: str, sha: str) -> git.Repo:
    """
    Clone the git repo at `repo` into an ephemeral repo.

    Parameters
    ----------
    suffix : pathlib.Path
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
    assert Path(repo.working_dir).is_dir()
    root = Path(tempfile.mkdtemp()) / suffix
    root.mkdir(parents=True)
    target = root / md5sum
    git.Repo.clone_from(repo.working_dir, target, branch=repo.active_branch)

    # Do a reset --hard to the given SHA.
    ephem: git.Repo = git.Repo(target)
    ephem.git.reset(sha, hard=True)
    return ephem


@beartype
def update_last_push_commit_sha(repo: git.Repo) -> None:
    """Dump the SHA of current HEAD commit to ``last_push file``."""
    last_push_path = Path(repo.working_dir) / ".ki" / "last_push"
    last_push_path.write_text(f"{str(repo.head.commit)}")


@beartype
def echo(string: str, silent: bool = False) -> None:
    """Call `click.secho()` with formatting."""
    if not silent:
        click.secho(string, bold=True)


@beartype
def append_md5sum(
    kidir: Path, colpath: Path, md5sum: str, silent: bool = False
) -> None:
    """Append an md5sum hash to the hashes file."""
    hashes_path = kidir / "hashes"
    with open(hashes_path, "a", encoding="UTF-8") as hashes_file:
        hashes_file.write(f"{md5sum}  {colpath.name}")
    echo(f"Wrote md5sum to '{hashes_path}'", silent)


@beartype
def add_note_from_notemap(apyanki: Anki, notemap: Dict[str, Any]) -> KiNote:
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
    note = _add_note(
        apyanki,
        field_values,
        f"{notemap['tags']}",
        notemap["markdown"],
        notemap.get("deck"),
    )

    return note


@beartype
def _add_note(
    apyanki: Anki,
    fields: Iterable[str],
    tags: str,
    markdown: bool = True,
    deck: Optional[str] = None,
) -> KiNote:
    """Add new note to collection. Apy method."""
    notetype = apyanki.col.models.current(for_deck=False)
    note = apyanki.col.new_note(notetype)

    if deck is not None:
        note.note_type()["did"] = apyanki.deck_name_to_id[deck]

    if markdown:
        note.fields = [markdown_to_html(x) for x in fields]
    else:
        note.fields = [plain_to_html(x) for x in fields]

    for tag in tags.strip().split():
        note.add_tag(tag)

    fields_health_check: int = note.fields_check()
    if fields_health_check == 0:
        apyanki.col.addNote(note)
        apyanki.modified = True
    elif fields_health_check == 2:
        logger.warning(
            f"\nDetected duplicate note while trying to add new note with nid {note.id}."
        )
        logger.warning(
            f"Note type and field values of note {note.id} exactly match an existing note."
        )
        logger.warning("Note was not added to collection!")
        logger.warning(f"First field: {list(fields)[0]}")
        logger.warning(f"Fields health check: {fields_health_check}")
    else:
        logger.error(f"Failed to add note '{note.id}'.")
        logger.error(
            f"Note failed fields health check with error code: {fields_health_check}"
        )

    return KiNote(apyanki, note)


@beartype
def markdown_file_to_notes(filename: Union[str, Path]):
    """
    Parse notes data from a Markdown file.

    The following example should adequately specify the syntax.

    ```
    # Note 1
    nid: 1
    model: Model
    tags: silly-tag
    markdown: true

    ## Front
    Question?

    ## Back
    Answer.
    ```
    """
    try:
        notes = _parse_file(filename)
    except KeyError as e:
        logger.error(f"Error {e.__class__} when parsing {filename}!")
        logger.error("Bad markdown formatting.")
        raise e

    # Ensure each note has all necessary properties.
    for note in notes:

        # Parse markdown flag.
        note["markdown"] = note["markdown"] in ("true", "yes")

        # Remove comma from tag list.
        note["tags"] = note["tags"].replace(",", "")

    return notes


@beartype
def _parse_file(filename: Union[str, Path]) -> List[Dict[str, Any]]:
    """Get data from file."""
    notes = []
    note = {}
    codeblock = False
    field = None
    with open(filename, "r", encoding="utf8") as f:
        for line in f:
            if codeblock:
                if field:
                    note["fields"][field] += line
                match = re.match(r"```\s*$", line)
                if match:
                    codeblock = False
                continue

            match = re.match(r"```\w*\s*$", line)
            if match:
                codeblock = True
                if field:
                    note["fields"][field] += line
                continue

            if not field:
                match = re.match(r"(\w+): (.*)", line)
                if match:
                    k, v = match.groups()
                    k = k.lower()
                    if k == "tag":
                        k = "tags"
                    note[k] = v.strip()
                    continue

            match = re.match(r"(#+)\s*(.*)", line)
            if not match:
                if field:
                    note["fields"][field] += line
                continue

            level, title = match.groups()

            if len(level) == 2:
                if note:
                    if field:
                        note["fields"][field] = note["fields"][field].strip()
                        notes.append(note)

                note = {"title": title, "fields": {}}
                field = None
                continue

            if len(level) == 3:
                if field:
                    note["fields"][field] = note["fields"][field].strip()

                if title in note:
                    click.echo(f"Error when parsing {filename}!")
                    raise click.Abort()

                field = title
                note["fields"][field] = ""

    if note and field:
        note["fields"][field] = note["fields"][field].strip()
        note["tags"] = ""
        notes.append(note)

    return notes




class NoteTransformer(Transformer):
    r"""
    file
      note
        header
          title     Note
          nid: 123412341234

          model: Basic

          deck      a
          tags      None
          markdown: false


        field
          fieldheader
            ###
            Front
          r


        field
          fieldheader
            ###
            Back
          s
    """
    @beartype
    def file(self, filetree: List[FlatNote]) -> List[FlatNote]:
        return filetree

    @beartype
    def note(self, n: List[Union[Header, Field]]) -> FlatNote:
        assert len(n) >= 2
        header = n[0]
        fields = n[1:]
        return FlatNote(*header, fields)

    @beartype
    def header(self, h: List[Union[str, List[str]]]) -> Header:
        return Header(*h)

    @beartype
    def title(self, t: List[str]) -> str:
        """``title: "##" TITLENAME "\n"+``"""
        assert len(t) == 1
        return t[0]

    @beartype
    def tags(self, tags: List[Optional[str]]) -> List[str]:
        return [tag for tag in tags if tag is not None]

    @beartype
    def field(self, f: List[str]) -> Field:
        assert len(f) >= 1
        fheader = f[0]
        lines = f[1:]
        content = "".join(lines)
        return Field(fheader, content)

    @beartype
    def fieldheader(self, f: List[str]) -> str:
        """``fieldheader: FIELDSENTINEL " "* ANKINAME "\n"+``"""
        assert len(f) == 2
        return f[1]

    @beartype
    def deck(self, d: List[str]) -> str:
        """``deck: "deck:" DECKNAME "\n"``"""
        assert len(d) == 1
        return d[0]

    @beartype
    def NID(self, t: Token) -> str:
        """Could be empty!"""
        nid = re.sub(r"^nid:", "", str(t)).strip()
        return nid

    @beartype
    def MODEL(self, t: Token) -> str:
        model = re.sub(r"^model:", "", str(t)).strip()
        return model

    @beartype
    def MARKDOWN(self, t: Token) -> str:
        md = re.sub(r"^markdown:", "", str(t)).strip()
        return md

    @beartype
    def DECKNAME(self, t: Token) -> str:
        return str(t).strip()

    @beartype
    def FIELDLINE(self, t: Token) -> str:
        return str(t)

    @beartype
    def TITLENAME(self, t: Token) -> str:
        return str(t)

    @beartype
    def ANKINAME(self, t: Token) -> str:
        return str(t)

    @beartype
    def TAGNAME(self, t: Token) -> str:
        return str(t)
