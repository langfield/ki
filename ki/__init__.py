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
import glob
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
import unicodedata
import configparser
from pathlib import Path
from dataclasses import dataclass

import git
import click
import gitdb
import prettyprinter as pp
from tqdm import tqdm
from loguru import logger
from pyinstrument import Profiler

from bs4 import MarkupResemblesLocatorWarning

from lark import Lark, Transformer
from lark.lexer import Token

import anki

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
    Tuple,
    Generator,
)

from ki.note import KiNote, is_generated_html

logging.basicConfig(level=logging.INFO)


LARK = True
BATCH_SIZE = 500
HTML_REGEX = r"</?\s*[a-z-][^>]*\s*>|(\&(?:[\w\d]+|#\d+|#x[a-f\d]+);)"
REMOTE_NAME = "anki"
CHANGE_TYPES = "A D R M T".split()
TQDM_NUM_COLS = 70
MAX_FIELNAME_LEN = 30
HINT = (
    "hint: Updates were rejected because the tip of your current branch is behind\n"
    + "hint: the Anki remote collection. Integrate the remote changes (e.g.\n"
    + "hint: 'ki pull ...') before pushing again."
)


@beartype
@dataclass(frozen=True)
class Field:
    """Field content pair."""

    title: str
    content: str


@beartype
@dataclass(frozen=True)
class FlatNote:
    """Flat (as possible) representation of a note."""

    title: str
    nid: int
    model: str
    deck: str
    tags: List[str]
    markdown: bool
    fields: Dict[str, str]


@beartype
@dataclass(frozen=True)
class Header:
    """Note metadata."""

    title: str
    nid: int
    model: str
    deck: str
    tags: List[str]
    markdown: bool


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

    colpath = colpath.resolve()
    if not colpath.is_file():
        echo(f"Failed: couln't find file '{colpath}'")
        return

    # Create default target directory.
    targetdir = Path(directory) if directory != "" else None
    if targetdir is None:
        targetdir = Path.cwd() / colpath.stem

    # Clean up nicely if the call fails.
    try:
        repo = _clone(colpath, targetdir, msg="Initial commit", silent=False)
        update_last_push_commit_sha(repo)
    except Exception as _err:
        echo("Failed: exiting.")
        if targetdir.is_dir():
            shutil.rmtree(targetdir)
    return


@beartype
def _clone(colpath: Path, targetdir: Path, msg: str, silent: bool) -> git.Repo:
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
        echo(f"Failed: couln't find file '{colpath}'")
        raise click.Abort()
    echo(f"Found .anki2 file at '{colpath}'", silent=silent)

    # Create default target directory.
    if targetdir is None:
        targetdir = Path.cwd() / colpath.stem
    if targetdir.is_dir():
        if len(set(targetdir.iterdir())) > 0:
            echo(f"fatal: destination path '{targetdir}' already exists and is not an empty directory.")
            raise FileExistsError
    else:
        targetdir.mkdir()

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
        all_nids = list(a.col.find_notes(query=""))

        # Create temp directory for htmlfield text files.
        root = Path(tempfile.mkdtemp()) / "ki" / "fieldhtml"
        root.mkdir(parents=True, exist_ok=True)

        # Map nids to maps of the form {fieldname -> htmlpath}.
        fieldtext_path_maps: Dict[int, Dict[str, Path]] = {}

        # Map decknames to sets of nids and nids to KiNotes.
        nidmap = {}
        kinotes = {}
        for nid in tqdm(all_nids, ncols=TQDM_NUM_COLS, disable=silent):
            kinote = KiNote(a, a.col.getNote(nid))
            assert nid == kinote.n.id

            # Check if fieldtext is HTML, and write to file if so.
            fieldtext_paths: Dict[str, Path] = {}
            for fieldname, fieldtext in kinote.fields.items():
                if re.search(HTML_REGEX, fieldtext):
                    htmlpath = root / f"{nid}{slugify(fieldname, allow_unicode=True)}"
                    htmlpath.write_text(fieldtext)
                    fieldtext_paths[fieldname] = htmlpath

            # Save paths for tidyable fields for this note.
            if len(fieldtext_paths) > 0:
                fieldtext_path_maps[nid] = fieldtext_paths

            kinotes[nid] = kinote

            nids = nidmap.get(kinote.deck, set())
            assert nid not in nids
            nids.add(nid)
            nidmap[kinote.deck] = nids

        # Spin up subprocesses for tidying field HTML in-place.
        batches = list(get_batches(glob.glob(str(root / "*")), BATCH_SIZE))
        for batch in tqdm(batches, ncols=TQDM_NUM_COLS, disable=silent):
            pathsline = " ".join(batch)
            command = f"tidy -q -m -i -omit -utf8 --tidy-mark no {pathsline}"
            subprocess.run(command, shell=True, check=False, capture_output=True)

        written = set()
        collisions = 0

        # Construct paths for each deck manifest.
        for deckname in sorted(list(nidmap.keys()), key=len, reverse=True):

            # Strip leading periods so we don't get hidden folders.
            components = deckname.split("::")
            components = [re.sub(r"^\.", r"", comp) for comp in components]
            deckpath = Path(targetdir, *components)
            deckpath.mkdir(parents=True, exist_ok=True)

            # Keep track of sort field for all seen notetypes.
            sortfmap = {}

            # Dump note payloads to FS.
            for nid in nidmap[deckname]:
                assert nid not in written
                kinote = kinotes[nid]

                # Get notetype from kinote.
                notetype: Union[Dict[str, Any], None] = kinote.n.note_type()
                assert notetype is not None

                # Get the sort field name, and cache it in a dictionary.
                ntname = notetype["name"]
                sort_fieldname = sortfmap.get(ntname, get_sort_fieldname(a, notetype))
                sortfmap[ntname] = sort_fieldname

                # Get the filename for this note.
                assert sort_fieldname in kinote.n
                field_text = kinote.n[sort_fieldname]

                # Construct filename, stripping HTML tags and sanitizing (quickly).
                field_text = plain_to_html(field_text)
                field_text = re.sub("<[^<]+?>", "", field_text)
                filename = field_text[:MAX_FIELNAME_LEN]
                filename = slugify(filename, allow_unicode=True)
                basepath = deckpath / f"{filename}"
                notepath = basepath.with_suffix(".md")

                # Construct path to note file.
                i = 1
                while notepath.exists():
                    notepath = Path(f"{basepath}_{i}").with_suffix(".md")
                    i += 1
                if i > 1:
                    collisions += 1

                # Get tidied html if it exists.
                tidyfields = {}
                if nid in fieldtext_path_maps:
                    fieldtext_paths = fieldtext_path_maps[nid]
                    for fieldname, fieldtext in kinote.fields.items():
                        if fieldname in fieldtext_paths:
                            htmlpath = fieldtext_paths[fieldname]
                            fieldtext = htmlpath.read_text()
                        tidyfields[fieldname] = fieldtext
                else:
                    tidyfields = kinote.fields

                # Construct note repr from tidyfields map.
                lines = kinote.get_header_lines()
                for fieldname, fieldtext in tidyfields.items():
                    lines.append("### " + fieldname)
                    lines.append(fieldtext)
                    lines.append("")

                # Dump payload to filesystem.
                notepath.write_text("\n".join(lines), encoding="UTF-8")
                written.add(nid)

    shutil.rmtree(root)
    logger.debug(f"Collision ratio: {collisions} / {len(written)}")

    # Initialize git repo and commit contents.
    repo = git.Repo.init(targetdir)
    repo.git.add(all=True)
    _ = repo.index.commit(msg)

    return repo


@beartype
def get_sort_fieldname(a: Anki, notetype: Dict[str, Any]) -> str:
    """Return the sort field name of a model."""
    # Get fieldmap from notetype.
    fieldmap: Dict[str, Tuple[int, Dict[str, Any]]]
    fieldmap = a.col.models.field_map(notetype)

    # Map field indices to field names.
    fieldnames: Dict[int, str] = {}
    for fieldname, (idx, _) in fieldmap.items():
        assert idx not in fieldnames
        fieldnames[idx] = fieldname

    # Get sort fieldname.
    sort_idx = a.col.models.sort_idx(notetype)
    return fieldnames[sort_idx]


@beartype
def get_batches(lst: List[str], n: int) -> Generator[str, None, None]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


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
            for flatnote in parse_markdown_notes(notepath):
                try:
                    note: KiNote = KiNote(a, a.col.get_note(flatnote.nid))
                    update_kinote(note, flatnote)
                    notes.append(note)
                except anki.errors.NotFoundError:
                    note: KiNote = add_note_from_flatnote(a, flatnote)
                    log.append(f"Reassigned nid: '{flatnote.nid}' -> '{note.n.id}'")
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
def parse_markdown_notes(path: Union[str, Path]) -> List[FlatNote]:
    """Allow for choosing which parser is used."""
    if LARK:
        return lark_parse_markdown_notes(path)
    return apy_parse_markdown_notes(path)


@beartype
def lark_parse_markdown_notes(path: Union[str, Path]) -> List[FlatNote]:
    """Parse with lark."""
    # Read grammar.
    grammar_path = Path(__file__).resolve().parent.parent / "grammar.lark"
    grammar = grammar_path.read_text(encoding="UTF-8")

    # Instantiate parser.
    parser = Lark(grammar, start="file", parser="lalr")
    transformer = NoteTransformer()
    tree = parser.parse(Path(path).read_text(encoding="UTF-8"))
    flatnotes: List[FlatNote] = transformer.transform(tree)
    return flatnotes


@beartype
def apy_parse_markdown_notes(path: Union[str, Path]) -> List[Dict[str, Any]]:
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
    flatnotes = parse_markdown_notes(path)
    return [flatnote.nid for flatnote in flatnotes]


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
def update_kinote(note: KiNote, flatnote: FlatNote) -> None:
    """Update an `apy` Note in a collection."""
    if flatnote.tags != note.n.tags:
        note.n.tags = flatnote.tags

    if flatnote.deck is not None and flatnote.deck != note.get_deck():
        note.set_deck(flatnote.deck)

    for i, value in enumerate(flatnote.fields.values()):
        if flatnote.markdown:
            note.n.fields[i] = markdown_to_html(value)
        else:
            note.n.fields[i] = plain_to_html(value)

    note.n.flush()
    note.a.modified = True
    health = note.n.fields_check()

    if health == 1:
        logger.warning(f"Found empty note:\n {note}")
        return
    if health == 2:
        logger.warning(f"Found duplicate note:\n {note}")
        return

    if health:
        logger.warning(f"Found duplicate or empty note:\n {note}")
        logger.warning(f"Fields health check: {health}")
        logger.warning(f"Fields health check (type): {type(health)}")


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
def add_note_from_flatnote(apyanki: Anki, flatnote: FlatNote) -> KiNote:
    """
    Add a note given its FlatNote representation.

    This class just validates that the fields given in the note match the notetype.
    """
    # Set current notetype for collection to `model_name`.
    model = apyanki.set_model(flatnote.model)
    model_field_names = [field["name"] for field in model["flds"]]

    if len(flatnote.fields.keys()) != len(model_field_names):
        logger.error(f"Not enough fields for model {flatnote.model}!")
        apyanki.modified = False
        raise ValueError

    for x, y in zip(model_field_names, flatnote.fields.keys()):
        if x != y:
            logger.warning("Inconsistent field names " f"({x} != {y})")

    note = _add_note(apyanki, flatnote)

    return note


@beartype
def _add_note(
    apyanki: Anki,
    flatnote: FlatNote,
) -> KiNote:
    """Add new note to collection. Apy method."""
    # Recall that we call ``apyanki.set_model(flatnote.model)`` above.
    notetype = apyanki.col.models.current(for_deck=False)
    note = apyanki.col.new_note(notetype)

    # Create new deck if deck does not exist.
    note.note_type()["did"] = apyanki.col.decks.id(flatnote.deck)

    if flatnote.markdown:
        note.fields = [markdown_to_html(x) for x in flatnote.fields.values()]
    else:
        note.fields = [plain_to_html(x) for x in flatnote.fields.values()]

    for tag in flatnote.tags:
        note.add_tag(tag)

    if note.fields_check() == 0:
        apyanki.col.addNote(note)
        apyanki.modified = True
    elif note.fields_check() == 2:
        logger.warning(f"\nFound duplicate note when adding new note w/ nid {note.id}.")
        logger.warning(f"Notetype/fields of note {note.id} match existing note.")
        logger.warning("Note was not added to collection!")
        logger.warning(f"First field: {list(flatnote.fields.values())[0]}")
        logger.warning(f"Fields health check: {note.fields_check()}")
    else:
        logger.error(f"Failed to add note '{note.id}'.")
        logger.error(f"Note failed fields check with error code: {note.fields_check()}")

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


@beartype
def slugify(value: str, allow_unicode: bool = False) -> str:
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


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
    # pylint: disable=no-self-use, missing-function-docstring

    @beartype
    def file(self, flatnotes: List[FlatNote]) -> List[FlatNote]:
        return flatnotes

    @beartype
    def note(self, n: List[Union[Header, Field]]) -> FlatNote:
        assert len(n) >= 2

        header = n[0]
        fields = n[1:]
        assert isinstance(header, Header)
        assert isinstance(fields[0], Field)

        fieldmap: Dict[str, str] = {}
        for field in fields:
            fieldmap[field.title] = field.content

        return FlatNote(
            header.title,
            header.nid,
            header.model,
            header.deck,
            header.tags,
            header.markdown,
            fieldmap,
        )

    @beartype
    def header(self, h: List[Union[str, int, bool, List[str]]]) -> Header:
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
    def NID(self, t: Token) -> int:
        """Return ``-1`` if empty."""
        nid = re.sub(r"^nid:", "", str(t)).strip()
        try:
            return int(nid)
        except ValueError:
            return -1

    @beartype
    def MODEL(self, t: Token) -> str:
        model = re.sub(r"^model:", "", str(t)).strip()
        return model

    @beartype
    def MARKDOWN(self, t: Token) -> bool:
        md = re.sub(r"^markdown:", "", str(t)).strip()
        assert md in ("true", "false")
        return md == "true"

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
