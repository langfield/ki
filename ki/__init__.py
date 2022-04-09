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

# pylint: disable=unnecessary-pass, too-many-lines, invalid-name

import os
import re
import json
import glob
import pprint
import shutil
import logging
import tarfile
import hashlib
import sqlite3
import tempfile
import warnings
import itertools
import functools
import subprocess
import collections
import unicodedata
import configparser
from enum import Enum
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
    Set,
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

import ki.architecture as ARCH
from ki.note import KiNote
from ki.transformer import NoteTransformer, FlatNote

logging.basicConfig(level=logging.INFO)

FieldDict = Dict[str, Any]
NotetypeDict = Dict[str, Any]
TemplateDict = Dict[str, Union[str, int, None]]

BATCH_SIZE = 500
HTML_REGEX = r"</?\s*[a-z-][^>]*\s*>|(\&(?:[\w\d]+|#\d+|#x[a-f\d]+);)"
REMOTE_NAME = "anki"
BRANCH_NAME = "main"
CHANGE_TYPES = "A D R M T".split()
TQDM_NUM_COLS = 70
MAX_FIELNAME_LEN = 30
HINT = (
    "hint: Updates were rejected because the tip of your current branch is behind\n"
    + "hint: the Anki remote collection. Integrate the remote changes (e.g.\n"
    + "hint: 'ki pull ...') before pushing again."
)
IGNORE = [".git", ".ki", ".gitignore", ".gitmodules", "models.json"]
MODELS_FILENAME = "models.json"


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
        md5sum = _clone(colpath, targetdir, msg="Initial commit", silent=False)

        # ARCH STUFF. SHOULD NOT BE HERE, BECAUSE SHOULD NOT RUN DURING pull() CALLS.

        # Check that we are inside a ki repository, and get the associated collection.
        root: ARCH.ExtantDir = ARCH.ftest(targetdir)
        kirepo: ARCH.KiRepo = ARCH.IO(ARCH.get_ki_repo(root))

        # Get reference to HEAD of current repo.
        head: ARCH.KiRepoRef = ARCH.IO(ARCH.MaybeHeadKiRepoRef(kirepo))

        # Get staging repository in temp directory, and copy to ``no_submodules_tree``.
        stage_kirepo: ARCH.KiRepo = ARCH.get_stage_repo(head, md5sum)
        stage_kirepo.repo.git.add(all=True)
        stage_kirepo.repo.index.commit(f"Pull changes from ref {head.sha}")
        ARCH.update_no_submodules_tree(kirepo, stage_kirepo.root)

    # pylint: disable=broad-except
    except Exception as err:
        echo(str(err))
        echo("Failed: exiting.")
        if targetdir.is_dir() and not isinstance(err, FileExistsError):
            shutil.rmtree(targetdir)
    return


@beartype
def tidy_html_recursively(root: Path, silent: bool) -> None:
    """Call html5-tidy on each file in ``root``, editing in-place."""
    # Spin up subprocesses for tidying field HTML in-place.
    batches = list(get_batches(glob.glob(str(root / "*")), BATCH_SIZE))
    for batch in tqdm(batches, ncols=TQDM_NUM_COLS, disable=silent):

        # Fail silently here, so as to not bother user with tidy warnings.
        command = ["tidy", "-q", "-m", "-i", "-omit", "-utf8", "--tidy-mark", "no"]
        command += batch
        subprocess.run(command, check=False, capture_output=True)


@beartype
def _clone(colpath: Path, targetdir: Path, msg: str, silent: bool) -> str:
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
        raise FileNotFoundError
    echo(f"Found .anki2 file at '{colpath}'", silent=silent)

    # Create default target directory.
    if targetdir.is_dir():
        if len(set(targetdir.iterdir())) > 0:
            echo(
                f"fatal: destination path '{targetdir}' already exists "
                "and is not an empty directory."
            )
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
    ARCH.append_md5sum(kidir, colpath, md5sum, silent)
    echo(f"Cloning into '{targetdir}'...", silent=silent)

    # Add `.ki/` to gitignore.
    ignore_path = targetdir / ".gitignore"
    ignore_path.write_text(".ki/\n")

    # Write notes to disk.
    write_notes(colpath, targetdir, silent)

    # Initialize git repo and commit contents.
    repo = git.Repo.init(targetdir, initial_branch=BRANCH_NAME)
    repo.git.add(all=True)
    _ = repo.index.commit(msg)

    return md5sum


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
    root = find_repo_root()
    os.chdir(root)
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
    temproot = Path(tempfile.mkdtemp()) / "ki" / "remote"
    temproot.mkdir(parents=True)
    anki_remote_dir = temproot / md5sum
    _clone(colpath, anki_remote_dir, msg, silent=True)

    # Create git remote pointing to anki remote repo.
    anki_remote = last_push_repo.create_remote(REMOTE_NAME, anki_remote_dir / ".git")

    # Pull anki remote repo into ``last_push_repo``.
    os.chdir(last_push_repo.working_dir)
    logger.debug(f"Pulling into {last_push_repo.working_dir}")
    last_push_repo.git.config("pull.rebase", "false")
    git_subprocess_pull(REMOTE_NAME, BRANCH_NAME)
    last_push_repo.delete_remote(anki_remote)

    # Create remote pointing to ``last_push`` repo and pull into ``repo``.
    os.chdir(cwd)
    last_push_remote_path = Path(last_push_repo.working_dir) / ".git"
    last_push_remote = repo.create_remote(REMOTE_NAME, last_push_remote_path)
    repo.git.config("pull.rebase", "false")
    p = subprocess.run(
        ["git", "pull", "-v", REMOTE_NAME, BRANCH_NAME],
        check=False,
        capture_output=True,
    )
    click.secho(f"{p.stdout.decode()}", bold=True)
    click.secho(f"{p.stderr.decode()}", bold=True)
    repo.delete_remote(last_push_remote)

    # Append to hashes file.
    ARCH.append_md5sum(cwd / ".ki", colpath, md5sum)

    # Check that md5sum hasn't changed.
    assert md5(colpath) == md5sum
    unlock(con)


@beartype
def get_models_recursively(root: Path) -> Dict[int, NotetypeDict]:
    """Find and merge all ``models.json`` files recursively."""
    all_new_models: Dict[int, NotetypeDict] = {}
    for models_path in root.rglob(MODELS_FILENAME):

        # Load notetypes from json file.
        with open(models_path, "r", encoding="UTF-8") as models_file:
            new_models: Dict[int, NotetypeDict] = json.load(models_file)

        # Add mappings to dictionary.
        all_new_models.update(new_models)

    return all_new_models


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
    ARCH._push()


# UTILS


@beartype
def get_sort_fieldname(a: Anki, notetype: Dict[str, Any]) -> str:
    """Return the sort field name of a model."""
    assert notetype is not None

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


@beartype
def parse_markdown_notes(path: Union[str, Path]) -> List[FlatNote]:
    """Parse with lark."""
    # Read grammar.
    grammar_path = Path(__file__).resolve().parent / "grammar.lark"
    grammar = grammar_path.read_text(encoding="UTF-8")

    # Instantiate parser.
    parser = Lark(grammar, start="file", parser="lalr")
    transformer = NoteTransformer()
    tree = parser.parse(Path(path).read_text(encoding="UTF-8"))
    flatnotes: List[FlatNote] = transformer.transform(tree)
    return flatnotes


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
def update_kinote(kinote: KiNote, flatnote: FlatNote) -> None:
    """
    Update an `apy` Note in a collection.

    Currently fails if the model has changed. It does not update the model name
    of the KiNote, and it also assumes that the field names have not changed.
    """
    kinote.n.tags = flatnote.tags

    # TODO: Can this raise an error? What if the deck is not in the collection?
    # Should we create it? Probably.
    kinote.set_deck(flatnote.deck)

    # Get new notetype from collection (None if nonexistent).
    notetype: Optional[NotetypeDict] = kinote.n.col.models.by_name(flatnote.model)

    # If notetype doesn't exist, raise an error.
    if notetype is None:
        msg = f"Notetype '{flatnote.model}' doesn't exist. "
        msg += "Create it in Anki before adding notes via ki."
        raise FileNotFoundError(msg)

    # Validate field keys against notetype.
    old_model = kinote.n.note_type()
    validate_flatnote_fields(old_model, flatnote)

    # Set field values.
    for key, field in flatnote.fields.items():
        if flatnote.markdown:
            kinote.n[key] = markdown_to_html(field)
        else:
            kinote.n[key] = plain_to_html(field)

    # Flush note, mark collection as modified, and display any warnings.
    kinote.n.flush()
    kinote.a.modified = True
    display_fields_health_warning(kinote.n)


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
def find_repo_root() -> Path:
    """
    Find the root of the current ki repository, raising an error if it does not
    exist.
    """
    current = Path.cwd()
    while not (current / ".ki").is_dir() and current.resolve() != Path("/"):
        current = current.parent.resolve()
    if current.resolve() == Path("/"):
        msg = "fatal: not a ki repository (or any parent up to mount point /)\n"
        msg += "Stopping at filesystem boundary."
        raise FileNotFoundError(msg)
    return current


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
    try:
        return last_push_path.read_text()
    except FileNotFoundError:
        logger.warning("Couldn't find '.ki/last_push' file!")
        return ""


@beartype
def path_ignore_fn(path: Path, patterns: List[str], repo: git.Repo) -> bool:
    """Lambda to be used as first argument to filter(). Filters out paths-to-ignore."""
    for p in patterns:
        if p == path.name:
            return False

    # Ignore files that match a pattern in ``patterns`` ('*' not supported).
    for ignore_path in [Path(repo.working_dir) / p for p in patterns]:
        parents = [path.resolve()] + [p.resolve() for p in path.parents]
        if ignore_path.resolve() in parents:
            return False

    # If ``path`` is an extant file (not a directory) and NOT a note, ignore it.
    if path.exists() and not path.resolve().is_dir() and not is_anki_note(path):
        return False

    # DEBUG
    if ".json" in str(path):
        raise ValueError(f"Bad path: {path}")
    return True


class GitChangeType(Enum):
    """Enum for git file change types."""

    ADDED = "A"
    DELETED = "D"
    RENAMED = "R"
    MODIFIED = "M"
    TYPECHANGED = "T"


@beartype
@dataclass(frozen=True)
class Delta:
    """The git delta for a single file."""

    status: GitChangeType
    path: Path


# TODO: This must also check for changes in ``models.json``.
@beartype
def get_note_files_changed_since_last_push(repo: git.Repo) -> Sequence[Delta]:
    """Gets a list of paths to modified/new/deleted note md files since last push."""
    deltas: Iterator[Delta]
    last_push_sha = get_last_push_sha(repo)
    ignore_fn = functools.partial(path_ignore_fn, patterns=IGNORE, repo=repo)

    # Treat case where there is no last push.
    if last_push_sha == "":
        p = subprocess.run(
            ["find", ".", "-type", "f"],
            check=False,
            capture_output=True,
        )
        paths = filter(ignore_fn, map(Path, p.stdout.decode().split()))
        deltas = [Delta(GitChangeType.ADDED, path) for path in paths]

    else:
        # Use a `DiffIndex` to get the changed files.
        deltas = set()
        hcommit = repo.head.commit
        last_push_commit = repo.commit(last_push_sha)

        logger.debug(f"Working dir: {repo.working_dir}")
        out = repo.git.diff(f"{last_push_sha}", submodule="diff")
        logger.debug(f"Submodule diff output:\n{out}")

        # Checkout last push commit.
        repo.git.checkout(last_push_sha)

        # Look for all submodules that existed at last push commit.
        sm_paths = get_submodule_paths(repo)
        logger.debug(f"Submodule paths at last push:\n{pp.pformat(sm_paths)}")

        # Checkout head.
        repo.git.checkout(hcommit.hexsha)

        # Remove submodules (which makes commits).
        unsubmodule_repo(repo)

        # Update head commit variable.
        hcommit = repo.head.commit

        diff_index = last_push_commit.diff(hcommit)
        for change_type in GitChangeType:
            for diff in diff_index.iter_change_type(change_type.value):
                a_path = Path(repo.working_dir) / diff.a_path
                b_path = Path(repo.working_dir) / diff.b_path
                logger.debug(f"{a_path = }")
                logger.debug(f"{b_path = }")

                if not ignore_fn(a_path) or not ignore_fn(b_path):
                    continue

                if change_type == GitChangeType.RENAMED:
                    deltas.add(Delta(GitChangeType.DELETED, a_path))
                    deltas.add(Delta(GitChangeType.RENAMED, b_path))
                else:
                    deltas.add(Delta(change_type, b_path))

    return list(deltas)


@beartype
def get_ephemeral_repo(suffix: Path, repo: git.Repo, md5sum: str, sha: str) -> git.Repo:
    """
    Clone ``repo`` at ``sha`` into an ephemeral repo.

    Parameters
    ----------
    suffix : pathlib.Path
        /tmp/.../ path suffix, e.g. ``ki/local/``.
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
    git.Repo.clone_from(
        repo.working_dir, target, branch=repo.active_branch, recursive=True
    )

    # Do a reset --hard to the given SHA.
    ephem: git.Repo = git.Repo(target)
    ephem.git.reset(sha, hard=True)

    return ephem


@beartype
def get_submodule_paths(repo: git.Repo) -> List[Path]:
    """
    Return a list of submodule paths.

    MUTATES ``repo`` in-place! (Calls ``sm.update()``.)
    """
    paths = []
    for sm in repo.submodules:
        sm.update()
        assert sm.module_exists(), f"Module {sm.name} doesn't exist :("
        sm_path = Path(sm.module().working_tree_dir)
        paths.append(sm_path)

    return paths


@beartype
def unsubmodule_repo(repo: git.Repo) -> None:
    """
    Un-submodule all the git submodules (convert to ordinary subdirectories and
    destroy commit history).

    MUTATES REPO in-place!
    """
    gitmodules_path = Path(repo.working_dir) / ".gitmodules"
    for sm in repo.submodules:
        sm.update()
        assert sm.module_exists(), f"Module {sm.name} doesn't exist :("

        # Untrack, remove gitmodules file, remove .git file, and add directory back.
        sm_path = Path(sm.module().working_tree_dir)
        repo.git.rm(sm_path, cached=True)
        repo.git.rm(gitmodules_path)
        (sm_path / ".git").unlink()
        repo.git.add(sm_path)
        _ = repo.index.commit(f"Add submodule {sm.name} as ordinary directory.")

    if gitmodules_path.exists():
        repo.git.rm(gitmodules_path)
        _ = repo.index.commit("Remove '.gitmodules' file.")

    assert not gitmodules_path.exists()


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
def validate_flatnote_fields(model: NotetypeDict, flatnote: FlatNote) -> None:
    """Validate that the fields given in the note match the notetype."""
    # Set current notetype for collection to `model_name`.
    model_field_names = [field["name"] for field in model["flds"]]

    if len(flatnote.fields.keys()) != len(model_field_names):
        logger.error(f"Not enough fields for model {flatnote.model}!")
        raise ValueError

    for x, y in zip(model_field_names, flatnote.fields.keys()):
        if x != y:
            logger.error("Inconsistent field names " f"({x} != {y})")
            raise ValueError


@beartype
def add_note_from_flatnote(a: Anki, flatnote: FlatNote) -> Optional[KiNote]:
    """Add a note given its FlatNote representation, provided it passes health check."""
    # TODO: Does this assume model exists?
    model = a.set_model(flatnote.model)
    validate_flatnote_fields(model, flatnote)

    # Note that we call ``a.set_model(flatnote.model)`` above, so the current
    # model is the model given in ``flatnote``.
    notetype = a.col.models.current(for_deck=False)
    note = a.col.new_note(notetype)

    # Create new deck if deck does not exist.
    note.note_type()["did"] = a.col.decks.id(flatnote.deck)

    if flatnote.markdown:
        note.fields = [markdown_to_html(x) for x in flatnote.fields.values()]
    else:
        note.fields = [plain_to_html(x) for x in flatnote.fields.values()]

    for tag in flatnote.tags:
        note.add_tag(tag)

    health = display_fields_health_warning(note)

    result = None
    if health == 0:
        a.col.addNote(note)
        a.modified = True
        result = KiNote(a, note)

    return result


@beartype
def display_fields_health_warning(note: anki.notes.Note) -> int:
    """Display warnings when Anki's fields health check fails."""
    health = note.fields_check()
    if health == 1:
        logger.warning(f"Found empty note:\n {note}")
        logger.warning(f"Fields health check code: {health}")
    elif health == 2:
        logger.warning(f"\nFound duplicate note when adding new note w/ nid {note.id}.")
        logger.warning(f"Notetype/fields of note {note.id} match existing note.")
        logger.warning("Note was not added to collection!")
        logger.warning(f"First field: {note.fields[0]}")
        logger.warning(f"Fields health check code: {health}")
    elif health != 0:
        logger.error(f"Failed to process note '{note.id}'.")
        logger.error(f"Note failed fields check with unknown error code: {health}")
    return health


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


@beartype
def write_notes(colpath: Path, targetdir: Path, silent: bool):
    """Write notes to appropriate directories in ``targetdir``."""
    # Create temp directory for htmlfield text files.
    root = Path(tempfile.mkdtemp()) / "ki" / "fieldhtml"
    root.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {}
    decks: Dict[str, List[KiNote]] = {}

    # Open deck with `apy`, and dump notes and markdown files.
    with Anki(path=colpath) as a:
        all_nids = list(a.col.find_notes(query=""))
        for nid in tqdm(all_nids, ncols=TQDM_NUM_COLS, disable=silent):
            kinote = KiNote(a, a.col.get_note(nid))
            decks[kinote.deck] = decks.get(kinote.deck, []) + [kinote]
            for fieldname, fieldtext in kinote.fields.items():
                if re.search(HTML_REGEX, fieldtext):
                    fid = get_field_note_id(nid, fieldname)
                    paths[fid] = root / fid
                    paths[fid].write_text(fieldtext)

        tidy_html_recursively(root, silent)

        # Write models to disk.
        models_map: Dict[int, NotetypeDict] = {}
        for model in a.col.models.all():
            model_id = a.col.models.id_for_name(model["name"])
            assert model_id is not None
            models_map[model_id] = model

        with open(targetdir / MODELS_FILENAME, "w", encoding="UTF-8") as f:
            json.dump(models_map, f, ensure_ascii=False, indent=4)

        deck_model_ids: Set[int] = set()
        for deckname in sorted(set(decks.keys()), key=len, reverse=True):
            deckpath = create_deckpath(deckname, targetdir)
            for kinote in decks[deckname]:

                # Add the notetype id.
                model = kinote.n.note_type()
                model_id = a.col.models.id_for_name(model["name"])
                deck_model_ids.add(model_id)

                sort_fieldname = get_sort_fieldname(a, model)
                notepath = get_notepath(kinote, sort_fieldname, deckpath)
                payload = get_tidy_payload(kinote, paths)
                notepath.write_text(payload, encoding="UTF-8")

            # Write ``models.json`` for current deck.
            deck_models_map = {mid: models_map[mid] for mid in deck_model_ids}
            with open(deckpath / MODELS_FILENAME, "w", encoding="UTF-8") as f:
                json.dump(deck_models_map, f, ensure_ascii=False, indent=4)

    shutil.rmtree(root)


@beartype
def create_deckpath(deckname: str, targetdir: Path) -> Path:
    """Construct path to deck directory and create it."""
    # Strip leading periods so we don't get hidden folders.
    components = deckname.split("::")
    components = [re.sub(r"^\.", r"", comp) for comp in components]
    deckpath = Path(targetdir, *components)
    deckpath.mkdir(parents=True, exist_ok=True)
    return deckpath


@beartype
def get_field_note_id(nid: int, fieldname: str) -> str:
    """A str ID that uniquely identifies field-note pairs."""
    return f"{nid}{slugify(fieldname, allow_unicode=True)}"


@beartype
def get_tidy_payload(kinote: KiNote, paths: Dict[str, Path]) -> str:
    """Get the payload for the note (HTML-tidied if necessary)."""
    # Get tidied html if it exists.
    tidyfields = {}
    for fieldname, fieldtext in kinote.fields.items():
        fid = get_field_note_id(kinote.n.id, fieldname)
        if fid in paths:
            tidyfields[fieldname] = paths[fid].read_text()
        else:
            tidyfields[fieldname] = fieldtext

    # Construct note repr from tidyfields map.
    lines = kinote.get_header_lines()
    for fieldname, fieldtext in tidyfields.items():
        lines.append("### " + fieldname)
        lines.append(fieldtext)
        lines.append("")

    # Dump payload to filesystem.
    return "\n".join(lines)


@beartype
def get_notepath(kinote: KiNote, sort_fieldname: str, deckpath: Path) -> Path:
    """Get notepath from sort field name."""
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

    i = 1
    while notepath.exists():
        notepath = Path(f"{basepath}_{i}").with_suffix(".md")
        i += 1

    return notepath


@beartype
def git_subprocess_pull(remote: str, branch: str) -> int:
    """Pull remote into branch using a subprocess call."""
    p = subprocess.run(
        [
            "git",
            "pull",
            "-v",
            "--allow-unrelated-histories",
            "--strategy-option",
            "theirs",
            remote,
            branch,
        ],
        check=False,
        capture_output=True,
    )
    pull_stderr = p.stderr.decode()
    logger.debug(f"\n{pull_stderr}")
    logger.debug(f"Return code: {p.returncode}")
    if p.returncode != 0:
        raise ValueError(pull_stderr)
    return p.returncode
