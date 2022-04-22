#!/usr/bin/env python3
"""
Python package `ki` is a command-line interface for the version control and
editing of `.anki2` collections as git repositories of markdown files.
Rather than providing an interactive UI like the Anki desktop client, `ki` aims
to allow natural editing *in the filesystem*.

In general, the purpose of `ki` is to allow users to work on large, complex
Anki decks in exactly the same way they work on large, complex software
projects.
.. include:: ./DOCUMENTATION.md
"""

# pylint: disable=invalid-name, missing-class-docstring, broad-except
# pylint: disable=too-many-return-statements, too-many-lines

import os
import re
import json
import copy
import shutil
import logging
import secrets
import sqlite3
import functools
import subprocess
import configparser
from pathlib import Path

import git
import click
import markdownify
import prettyprinter as pp
from tqdm import tqdm
from lark import Lark
from loguru import logger
from result import Result, Err, Ok, OkErr

import anki
from anki import notetypes_pb2
from anki.collection import Collection, Note, OpChangesWithId

from apy.convert import markdown_to_html, plain_to_html, html_to_markdown

from beartype import beartype
from beartype.typing import (
    Set,
    List,
    Dict,
    Any,
    Optional,
    Callable,
    Union,
)

import ki.maybes as M
import ki.functional as F
from ki.types import (
    MODELS_FILE,
    ExtantFile,
    ExtantDir,
    EmptyDir,
    NoPath,
    GitChangeType,
    Delta,
    KiRepo,
    Field,
    Template,
    Notetype,
    ColNote,
    KiRepoRef,
    RepoRef,
    Leaves,
    UpdatesRejectedError,
    TargetExistsError,
    CollectionChecksumError,
    MissingNotetypeError,
    MissingFieldOrdinalError,
    MissingNoteIdError,
    NotetypeMismatchError,
    NoteFieldValidationWarning,
    UnhealthyNoteWarning,
    UnPushedPathWarning,
    NotAnkiNoteWarning,
)
from ki.maybes import (
    GIT,
    GITIGNORE_FILE,
    GITMODULES_FILE,
    KI,
    NO_SM_DIR,
    CONFIG_FILE,
    HASHES_FILE,
    BACKUPS_DIR,
    LAST_PUSH_FILE,
)
from ki.monadic import monadic
from ki.transformer import NoteTransformer, FlatNote

logging.basicConfig(level=logging.INFO)

ChangeNotetypeInfo = notetypes_pb2.ChangeNotetypeInfo
ChangeNotetypeRequest = notetypes_pb2.ChangeNotetypeRequest
NotetypeDict = Dict[str, Any]

# Type alias for OkErr types. Subscript indicates the Ok type.
Res = List

BATCH_SIZE = 500
HTML_REGEX = r"</?\s*[a-z-][^>]*\s*>|(\&(?:[\w\d]+|#\d+|#x[a-f\d]+);)"
REMOTE_NAME = "anki"
BRANCH_NAME = "main"
CHANGE_TYPES = "A D R M T".split()
TQDM_NUM_COLS = 70
MAX_FIELNAME_LEN = 30
IGNORE = [GIT, KI, GITIGNORE_FILE, GITMODULES_FILE, MODELS_FILE]
LOCAL_SUFFIX = Path("ki/local")
STAGE_SUFFIX = Path("ki/stage")
REMOTE_SUFFIX = Path("ki/remote")
DELETED_SUFFIX = Path("ki/deleted")
FIELD_HTML_SUFFIX = Path("ki/fieldhtml")

GENERATED_HTML_SENTINEL = "data-original-markdown"

MD = ".md"


# TODO: Should catch exception and transform into nice Err that tells user what to do.
@beartype
def lock(kirepo: KiRepo) -> sqlite3.Connection:
    """Acquire a lock on a SQLite3 database given a path."""
    con = sqlite3.connect(kirepo.col_file)
    con.isolation_level = "EXCLUSIVE"
    con.execute("BEGIN EXCLUSIVE")
    return con


@beartype
def unlock(con: sqlite3.Connection) -> bool:
    """Unlock a SQLite3 database."""
    con.commit()
    con.close()
    return True


@beartype
def get_ephemeral_repo(
    suffix: Path, repo_ref: RepoRef, md5sum: str
) -> git.Repo:
    """Get a temporary copy of a git repository in /tmp/<suffix>/."""
    tempdir: EmptyDir = F.mkdtemp()
    root: EmptyDir = F.mksubdir(tempdir, suffix)

    # Git clone `repo` at latest commit in `/tmp/.../<suffix>/<md5sum>`.
    repo: git.Repo = repo_ref.repo
    branch = repo.active_branch
    target: Path = root / md5sum

    # UNSAFE: But only called here, and it should be guaranteed to work because
    # `repo` is actually a git repository, presumably there is always an
    # active branch, and `target` does not exist.
    ephem = git.Repo.clone_from(repo.working_dir, target, branch=branch, recursive=True)

    # Do a reset --hard to the given SHA.
    ephem.git.reset(repo_ref.sha, hard=True)

    return ephem


@monadic
@beartype
def get_ephemeral_kirepo(
    suffix: Path, kirepo_ref: KiRepoRef, md5sum: str
) -> Result[KiRepo, Exception]:
    """
    Given a KiRepoRef, i.e. a pair of the form (kirepo, SHA), we clone
    `kirepo.repo` into a temp directory and hard reset to the given commit
    hash.

    Parameters
    ----------
    suffix : pathlib.Path
        /tmp/.../ path suffix, e.g. `ki/local/`.
    kirepo_ref : KiRepoRef
        The ki repository to clone, and a commit for it.
    md5sum : str
        The md5sum of the associated anki collection.

    Returns
    -------
    KiRepo
        The cloned repository.
    """
    ref: RepoRef = F.kirepo_ref_to_repo_ref(kirepo_ref)
    ephem: git.Repo = get_ephemeral_repo(suffix, ref, md5sum)
    ephem_ki_dir: OkErr = M.nopath(Path(ephem.working_dir) / KI)
    if ephem_ki_dir.is_err():
        return ephem_ki_dir
    ephem_ki_dir: NoPath = ephem_ki_dir.unwrap()
    F.copytree(kirepo_ref.kirepo.ki_dir, ephem_ki_dir)
    kirepo: Res[KiRepo] = M.kirepo(F.working_dir(ephem))

    return kirepo


@beartype
def is_anki_note(path: ExtantFile) -> bool:
    """Check if file is an `apy`-style markdown anki note."""
    path = str(path)

    # Ought to have markdown file extension.
    if path[-3:] != ".md":
        return False
    with open(path, "r", encoding="UTF-8") as md_f:
        lines = md_f.readlines()
    if len(lines) < 2:
        return False
    if lines[0] != "## Note\n":
        return False
    if not re.match(r"^nid: [0-9]+$", lines[1]):
        return False
    return True


@beartype
def filter_note_path(path: Path, patterns: List[str], root: ExtantDir) -> Result[bool, Warning]:
    """
    Filter out paths in a git repository diff that do not correspond to Anki
    notes.

    We could do this purely using calls to `is_anki_note()`, but these are
    expensive, so we try to find matches without opening any files first.
    """
    # If `path` is an exact match for one of the patterns in `patterns`, we
    # immediately return a warning. Since the contents of a git repository diff
    # are always going to be files, this alone will not correctly ignore
    # directory names given in `patterns`.
    for p in patterns:
        if p == path.name:
            return Err(UnPushedPathWarning(path, p))

    # If any of the patterns in `patterns` resolve to one of the parents of
    # `path`, return a warning, so that we are able to filter out entire
    # directories.
    for ignore_path in [root / p for p in patterns]:
        parents = [path.resolve()] + [p.resolve() for p in path.parents]
        if ignore_path.resolve() in parents:
            return Err(UnPushedPathWarning(path, str(ignore_path)))

    # If `path` is an extant file (not a directory) and NOT a note, ignore it.
    logger.debug(f"Checking if note: {path}")
    logger.debug(f"Exists: {path.exists()}")
    logger.debug(f"Is file: {path.is_file()}")
    if path.exists() and path.resolve().is_file():
        file = ExtantFile(path.resolve())
        if not is_anki_note(file):
            return Err(NotAnkiNoteWarning(file))

    return Ok()


@beartype
def unsubmodule_repo(repo: git.Repo) -> None:
    """
    Un-submodule all the git submodules (convert to ordinary subdirectories and
    destroy commit history).

    MUTATES REPO in-place!

    UNSAFE: git.rm() calls.
    """
    gitmodules_path: Path = Path(repo.working_dir) / GITMODULES_FILE
    for sm in repo.submodules:

        # Untrack, remove gitmodules file, remove .git file, and add directory back.
        sm.update()

        # Guaranteed to exist by gitpython.
        sm_path = Path(sm.module().working_tree_dir)
        repo.git.rm(sm_path, cached=True)

        # May not exist.
        repo.git.rm(gitmodules_path)

        # Guaranteed to exist by gitpython, and safe because we pass
        # `missing_ok=True`, which means no error is raised.
        (sm_path / GIT).unlink(missing_ok=True)

        # Should still exist after git.rm().
        repo.git.add(sm_path)
        _ = repo.index.commit(f"Add submodule {sm.name} as ordinary directory.")

    if gitmodules_path.exists():
        repo.git.rm(gitmodules_path)
        _ = repo.index.commit("Remove '.gitmodules' file.")


@monadic
@beartype
def diff_repos(
    a_repo: git.Repo,
    b_repo: git.Repo,
    ref: RepoRef,
    filter_fn: Callable[[Path], Result[bool, Warning]],
    parser: Lark,
    transformer: NoteTransformer,
) -> Result[List[Union[Delta, Warning]], Exception]:
    # Use a `DiffIndex` to get the changed files.
    deltas = []
    a_dir = Path(a_repo.working_dir)
    b_dir = Path(b_repo.working_dir)
    logger.debug(f"{a_dir = }")
    logger.debug(f"{b_dir = }")
    logger.debug(f"Diffing {ref.sha} against {b_repo.head.commit.hexsha}")
    diff_index = b_repo.commit(ref.sha).diff(b_repo.head.commit)
    for change_type in GitChangeType:
        for diff in diff_index.iter_change_type(change_type.value):

            a_status: Result[bool, Warning] = filter_fn(a_dir / diff.a_path)
            b_status: Result[bool, Warning] = filter_fn(b_dir / diff.b_path)
            if a_status.is_err():
                logger.debug(f"{a_status = }")
                deltas.append(a_status.err())
                continue
            if b_status.is_err():
                logger.debug(f"{b_status = }")
                deltas.append(b_status.err())
                continue

            a_path = F.test(a_dir / diff.a_path)
            b_path = F.test(b_dir / diff.b_path)

            a_relpath = Path(diff.a_path)
            b_relpath = Path(diff.b_path)

            if change_type == GitChangeType.DELETED:
                if not isinstance(a_path, ExtantFile):
                    logger.warning(f"Deleted file not found in source commit: {a_path}")
                    continue

                deltas.append(Delta(change_type, a_path, a_relpath))
                continue

            if not isinstance(b_path, ExtantFile):
                logger.warning(f"Diff target not found: {b_path}")
                continue

            if change_type == GitChangeType.RENAMED:
                a_flatnote: FlatNote = parse_markdown_note(parser, transformer, a_path)
                b_flatnote: FlatNote = parse_markdown_note(parser, transformer, b_path)
                if a_flatnote.nid != b_flatnote.nid:
                    logger.debug(f"Adding delta: {change_type} {a_path} {b_path}")
                    deltas.append(Delta(GitChangeType.DELETED, a_path, a_relpath))
                    deltas.append(Delta(GitChangeType.ADDED, b_path, b_relpath))
                    continue

            logger.debug(f"Adding delta: {change_type} {b_path}")
            deltas.append(Delta(change_type, b_path, b_relpath))

    logger.debug(f"{deltas = }")
    return Ok(deltas)


@beartype
def parse_notetype_dict(nt: Dict[str, Any]) -> Result[Notetype, Exception]:
    """
    Convert an Anki NotetypeDict into a Notetype dataclass.

    Anki returns objects of type `NotetypeDict` (see pylib/anki/models.py)
    when you call a method like `col.models.all()`. This is a dictionary
    mapping strings to various stuff, and we read all its data into a python
    dataclass here so that we can access it safely. Since we don't expect Anki
    to ever give us 'invalid' notetypes (since we define 'valid' as being
    processable by Anki), we return an exception if the parse fails.

    Note on naming convention: Below, abbreviated variable names represent
    dicts coming from Anki, like `nt: NotetypeDict` or `fld: FieldDict`.
    Full words like `field: Field` represent ki dataclasses. The parameters
    of the dataclasses, however, use abbreviations for consistency with Anki
    map keys.
    """
    try:
        fields: Dict[int, Field] = {}
        for fld in nt["flds"]:
            ordinal = fld["ord"]
            fields[ordinal] = Field(name=fld["name"], ord=ordinal)

        templates: List[Template] = []
        for tmpl in nt["tmpls"]:
            templates.append(
                Template(
                    name=tmpl["name"],
                    qfmt=tmpl["qfmt"],
                    afmt=tmpl["afmt"],
                    ord=tmpl["ord"],
                )
            )

        # Guarantee that 'sortf' exists in `notetype.flds`.
        sort_ordinal: int = nt["sortf"]
        if sort_ordinal not in fields:

            # If we get a KeyError here, it will be caught.
            return Err(MissingFieldOrdinalError(sort_ordinal, nt["name"]))

        notetype = Notetype(
            id=nt["id"],
            name=nt["name"],
            type=nt["type"],
            flds=list(fields.values()),
            tmpls=templates,
            sortf=fields[sort_ordinal],
            dict=copy.deepcopy(nt),
        )

    except KeyError as err:
        return Err(err)
    return Ok(notetype)


@monadic
@beartype
def get_models_recursively(kirepo: KiRepo) -> Result[Dict[str, Notetype], Exception]:
    """
    Find and merge all `models.json` files recursively.

    Should we check for duplicates?

    Returns
    -------
    Result[Dict[int, Notetype], Exception]
        A result.Result that returns a dictionary sending model names to
        Notetypes.
    """
    all_models: Dict[str, Notetype] = {}

    # Load notetypes from json files.
    for models_file in F.rglob(kirepo.root, MODELS_FILE):
        with open(models_file, "r", encoding="UTF-8") as models_f:
            models: Dict[str, Notetype] = {}
            new_nts: Dict[int, Dict[str, Any]] = json.load(models_f)
            for _, nt in new_nts.items():
                parsed = parse_notetype_dict(nt)
                if parsed.is_err():
                    return parsed
                notetype: Notetype = parsed.ok()
                models[notetype.name] = notetype

        # Add mappings to dictionary.
        all_models.update(models)

    return Ok(all_models)


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


# TODO: This function can definitely raise a key error. It should return a
# `Result`.
@beartype
def parse_markdown_note(
    parser: Lark, transformer: NoteTransformer, notes_file: ExtantFile
) -> FlatNote:
    """Parse with lark."""
    tree = parser.parse(notes_file.read_text(encoding="UTF-8"))
    flatnotes: List[FlatNote] = transformer.transform(tree)

    # UNSAFE!
    return flatnotes[0]


@monadic
@beartype
def update_note(
    note: Note, flatnote: FlatNote, old_notetype: Notetype, new_notetype: Notetype
) -> Result[Note, Exception]:
    """
    Change all the data of `note` to that given in `flatnote`.

    This is only to be called on notes whose nid already exists in the
    database.  Creates a new deck if `flatnote.deck` doesn't exist.  Assumes
    that the model has already been added to the collection, and raises an
    exception if it finds otherwise.  Changes notetype to that specified by
    `flatnote.model`.  Overwrites all fields with `flatnote.fields`.

    Updates:
    - tags
    - deck
    - model
    - fields
    """

    # Check that the passed argument `new_notetype` has a name consistent with
    # the model specified in `flatnote`. The former should be derived from the
    # latter, and if they don't match, there is a bug in the caller.
    if flatnote.model != new_notetype.name:
        return Err(NotetypeMismatchError(flatnote, new_notetype))

    note.tags = flatnote.tags
    note.flush()

    # Set the deck of the given note, and create a deck with this name if it
    # doesn't already exist. See the comments/docstrings in the implementation.
    newdid: int = note.col.decks.id(flatnote.deck, create=True)
    cids = [c.id for c in note.cards()]

    # Set deck for all cards of this note.
    if cids:
        note.col.set_deck(cids, newdid)

    # Change notetype of note.
    fmap: Dict[str, None] = {}
    for field in old_notetype.flds:
        fmap[field.ord] = None
    note.col.models.change(old_notetype.dict, [note.id], new_notetype.dict, fmap, None)
    note.load()

    # Validate field keys against notetype.
    validated: OkErr = validate_flatnote_fields(new_notetype, flatnote)
    if validated.is_err():
        # TODO: Decide where warnings should be printed.
        logger.warning(validated.err())
        return validated

    # Set field values. This is correct because every field name that appears
    # in `new_notetype` is contained in `flatnote.fields`, or else we would
    # have printed a warning and returned above.
    # TODO: Check if these apy methods can raise exceptions.
    for key, field in flatnote.fields.items():
        if flatnote.markdown:
            note[key] = markdown_to_html(field)
        else:
            note[key] = plain_to_html(field)

    # Flush fields to collection object.
    note.flush()

    # Remove if unhealthy.
    health = display_fields_health_warning(note)
    if health != 0:
        note.col.remove_notes([note.id])
        return Err(UnhealthyNoteWarning(str(note.id)))

    return Ok(note)


@monadic
@beartype
def validate_flatnote_fields(
    notetype: Notetype, flatnote: FlatNote
) -> Result[bool, Warning]:
    """Validate that the fields given in the note match the notetype."""
    # Set current notetype for collection to `model_name`.
    field_names: List[str] = [field.name for field in notetype.flds]

    # TODO: Use a more descriptive error message.
    if len(flatnote.fields.keys()) != len(field_names):
        msg = f"Wrong number of fields for model {flatnote.model}!"
        return Err(NoteFieldValidationWarning(msg))

    for x, y in zip(field_names, flatnote.fields.keys()):
        if x != y:
            msg = f"Inconsistent field names ({x} != {y})"
            return Err(NoteFieldValidationWarning(msg))
    return Ok()


@beartype
def get_note_path(sort_field_text: str, deck_dir: ExtantDir) -> ExtantFile:
    """Get note path from sort field text."""
    field_text = sort_field_text

    # Construct filename, stripping HTML tags and sanitizing (quickly).
    field_text = plain_to_html(field_text)
    field_text = re.sub("<[^<]+?>", "", field_text)

    # If the HTML stripping removed all text, we just slugify the raw sort
    # field text.
    if len(field_text) == 0:
        field_text = sort_field_text

    name = field_text[:MAX_FIELNAME_LEN]
    slug = F.slugify(name, allow_unicode=True)

    # Make it so `slug` cannot possibly be an empty string, because then we get
    # a `Path('.')` which is a bug, and causes a runtime exception.  If all
    # else fails, generate a random hex string to use as the filename.
    if len(slug) == 0:
        slug = secrets.token_hex(10)
        logger.warning(f"Slug for {name} is empty. Using {slug} as filename")

    filename = Path(slug)
    filename = filename.with_suffix(MD)
    note_path = F.test(deck_dir / filename)

    i = 1
    while not isinstance(note_path, NoPath):
        filename = Path(f"{name}_{i}").with_suffix(MD)
        note_path = F.test(deck_dir / filename)
        i += 1

    note_path: ExtantFile = F.touch(deck_dir, str(filename))

    return note_path


@beartype
def backup(kirepo: KiRepo) -> None:
    """Backup collection to `.ki/backups`."""
    md5sum = F.md5(kirepo.col_file)
    name = f"{md5sum}.anki2"
    backup_file = F.test(kirepo.backups_dir / name)

    # We assume here that no one would ever make e.g. a directory called
    # `name`, since `name` contains the md5sum of the collection file, and
    # thus that is extraordinarily improbable. So the only thing we have to
    # check for is that we haven't already written a backup file to this
    # location.
    if isinstance(backup_file, ExtantFile):
        echo("Backup already exists.")
        return

    echo(f"Writing backup of .anki2 file to '{backup_file}'")
    F.copyfile(kirepo.col_file, kirepo.backups_dir, name)


@beartype
def append_md5sum(
    ki_dir: ExtantDir, tag: str, md5sum: str, silent: bool = False
) -> None:
    """Append an md5sum hash to the hashes file."""
    hashes_file = ki_dir / HASHES_FILE
    with open(hashes_file, "a+", encoding="UTF-8") as hashes_f:
        hashes_f.write(f"{md5sum}  {tag}\n")
    echo(f"Wrote md5sum to '{hashes_file}'", silent)


@beartype
def create_deck_dir(deck_name: str, targetdir: ExtantDir) -> ExtantDir:
    """Construct path to deck directory and create it."""
    # Strip leading periods so we don't get hidden folders.
    components = deck_name.split("::")
    components = [re.sub(r"^\.", r"", comp) for comp in components]
    deck_path = Path(targetdir, *components)
    return F.force_mkdir(deck_path)


@beartype
def get_field_note_id(nid: int, fieldname: str) -> str:
    """A str ID that uniquely identifies field-note pairs."""
    return f"{nid}{F.slugify(fieldname, allow_unicode=True)}"


@monadic
@beartype
def push_flatnote_to_anki(
    col: Collection, flatnote: FlatNote
) -> Result[ColNote, Exception]:
    model_id: Optional[int] = col.models.id_for_name(flatnote.model)
    if model_id is None:
        return Err(MissingNotetypeError(flatnote.model))

    new = False
    note: Note
    try:
        note = col.get_note(flatnote.nid)
    except anki.errors.NotFoundError:
        logger.debug(f"Failed to find '{flatnote.nid}'")
        note = col.new_note(model_id)
        col.add_note(note, col.decks.id(flatnote.deck, create=True))
        logger.debug(f"Got new nid '{note.id}'")
        new = True

    old_notetype: Res[Notetype] = parse_notetype_dict(note.note_type())
    new_notetype: Res[Notetype] = parse_notetype_dict(col.models.get(model_id))

    note: OkErr = update_note(note, flatnote, old_notetype, new_notetype)
    if note.is_err():
        return note

    note: Note = note.unwrap()
    new_notetype: Notetype = new_notetype.unwrap()

    # Get sort field content.
    try:
        sortf_text: str = note[new_notetype.sortf.name]
    except KeyError as err:
        return Err(err)

    colnote = ColNote(
        n=note,
        new=new,
        deck=flatnote.deck,
        title=flatnote.title,
        old_nid=flatnote.nid,
        markdown=flatnote.markdown,
        notetype=new_notetype,
        sortf_text=sortf_text,
    )
    return Ok(colnote)


@beartype
def get_colnote(col: Collection, nid: int) -> Result[ColNote, Exception]:
    try:
        note = col.get_note(nid)
    except anki.errors.NotFoundError:
        return Err(MissingNoteIdError(nid))
    notetype: OkErr = parse_notetype_dict(note.note_type())
    if notetype.is_err():
        return notetype
    notetype: Notetype = notetype.unwrap()

    # Get sort field content.
    try:
        sortf_text: str = note[notetype.sortf.name]
    except KeyError as err:
        return Err(err)

    # TODO: Remove implicit assumption that all cards are in the same deck, and
    # work with cards instead of notes.
    deck = col.decks.name(note.cards()[0].did)
    colnote = ColNote(
        n=note,
        new=False,
        deck=deck,
        title="",
        old_nid=note.id,
        markdown=False,
        notetype=notetype,
        sortf_text=sortf_text,
    )
    return Ok(colnote)


@beartype
def get_header_lines(colnote) -> List[str]:
    """Get header of markdown representation of note."""
    lines = [
        "## Note",
        f"nid: {colnote.n.id}",
        f"model: {colnote.notetype.name}",
    ]

    lines += [f"deck: {colnote.deck}"]
    lines += [f"tags: {', '.join(colnote.n.tags)}"]

    if not any(GENERATED_HTML_SENTINEL in field for field in colnote.n.values()):
        lines += ["markdown: false"]

    lines += [""]
    return lines


@monadic
@beartype
def write_repository(
    col_file: ExtantFile, targetdir: ExtantDir, leaves: Leaves, silent: bool
) -> Result[bool, Exception]:
    """Write notes to appropriate directories in `targetdir`."""

    # Create config file.
    config_file: ExtantFile = leaves.files[CONFIG_FILE]
    config = configparser.ConfigParser()
    config["remote"] = {"path": col_file}
    with open(config_file, "w", encoding="UTF-8") as config_f:
        config.write(config_f)

    # Create temp directory for htmlfield text files.
    tempdir: EmptyDir = F.mkdtemp()
    root: EmptyDir = F.mksubdir(tempdir, FIELD_HTML_SUFFIX)

    paths: Dict[str, ExtantFile] = {}
    decks: Dict[str, List[ColNote]] = {}

    # Open deck with `apy`, and dump notes and markdown files.
    cwd: ExtantDir = F.cwd()
    col = Collection(col_file)
    F.chdir(cwd)

    all_nids = list(col.find_notes(query=""))
    for nid in tqdm(all_nids, ncols=TQDM_NUM_COLS, disable=silent):
        colnote: OkErr = get_colnote(col, nid)
        if colnote.is_err():
            return colnote
        colnote: ColNote = colnote.unwrap()
        decks[colnote.deck] = decks.get(colnote.deck, []) + [colnote]
        for fieldname, fieldtext in colnote.n.items():
            if re.search(HTML_REGEX, fieldtext):
                fid: str = get_field_note_id(nid, fieldname)
                html_file: ExtantFile = F.touch(root, fid)
                html_file.write_text(fieldtext, encoding="UTF-8")
                paths[fid] = html_file

    # TODO: Consider adding a block in `safe()` that looks for a token
    # keyword argument, like `_err`, and bypasses the function call if it
    # is an Err. If it is an Ok, it simply removes that key-value pair from
    # `kwargs` and calls the function as it normally would.
    tidied: OkErr = tidy_html_recursively(root, silent)
    if tidied.is_err():
        return tidied
    wrote: OkErr = write_decks(col, targetdir, decks, paths)

    # Replace with frmtree.
    shutil.rmtree(root)

    return wrote


@monadic
@beartype
def write_decks(
    col: Collection,
    targetdir: ExtantDir,
    decks: Dict[str, List[ColNote]],
    paths: Dict[str, ExtantFile],
) -> Result[bool, Exception]:
    """
    There is a bug in 'write_decks()'. The sorting of deck names is done by
    length, in reverse, which means we start from the deepest, most specific
    decks, and end up at the root. I.e. We are traversing up a tree from the
    leaves to the root. Previously (see earlier commits), we simply accumulated
    all model ids and wrote the entire list (so far) to each deck's model.json
    file. But this is actually wrong, because if we have two subtrees, the one
    with larger height may have its model ids end up in the other. Since we're
    sorting by string length, it's a very imprecise, wrong way to do things.
    The proper way to do this is a DFS traversal, perhaps recursively, which
    will make it easier to keep things purely functional, accumulating the
    model ids of the children in each node. For this, we must construct a tree
    from the deck names.
    """
    # Accumulate pairs of model ids and notetype maps. The return type of the
    # `ModelManager.get()` call below indicates that it may return `None`,
    # but we know it will not because we are getting the notetype id straight
    # from the Anki DB.
    models_map: Dict[int, NotetypeDict] = {}
    for nt_name_id in col.models.all_names_and_ids():
        models_map[nt_name_id.id] = col.models.get(nt_name_id.id)

    # Dump the models file for the whole repository.
    with open(targetdir / MODELS_FILE, "w", encoding="UTF-8") as f:
        json.dump(models_map, f, ensure_ascii=False, indent=4)

    for deck_name in sorted(set(decks.keys()), key=len, reverse=True):
        deck_dir: ExtantDir = create_deck_dir(deck_name, targetdir)
        model_ids: Set[int] = set()
        deck: List[ColNote] = decks[deck_name]
        for colnote in deck:
            model_ids.add(colnote.notetype.id)
            notepath: ExtantFile = get_note_path(colnote.sortf_text, deck_dir)
            payload: str = get_note_payload(colnote, paths)
            notepath.write_text(payload, encoding="UTF-8")

        # Write `models.json` for current deck.
        deck_models_map = {mid: models_map[mid] for mid in model_ids}
        with open(deck_dir / MODELS_FILE, "w", encoding="UTF-8") as f:
            json.dump(deck_models_map, f, ensure_ascii=False, indent=4)

    return Ok()


@beartype
def html_to_screen(html: str) -> str:
    """Convert html for printing to screen."""
    html = re.sub(r"\<style\>.*\<\/style\>", "", html, flags=re.S)

    generated = GENERATED_HTML_SENTINEL in html
    if generated:
        plain = html_to_markdown(html)
        if html != markdown_to_html(plain):
            html_clean = re.sub(r' data-original-markdown="[^"]*"', "", html)
            plain += (
                "\n\n### Current HTML → Markdown\n"
                f"{markdownify.markdownify(html_clean)}"
            )
            plain += f"\n### Current HTML\n{html_clean}"
    else:
        plain = html

    # For convenience: Un-escape some common LaTeX constructs
    plain = plain.replace(r"\\\\", r"\\")
    plain = plain.replace(r"\\{", r"\{")
    plain = plain.replace(r"\\}", r"\}")
    plain = plain.replace(r"\*}", r"*}")

    plain = plain.replace(r"&lt;", "<")
    plain = plain.replace(r"&gt;", ">")
    plain = plain.replace(r"&amp;", "&")
    plain = plain.replace(r"&nbsp;", " ")

    plain = plain.replace("<br>", "\n")
    plain = plain.replace("<br/>", "\n")
    plain = plain.replace("<br />", "\n")
    plain = plain.replace("<div>", "\n")
    plain = plain.replace("</div>", "")

    # For convenience: Fix mathjax escaping (but only if the html is generated)
    if generated:
        plain = plain.replace(r"\[", r"[")
        plain = plain.replace(r"\]", r"]")
        plain = plain.replace(r"\(", r"(")
        plain = plain.replace(r"\)", r")")

    plain = re.sub(r"\<b\>\s*\<\/b\>", "", plain)
    return plain.strip()


@beartype
def get_colnote_repr(colnote: ColNote) -> str:
    lines = get_header_lines(colnote)
    for field_name, field_text in colnote.n.items():
        lines.append("### " + field_name)
        lines.append(html_to_screen(field_text))
        lines.append("")

    return "\n".join(lines)


# TODO: Come up with a better name than `paths`.
@beartype
def get_note_payload(colnote: ColNote, paths: Dict[str, ExtantFile]) -> str:
    """
    Return the markdown-converted contents of the Anki note represented by
    `colnote` as a string.

    Given a `ColNote`, which is a dataclass wrapper around a `Note` object
    which has been loaded from the DB, and a mapping from `fid`s (unique
    identifiers of field-note pairs) to paths, we check for each field of each
    note whether that field's `fid` is contained in `paths`. If so, that means
    that the caller dumped the contents of this field to a file (the file with
    this path, in fact) in order to autoformat the HTML source. If this field
    was tidied/autoformatted, we read from that path to get the tidied source,
    otherwise, we use the field content present in the `ColNote`.
    """
    # Get tidied html if it exists.
    tidyfields = {}
    for field_name, field_text in colnote.n.items():
        fid = get_field_note_id(colnote.n.id, field_name)
        if fid in paths:
            tidyfields[field_name] = paths[fid].read_text()
        else:
            tidyfields[field_name] = field_text

    # TODO: Make this use `get_colnote_repr()`.
    # Construct note repr from tidyfields map.
    lines = get_header_lines(colnote)
    for field_name, field_text in tidyfields.items():
        lines.append("### " + field_name)
        lines.append(html_to_screen(field_text))
        lines.append("")

    return "\n".join(lines)


# TODO: Refactor into a safe function.
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


@beartype
def echo(string: str, silent: bool = False) -> None:
    """Call `click.secho()` with formatting."""
    if not silent:
        click.secho(string, bold=True)
        # logger.info(string)


@monadic
@beartype
def tidy_html_recursively(root: ExtantDir, silent: bool) -> Result[bool, Exception]:
    """Call html5-tidy on each file in `root`, editing in-place."""
    # Spin up subprocesses for tidying field HTML in-place.
    batches: List[List[ExtantFile]] = list(
        F.get_batches(F.rglob(root, "*"), BATCH_SIZE)
    )
    for batch in tqdm(batches, ncols=TQDM_NUM_COLS, disable=silent):

        # Fail silently here, so as to not bother user with tidy warnings.
        command = ["tidy", "-q", "-m", "-i", "-omit", "-utf8", "--tidy-mark", "no"]
        command += batch
        try:
            subprocess.run(command, check=False, capture_output=True)
        except Exception as err:
            return Err(err)
    return Ok()


@monadic
@beartype
def flatten_staging_repo(
    stage_kirepo: KiRepo, kirepo: KiRepo
) -> Result[KiRepo, Exception]:
    """
    Convert the staging repository into a format that is amenable to taking
    diffs across all files in all submodules.

    To do this, we first convert all submodules into ordinary subdirectories of
    the git repository. Then we replace the dot git directory of the staging
    repo with the .git directory of the repo in `.ki/no_submodules_tree/`,
    which, as its name suggests, is a copy of the main repository with all its
    submodules converted into directories.

    This is done in order to preserve the history of
    `.ki/no_submodules_tree/`. The staging repository can be thought of as
    the next commit to this repo.

    We return a reloaded version of the staging repository, re-read from disk.
    """
    unsubmodule_repo(stage_kirepo.repo)

    # Shutil rmtree the stage repo .git directory.
    stage_git_dir: NoPath = F.rmtree(F.git_dir(stage_kirepo.repo))
    stage_root: ExtantDir = stage_kirepo.root
    del stage_kirepo

    # Copy the .git folder from `no_submodules_tree` into the stage repo.
    stage_git_dir = F.copytree(F.git_dir(kirepo.no_modules_repo), stage_git_dir)
    stage_root: ExtantDir = F.parent(stage_git_dir)

    # Reload stage kirepo.
    stage_kirepo: Res[KiRepo] = M.kirepo(stage_root)

    return stage_kirepo


@monadic
@beartype
def get_target(
    cwd: ExtantDir, col_file: ExtantFile, directory: str
) -> Result[EmptyDir, Exception]:
    # Create default target directory.
    path = F.test(Path(directory) if directory != "" else cwd / col_file.stem)
    if isinstance(path, NoPath):
        path.mkdir(parents=True)
        return M.emptydir(path)
    if isinstance(path, EmptyDir):
        return Ok(path)
    return Err(TargetExistsError(path))


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
def clone(collection: str, directory: str = "") -> Result[bool, Exception]:
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
    echo("Cloning.")
    col_file: Res[ExtantFile] = M.xfile(Path(collection))

    cwd: ExtantDir = F.cwd()
    targetdir: Res[EmptyDir] = get_target(cwd, col_file, directory)
    md5sum: Res[str] = _clone(col_file, targetdir, msg="Initial commit", silent=False)

    # Check that we are inside a ki repository, and get the associated collection.
    targetdir: Res[ExtantDir] = M.xdir(targetdir)
    kirepo: Res[KiRepo] = M.kirepo(targetdir)

    # Get reference to HEAD of current repo.
    head: Res[KiRepoRef] = M.head_kirepo_ref(kirepo)

    if targetdir.is_err() or head.is_err():
        echo("Failed: exiting.")

        # We get an error here only in the case where the `M.xdir()` call
        # failed on `targetdir`. We cannot assume that it doesn't exist,
        # because we may have returned the exception inside `fmkdempty()`,
        # which errors-out when the target already exists and is nonempty. This
        # means we definitely do not want to remove `targetdir` or its
        # contents in this case, because we would be deleting the user's data.
        if targetdir.is_err():
            return targetdir

        # Otherwise, we must have that either we created `targetdir` and it did
        # not exist prior, or it was an empty directory before. In either case,
        # we can probably remove it safely.
        # TODO: Consider removing only its contents instead.
        targetdir: ExtantDir = targetdir.unwrap()
        if targetdir.is_dir():
            shutil.rmtree(targetdir)
        return head

    head: KiRepoRef = head.unwrap()

    # Get staging repository in temp directory, and copy to `no_submodules_tree`.

    # Copy current kirepo into a temp directory (the STAGE), hard reset to HEAD.
    stage_kirepo: Res[KiRepo] = get_ephemeral_kirepo(STAGE_SUFFIX, head, md5sum)
    stage_kirepo = flatten_staging_repo(stage_kirepo, kirepo)
    if stage_kirepo.is_err():
        echo("Failed: exiting.")
        return stage_kirepo
    stage_kirepo: KiRepo = stage_kirepo.unwrap()

    stage_kirepo.repo.git.add(all=True)
    stage_kirepo.repo.index.commit(f"Pull changes from ref {head.sha}")

    # Completely annihilate the `.ki/no_submodules_tree`
    # directory/repository, and replace it with `stage_kirepo`. This is a
    # sensible operation because earlier, we copied the `.git/` directory
    # from `.ki/no_submodules_tree` to the staging repo. So the history is
    # preserved.
    no_modules_root: NoPath = F.rmtree(F.working_dir(head.kirepo.no_modules_repo))
    F.copytree(stage_kirepo.root, no_modules_root)

    kirepo: Res[KiRepo] = M.kirepo(targetdir)
    if kirepo.is_err():
        echo("Failed: exiting.")
        return kirepo
    kirepo: KiRepo = kirepo.unwrap()

    # Dump HEAD ref of current repo in `.ki/last_push`.
    kirepo.last_push_file.write_text(head.sha)

    return Ok()


@monadic
@beartype
def _clone(
    col_file: ExtantFile, targetdir: EmptyDir, msg: str, silent: bool
) -> Result[str, Exception]:
    """
    Clone an Anki collection into a directory.

    The caller, realistically only `clone()`, expects that `targetdir` will
    be the root of a valid ki repository after this function is called, so we
    need to do our repo initialization with gitpython in here, as opposed to in
    `clone()`.

    Parameters
    ----------
    col_file : pathlib.Path
        The path to a `.anki2` collection file.
    targetdir : pathlib.Path
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
    echo(f"Found .anki2 file at '{col_file}'", silent=silent)

    # Create .ki subdirectory.
    ki_dir: EmptyDir = F.mksubdir(targetdir, Path(KI))

    # Populate the .ki subdirectory with empty metadata files.
    leaves: Res[Leaves] = F.fmkleaves(
        ki_dir,
        files={CONFIG_FILE: CONFIG_FILE, LAST_PUSH_FILE: LAST_PUSH_FILE},
        dirs={BACKUPS_DIR: BACKUPS_DIR, NO_SM_DIR: NO_SM_DIR},
    )

    md5sum = F.md5(col_file)
    echo(f"Computed md5sum: {md5sum}", silent)
    echo(f"Cloning into '{targetdir}'...", silent=silent)

    # Add `.ki/` to gitignore.
    ignore_path = targetdir / GITIGNORE_FILE
    ignore_path.write_text(".ki/\n")

    # Write notes to disk. We do explicit error checking here because if we
    # don't the repository initialization will run even when there's a failure.
    # This would be very bad for speed, because gitpython calls have quite a
    # bit of overhead sometimes (although maybe not for `Repo.init()` calls,
    # since they aren't networked).
    wrote: OkErr = write_repository(col_file, targetdir, leaves, silent)
    if wrote.is_err():
        return wrote

    initialized: OkErr = init_repos(targetdir, leaves, msg)
    if initialized.is_err():
        return initialized

    # Store the md5sum of the anki collection file in the hashes file (we
    # always append, never overwrite).
    append_md5sum(ki_dir, col_file.name, md5sum, silent)

    return Ok(md5sum)


# TODO: Remove this function.
@monadic
@beartype
def init_repos(
    targetdir: ExtantDir, leaves: Leaves, msg: str
) -> Result[bool, Exception]:
    """
    Initialize both git repos and commit contents of the main one.
    """
    repo = git.Repo.init(targetdir, initial_branch=BRANCH_NAME)
    repo.git.add(all=True)
    _ = repo.index.commit(msg)

    # Initialize the copy of the repository with submodules replaced with
    # subdirectories that lives in `.ki/no_submodules_tree/`.
    _ = git.Repo.init(leaves.dirs[NO_SM_DIR], initial_branch=BRANCH_NAME)

    return Ok()


@ki.command()
@beartype
def pull() -> Result[bool, Exception]:
    """
    Pull from a preconfigured remote Anki collection into an existing ki
    repository.
    """

    # Check that we are inside a ki repository, and get the associated collection.
    cwd: ExtantDir = F.cwd()
    kirepo: Res[KiRepo] = M.kirepo(cwd)
    if kirepo.is_err():
        echo(str(kirepo.err()))
        return kirepo
    kirepo: KiRepo = kirepo.unwrap()
    con: sqlite3.Connection = lock(kirepo)

    md5sum: str = F.md5(kirepo.col_file)
    hashes: List[str] = kirepo.hashes_file.read_text().split("\n")
    hashes = list(filter(lambda l: l != "", hashes))
    logger.debug(f"Hashes:\n{pp.pformat(hashes)}")
    if md5sum in hashes[-1]:
        echo("ki pull: up to date.")
        return Ok(unlock(con))

    echo(f"Pulling from '{kirepo.col_file}'")
    echo(f"Computed md5sum: {md5sum}")

    # Git clone `repo` at commit SHA of last successful `push()`.
    sha: str = kirepo.last_push_file.read_text()
    ref: Res[RepoRef] = M.repo_ref(kirepo.repo, sha)
    if ref.is_err():
        echo(str(ref.err()))
    ref: RepoRef = ref.unwrap()
    last_push_repo: git.Repo = get_ephemeral_repo(LOCAL_SUFFIX, ref, md5sum)

    # Ki clone collection into an ephemeral ki repository at `anki_remote_root`.
    msg = f"Fetch changes from DB at '{kirepo.col_file}' with md5sum '{md5sum}'"
    anki_remote_root: EmptyDir = F.mksubdir(F.mkdtemp(), REMOTE_SUFFIX / md5sum)

    # This should return the repository as well.
    cloned: OkErr = _clone(kirepo.col_file, anki_remote_root, msg, silent=True)
    if cloned.is_err():
        return echo(str(cloned.err()))
    pulled: OkErr = pull_changes_from_remote_repo(
        kirepo, anki_remote_root, last_push_repo, md5sum
    )
    if pulled.is_err():
        return echo(str(pulled.err()))
    return Ok(unlock(con))


# TODO: Remove this function.
@monadic
@beartype
def pull_theirs_from_remote(
    repo: git.Repo, root: ExtantDir, remote: git.Remote
) -> Result[bool, Exception]:
    cwd: ExtantDir = F.chdir(root)
    echo(f"Pulling into {root}")
    repo.git.config("pull.rebase", "false")
    # TODO: This is not yet safe. Consider rewriting with gitpython.
    git_subprocess_pull(REMOTE_NAME, BRANCH_NAME)
    repo.delete_remote(remote)
    F.chdir(cwd)
    return Ok()


@monadic
@beartype
def pull_changes_from_remote_repo(
    kirepo: KiRepo,
    anki_remote_root: ExtantDir,
    last_push_repo: git.Repo,
    md5sum: str,
) -> Result[bool, Exception]:
    """
    Load the git repository at `anki_remote_root`, force pull (preferring
    'theirs', i.e. the new stuff from the sqlite3 database) changes from that
    repository (which is cloned straight from the collection, which in general
    may have new changes) into `last_push_repo`, and then pull
    `last_push_repo` into the main repository.

    We pull in this sequence in order to avoid merge conflicts. Since we first
    pull into a snapshot of the repository as it looked when we last pushed to
    the database, we know that there cannot be any merge conflicts, because to
    git, it just looks like we haven't made any changes since then. Then we
    pull the result of that merge into our actual repository. So there could
    still be merge conflicts at that point, but they will only be 'genuine'
    merge conflicts in some sense, because as a result of using this snapshot
    strategy, we give the anki collection the appearance of being a persistent
    remote git repo. If we didn't do this, the fact that we did a fresh clone
    of the database every time would mean that everything would look like a
    merge conflict, because there is no shared history.
    """
    remote_repo: Res[git.Repo] = M.repo(anki_remote_root)
    if remote_repo.is_err():
        return remote_repo
    remote_repo: git.Repo = remote_repo.unwrap()

    # Create git remote pointing to anki remote repo.
    anki_remote = last_push_repo.create_remote(REMOTE_NAME, remote_repo.git_dir)

    # Pull anki remote repo into `last_push_repo`.
    last_push_root: Res[ExtantDir] = F.working_dir(last_push_repo)
    pulled: OkErr = pull_theirs_from_remote(last_push_repo, last_push_root, anki_remote)
    if pulled.is_err():
        return pulled

    # Create remote pointing to `last_push` repo and pull into `repo`.
    last_push_remote = kirepo.repo.create_remote(REMOTE_NAME, last_push_repo.git_dir)
    kirepo.repo.git.config("pull.rebase", "false")
    p = subprocess.run(
        ["git", "pull", "-v", REMOTE_NAME, BRANCH_NAME],
        check=False,
        capture_output=True,
    )
    click.secho(f"{p.stdout.decode()}", bold=True)
    click.secho(f"{p.stderr.decode()}", bold=True)
    kirepo.repo.delete_remote(last_push_remote)

    # Append to hashes file.
    append_md5sum(kirepo.ki_dir, kirepo.col_file.name, md5sum, silent=True)

    # Check that md5sum hasn't changed.
    if F.md5(kirepo.col_file) != md5sum:
        return Err(CollectionChecksumError(kirepo.col_file))
    return Ok()


# PUSH


@ki.command()
@beartype
def push() -> Result[bool, Exception]:
    """Push a ki repository into a .anki2 file."""
    pp.install_extras(exclude=["ipython", "django", "ipython_repr_pretty"])

    # Check that we are inside a ki repository, and get the associated collection.
    cwd: ExtantDir = F.cwd()
    kirepo: Res[KiRepo] = M.kirepo(cwd)
    if kirepo.is_err():
        return kirepo
    kirepo: KiRepo = kirepo.unwrap()
    con: sqlite3.Connection = lock(kirepo)

    md5sum: str = F.md5(kirepo.col_file)
    hashes: List[str] = kirepo.hashes_file.read_text().split("\n")
    hashes = list(filter(lambda l: l != "", hashes))
    logger.debug(f"Hashes:\n{pp.pformat(hashes)}")
    if md5sum not in hashes[-1]:
        return Err(UpdatesRejectedError(kirepo.col_file))

    # Get reference to HEAD of current repo.
    head: Res[KiRepoRef] = M.head_kirepo_ref(kirepo)
    if head.is_err():
        return head
    head: KiRepoRef = head.unwrap()

    # Copy current kirepo into a temp directory (the STAGE), hard reset to HEAD.
    stage_kirepo: OkErr = get_ephemeral_kirepo(STAGE_SUFFIX, head, md5sum)
    stage_kirepo = flatten_staging_repo(stage_kirepo, kirepo)
    if stage_kirepo.is_err():
        return stage_kirepo
    stage_kirepo: KiRepo = stage_kirepo.unwrap()

    # This statement cannot be any farther down because we must get a reference
    # to HEAD *before* we commit, and then after the following line, the
    # reference we got will be HEAD~1, hence the variable name.
    head_1: Res[RepoRef] = M.head_repo_ref(stage_kirepo.repo)
    if head_1.is_err():
        return head_1
    head_1: RepoRef = head_1.unwrap()

    stage_kirepo.repo.git.add(all=True)
    stage_kirepo.repo.index.commit(f"Pull changes from ref {head.sha}")

    # Get filter function.
    filter_fn = functools.partial(filter_note_path, patterns=IGNORE, root=kirepo.root)

    head_kirepo: Res[KiRepo] = get_ephemeral_kirepo(LOCAL_SUFFIX, head, md5sum)

    # Read grammar.
    # TODO:! Should we assume this always exists? A nice error message should
    # be printed on initialization if the grammar file is missing. No
    # computation should be done, and none of the click commands should work.
    grammar_path = Path(__file__).resolve().parent / "grammar.lark"
    grammar = grammar_path.read_text(encoding="UTF-8")

    # Instantiate parser.
    parser = Lark(grammar, start="file", parser="lalr")
    transformer = NoteTransformer()

    # Get deltas.
    a_repo: git.Repo = get_ephemeral_repo(DELETED_SUFFIX, head_1, md5sum)
    b_repo: git.Repo = head_1.repo
    deltas: OkErr = diff_repos(a_repo, b_repo, head_1, filter_fn, parser, transformer)

    # Map model names to models.
    models: Res[Dict[str, Notetype]] = get_models_recursively(head_kirepo)

    return push_deltas(
        deltas, models, kirepo, md5sum, parser, transformer, stage_kirepo, con
    )


@monadic
@beartype
def push_deltas(
    deltas: List[Union[Delta, Warning]],
    models: Dict[str, Notetype],
    kirepo: KiRepo,
    md5sum: str,
    parser: Lark,
    transformer: NoteTransformer,
    stage_kirepo: KiRepo,
    con: sqlite3.Connection,
) -> Result[bool, Exception]:
    warnings: List[Warning] = [delta for delta in deltas if isinstance(delta, Warning)]
    deltas: List[Delta] = [delta for delta in deltas if isinstance(delta, Delta)]
    logger.debug(f"Warnings: {warnings}")

    # If there are no changes, quit.
    if len(set(deltas)) == 0:
        echo("ki push: up to date.")
        return Ok()

    echo(f"Pushing to '{kirepo.col_file}'")
    echo(f"Computed md5sum: {md5sum}")
    echo(f"Verified md5sum matches latest hash in '{kirepo.hashes_file}'")

    # Copy collection to a temp directory.
    temp_col_dir: ExtantDir = F.mkdtemp()
    new_col_file = temp_col_dir / kirepo.col_file.name
    col_name: str = kirepo.col_file.name
    new_col_file: ExtantFile = F.copyfile(kirepo.col_file, temp_col_dir, col_name)

    head: Res[RepoRef] = M.head_repo_ref(kirepo.repo)
    if head.is_err():
        echo("Failed: no commits in repository. Couldn't find HEAD ref.")
        return Ok()
    head = head.unwrap()
    echo(f"Generating local .anki2 file from latest commit: {head.sha}")
    echo(f"Writing changes to '{new_col_file}'...")

    cwd: ExtantDir = F.cwd()
    col = Collection(new_col_file)
    F.chdir(cwd)
    modified = True

    # Add all new models.
    for model in models.values():

        # TODO: Consider waiting to parse `models` until after the
        # `add_dict()` call.
        if col.models.id_for_name(model.name) is not None:

            nt_copy: NotetypeDict = copy.deepcopy(model.dict)
            nt_copy["id"] = 0
            changes: OpChangesWithId = col.models.add_dict(nt_copy)
            nt: NotetypeDict = col.models.get(changes.id)
            model: OkErr = parse_notetype_dict(nt)
            if model.is_err():
                return model

    # Gather logging statements to display.
    log: List[str] = []

    # Stash both unstaged and staged files (including untracked).
    kirepo.repo.git.stash(include_untracked=True, keep_index=True)
    kirepo.repo.git.reset("HEAD", hard=True)

    is_delete = lambda d: d.status == GitChangeType.DELETED
    deletes: List[Delta] = list(filter(is_delete, deltas))
    logger.debug(f"Deleting {len(deletes)} notes.")

    for delta in tqdm(deltas, ncols=TQDM_NUM_COLS):

        # Parse the file at `delta.path` into a `FlatNote`, and
        # add/edit/delete in collection.
        flatnote = parse_markdown_note(parser, transformer, delta.path)
        logger.debug(f"Resolving delta:\n{pp.pformat(delta)}\n{pp.pformat(flatnote)}")

        if is_delete(delta):
            logger.debug(f"Deleting note {flatnote.nid}")
            col.remove_notes([flatnote.nid])
            continue

        # If a note with this nid exists in DB, update it.
        # TODO: If relevant prefix of sort field has changed, we regenerate
        # the file. Recall that the sort field is used to determine the
        # filename. If the content of the sort field has changed, then we
        # may need to update the filename.
        colnote: OkErr = push_flatnote_to_anki(col, flatnote)
        regenerated: OkErr = regenerate_note_file(colnote, kirepo.root, delta.relpath)
        if regenerated.is_err():
            regen_err: Exception = regenerated.unwrap_err()
            if isinstance(regen_err, Warning):
                continue
            return regenerated
        log += regenerated.unwrap()

    # Display all warnings returned by the filter function.
    for warning in warnings:
        echo(str(warning))

    # Commit nid reassignments.
    logger.warning(f"Reassigned {len(log)} nids.")
    if len(log) > 0:
        msg = "Generated new nid(s).\n\n" + "\n".join(log)

        # Commit in all submodules (doesn't support recursing yet).
        for sm in kirepo.repo.submodules:

            # TODO: Remove submodule update calls, and use the gitpython
            # API to check if the submodules exist instead. The update
            # calls make a remote fetch which takes an extremely long time,
            # and the user should have run `git submodule update`
            # themselves anyway.
            subrepo: git.Repo = sm.update().module()
            subrepo.git.add(all=True)
            subrepo.index.commit(msg)

        # Commit in main repository.
        kirepo.repo.git.add(all=True)
        _ = kirepo.repo.index.commit(msg)

    if modified:
        echo("Database was modified.")
        col.close()
    elif col.db:
        col.close(False)

    # Backup collection file and overwrite collection.
    backup(kirepo)
    new_col_file = F.copyfile(new_col_file, F.parent(kirepo.col_file), col_name)
    echo(f"Overwrote '{kirepo.col_file}'")

    # Append to hashes file.
    new_md5sum = F.md5(new_col_file)
    append_md5sum(kirepo.ki_dir, new_col_file.name, new_md5sum, silent=False)

    # Completely annihilate the `.ki/no_submodules_tree`
    # directory/repository, and replace it with `stage_kirepo`. This is a
    # sensible operation because earlier, we copied the `.git/` directory
    # from `.ki/no_submodules_tree` to the staging repo. So the history is
    # preserved.
    no_modules_root: NoPath = F.rmtree(F.working_dir(kirepo.no_modules_repo))
    F.copytree(stage_kirepo.root, no_modules_root)

    # Dump HEAD ref of current repo in `.ki/last_push`.
    kirepo.last_push_file.write_text(head.sha)

    # Unlock Anki SQLite DB.
    unlock(con)

    return Ok()


@monadic
@beartype
def regenerate_note_file(
    colnote: ColNote, root: ExtantDir, relpath: Path
) -> Result[List[str], Exception]:
    """
    Construct the contents of a note corresponding to the arguments `colnote`,
    which itself was created from `flatnote`, and then write it to disk.

    Returns a list of lines to add to the commit message (either an empty list,
    or a list containing a single line).

    This function is intended to be used when we are adding *completely* new
    notes, in which case the caller generates a new note with a newly generated
    nid, which can be accessed at `colnote.n.id`. In general, this is the
    branch taken whenever Anki fails to recognize the nid given in the note
    file, which can be accessed via `colnote.old_nid`. Thus, we only regenerate
    if `colnote.n.id` and `colnote.old_nid` to differ, since the former has the
    newly assigned nid for this note, yielded by the Anki runtime, and the
    latter has whatever was written in the file.
    """
    # If this is not a new note, then we didn't reassign its nid, and we don't
    # need to regenerate the file. So we don't add a line to the commit
    # message.
    if not colnote.new:
        return Ok([])

    # Get paths to note in local repo, as distinct from staging repo.
    repo_note_path: Path = root / relpath

    # If this is not an entirely new file, remove it.
    if repo_note_path.is_file():
        repo_note_path.unlink()

    parent: ExtantDir = F.force_mkdir(repo_note_path.parent)
    new_note_path: ExtantFile = get_note_path(colnote.sortf_text, parent)
    new_note_path.write_text(get_colnote_repr(colnote), encoding="UTF-8")

    # TODO: Figure out if this still works if we use pathlib instead.
    new_note_relpath = os.path.relpath(new_note_path, root)

    msg = f"Reassigned nid: '{colnote.old_nid}' -> '{colnote.n.id}' in '{new_note_relpath}'"
    return Ok([msg])
