"""
Ki is a command-line interface for the version control and editing of `.anki2`
collections as git repositories of markdown files.  Rather than providing an
interactive UI like the Anki desktop client, ki aims to allow natural editing
*in the filesystem*.

In general, the purpose of ki is to allow users to work on large, complex Anki
decks in exactly the same way they work on large, complex software projects.
.. include:: ./DOCUMENTATION.md
"""

# pylint: disable=invalid-name, missing-class-docstring, broad-except
# pylint: disable=too-many-return-statements, too-many-lines

import os
import re
import io
import gc
import sys
import json
import copy
import shutil
import random
import logging
import secrets
import sqlite3
import hashlib
import traceback
import functools
import subprocess
import dataclasses
import configparser
from pathlib import Path
from contextlib import redirect_stdout
from dataclasses import dataclass

import git
import click
import whatthepatch
import prettyprinter as pp
from tqdm import tqdm
from halo import Halo
from lark import Lark
from loguru import logger

# Required to avoid circular imports because the Anki pylib codebase is gross.
import anki.collection
from anki.cards import Card, CardId, TemplateDict
from anki.decks import DeckTreeNode
from anki.utils import ids2str
from anki.models import ChangeNotetypeInfo, ChangeNotetypeRequest, NotetypeDict
from anki.errors import NotFoundError
from anki.exporting import AnkiExporter
from anki.collection import Collection, Note, OpChangesWithId

from beartype import beartype
from beartype.typing import (
    Set,
    List,
    Dict,
    Any,
    Optional,
    Callable,
    Union,
    TypeVar,
    Sequence,
    Tuple,
)

import ki.maybes as M
import ki.functional as F
from ki.types import (
    MODELS_FILE,
    ExtantFile,
    ExtantDir,
    EmptyDir,
    NoPath,
    NoFile,
    Symlink,
    GitChangeType,
    Patch,
    Delta,
    KiRepo,
    Field,
    Template,
    Notetype,
    ColNote,
    KiRepoRef,
    RepoRef,
    Leaves,
    NoteDBRow,
    DeckNote,
    PushResult,
    WrittenNoteFile,
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
    DeletedFileNotFoundWarning,
    DiffTargetFileNotFoundWarning,
    NotetypeKeyError,
    UnnamedNotetypeError,
    SQLiteLockError,
    NoteFieldKeyError,
    MissingMediaDirectoryError,
    MissingMediaFileWarning,
    ExpectedNonexistentPathError,
    WrongFieldCountWarning,
    InconsistentFieldNamesWarning,
    MissingTidyExecutableError,
)
from ki.maybes import (
    GIT,
    GITIGNORE_FILE,
    GITMODULES_FILE,
    KI,
    CONFIG_FILE,
    HASHES_FILE,
    BACKUPS_DIR,
    LAST_PUSH_FILE,
)
from ki.transformer import NoteTransformer, FlatNote

logging.basicConfig(level=logging.INFO)

T = TypeVar("T")

# TODO: What if there is a deck called `_media`?
MEDIA = "_media"
DEV_NULL = "/dev/null"
BATCH_SIZE = 300
HTML_REGEX = r"</?\s*[a-z-][^>]*\s*>|(\&(?:[\w\d]+|#\d+|#x[a-f\d]+);)"
REMOTE_NAME = "anki"
BRANCH_NAME = "main"
CHANGE_TYPES = "A D R M T".split()
TQDM_NUM_COLS = 80
MAX_FILENAME_LEN = 40
IGNORE_DIRECTORIES = set([GIT, KI, MEDIA])
IGNORE_FILES = set([GITIGNORE_FILE, GITMODULES_FILE, MODELS_FILE])
HEAD_SUFFIX = Path("ki-head")
LOCAL_SUFFIX = Path("ki-local")
REMOTE_SUFFIX = Path("ki-remote")
FIELD_HTML_SUFFIX = Path("ki-fieldhtml")

GENERATED_HTML_SENTINEL = "data-original-markdown"
MEDIA_FILE_RECURSIVE_PATTERN = f"**/{MEDIA}/*"

# This is the key for media files associated with notetypes instead of the
# contents of a specific note.
NOTETYPE_NID = -57

MD = ".md"
FAILED = "Failed: exiting."

WARNING_IGNORE_LIST = [NotAnkiNoteWarning, UnPushedPathWarning, MissingMediaFileWarning]

SPINNER = "bouncingBall"
VERBOSE = False
PROFILE = False


@beartype
def lock(col_file: ExtantFile) -> sqlite3.Connection:
    """Check that lock can be acquired on a SQLite3 database given a path."""
    try:
        con = sqlite3.connect(col_file, timeout=0.1)
        con.isolation_level = "EXCLUSIVE"
        con.execute("BEGIN EXCLUSIVE")
    except sqlite3.DatabaseError as err:
        raise SQLiteLockError(col_file, err) from err
    if sys.platform == "win32":
        con.commit()
        con.close()
    return con


@beartype
def unlock(con: sqlite3.Connection) -> None:
    """Unlock a SQLite3 database."""
    if sys.platform == "win32":
        return
    con.commit()
    con.close()


@beartype
def copy_repo(repo_ref: RepoRef, suffix: str) -> git.Repo:
    """Get a temporary copy of a git repository in /tmp/<suffix>/."""
    # Copy the entire repo into `<tmp_dir>/suffix/`.
    target: NoFile = F.test(F.mkdtemp() / suffix)
    ephem = git.Repo(F.copytree(F.working_dir(repo_ref.repo), target))

    # Annihilate the .ki subdirectory.
    ki_dir = F.test(F.working_dir(ephem) / KI)
    if isinstance(ki_dir, ExtantDir):
        F.rmtree(ki_dir)

    # Do a reset --hard to the given SHA.
    ephem.git.reset(repo_ref.sha, hard=True)

    return ephem


@beartype
def copy_kirepo(kirepo_ref: KiRepoRef, suffix: str) -> KiRepo:
    """
    Given a KiRepoRef, i.e. a pair of the form (kirepo, SHA), we clone
    `kirepo.repo` into a temp directory and hard reset to the given commit
    hash. Copies the .ki/ directory from `kirepo_ref.kirepo` without making any
    changes.

    Parameters
    ----------
    suffix : pathlib.Path
        /tmp/.../ path suffix, e.g. `ki/local/`.
    kirepo_ref : KiRepoRef
        The ki repository to clone, and a commit for it.

    Returns
    -------
    KiRepo
        The cloned repository.
    """
    ref: RepoRef = F.kirepo_ref_to_repo_ref(kirepo_ref)
    ephem: git.Repo = copy_repo(ref, suffix)
    ki_dir: Path = F.test(F.working_dir(ephem) / KI)
    if not isinstance(ki_dir, NoFile):
        raise ExpectedNonexistentPathError(ki_dir)
    F.copytree(kirepo_ref.kirepo.ki_dir, ki_dir)
    kirepo: KiRepo = M.kirepo(F.working_dir(ephem))

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
def get_note_warnings(
    path: Path, root: ExtantDir, ignore_files: Set[str], ignore_dirs: Set[str]
) -> Optional[Warning]:
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
    if path.name in ignore_files | ignore_dirs:
        return UnPushedPathWarning(path, path.name)

    # If any of the patterns in `patterns` resolve to one of the parents of
    # `path`, return a warning, so that we are able to filter out entire
    # directories.
    components: Tuple[str, ...] = path.parts
    dirnames: Set[str] = set(components) & ignore_dirs
    for dirname in dirnames:
        return UnPushedPathWarning(path, dirname)

    # If `path` is an extant file (not a directory) and NOT a note, ignore it.
    abspath = (root / path).resolve()
    if abspath.exists() and abspath.is_file():
        file = ExtantFile(abspath)
        if not is_anki_note(file):
            return NotAnkiNoteWarning(file)

    return None


@beartype
def unsubmodule_repo(repo: git.Repo) -> git.Repo:
    """
    Un-submodule all the git submodules (convert them to ordinary
    subdirectories and destroy their commit history).  Commit the changes to
    the main repository.

    MUTATES REPO in-place!

    UNSAFE: git.rm() calls.
    """
    gitmodules_path: Path = Path(repo.working_dir) / GITMODULES_FILE
    for sm in repo.submodules:

        # The submodule path is guaranteed to exist by gitpython.
        sm_path = Path(sm.module().working_tree_dir)
        repo.git.rm(sm_path, cached=True)

        # Annihilate `.gitmodules` file.
        if gitmodules_path.is_file:
            repo.git.rm(gitmodules_path, ignore_unmatch=True)

        sm_git_path = F.test(sm_path / GIT)
        if isinstance(sm_git_path, ExtantDir):
            F.rmtree(sm_git_path)
        else:
            (sm_path / GIT).unlink(missing_ok=True)

        # Directory should still exist after `git.rm()`.
        repo.git.add(sm_path)
        _ = repo.index.commit(f"Add submodule `{sm.name}` as ordinary directory.")

    if gitmodules_path.exists():
        repo.git.rm(gitmodules_path)
        _ = repo.index.commit("Remove `.gitmodules` file.")
    return repo


@beartype
def diff2(
    repo: git.Repo,
    parser: Lark,
    transformer: NoteTransformer,
) -> List[Union[Delta, Warning]]:
    """Diff `repo` from `HEAD~1` to `HEAD`."""
    with F.halo(text=f"Checking out repo '{F.working_dir(repo)}' at HEAD~1..."):
        head1: RepoRef = M.repo_ref(repo, repo.commit("HEAD~1").hexsha)
        # pylint: disable=consider-using-f-string
        uuid = "%4x" % random.randrange(16**4)
        # pylint: enable=consider-using-f-string
        head1_repo = copy_repo(head1, suffix=f"HEAD~1-{uuid}")

    # We diff from A~B.
    b_repo = repo
    a_repo = head1_repo

    # Use a `DiffIndex` to get the changed files.
    deltas = []
    a_dir = F.test(Path(a_repo.working_dir))
    b_dir = F.test(Path(b_repo.working_dir))

    head = repo.commit("HEAD")
    with F.halo(text=f"Diffing '{head1.sha}' ~ '{head.hexsha}'..."):
        diff_index = repo.commit("HEAD~1").diff(head, create_patch=VERBOSE)

    for change_type in GitChangeType:

        diffs = diff_index.iter_change_type(change_type.value)
        bar = tqdm(diffs, ncols=TQDM_NUM_COLS, leave=False)
        bar.set_description(f"{change_type}")
        for diff in bar:
            a_relpath: str = diff.a_path
            b_relpath: str = diff.b_path
            if diff.a_path is None:
                a_relpath = b_relpath
            if diff.b_path is None:
                b_relpath = a_relpath

            a_warning: Optional[Warning] = get_note_warnings(
                Path(a_relpath),
                a_dir,
                ignore_files=IGNORE_FILES,
                ignore_dirs=IGNORE_DIRECTORIES,
            )
            b_warning: Optional[Warning] = get_note_warnings(
                Path(b_relpath),
                b_dir,
                ignore_files=IGNORE_FILES,
                ignore_dirs=IGNORE_DIRECTORIES,
            )

            if VERBOSE:
                logger.debug(f"{a_relpath} -> {b_relpath}")
                logger.debug(diff.diff.decode())

            if a_warning is not None:
                deltas.append(a_warning)
                continue
            if b_warning is not None:
                deltas.append(b_warning)
                continue

            a_path = F.test(a_dir / a_relpath)
            b_path = F.test(b_dir / b_relpath)

            a_relpath = Path(a_relpath)
            b_relpath = Path(b_relpath)

            if change_type == GitChangeType.DELETED:
                if not isinstance(a_path, ExtantFile):
                    deltas.append(DeletedFileNotFoundWarning(a_relpath))
                    continue

                deltas.append(Delta(change_type, a_path, a_relpath))
                continue

            if not isinstance(b_path, ExtantFile):
                deltas.append(DiffTargetFileNotFoundWarning(b_relpath))
                continue

            if change_type == GitChangeType.RENAMED:
                a_delta = Delta(GitChangeType.DELETED, a_path, a_relpath)
                b_delta = Delta(GitChangeType.ADDED, b_path, b_relpath)
                a_decknote: DeckNote = parse_markdown_note(parser, transformer, a_delta)
                b_decknote: DeckNote = parse_markdown_note(parser, transformer, b_delta)
                if a_decknote.nid != b_decknote.nid:
                    deltas.append(a_delta)
                    deltas.append(b_delta)
                    continue

            deltas.append(Delta(change_type, b_path, b_relpath))

    if VERBOSE:
        echo(f"Diffing '{repo.working_dir}': '{head1.sha}' ~ '{head.hexsha}'")

    return deltas


@beartype
def parse_notetype_dict(nt: Dict[str, Any]) -> Notetype:
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
    # If we can't even read the name of the notetype, then we can't print out a
    # nice error message in the event of a `KeyError`. So we have to print out
    # a different error message saying that the notetype doesn't have a name
    # field.
    try:
        nt["name"]
    except KeyError as err:
        raise UnnamedNotetypeError(nt) from err
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
            raise MissingFieldOrdinalError(sort_ordinal, nt["name"])

        notetype = Notetype(
            id=nt["id"],
            name=nt["name"],
            type=nt["type"],
            flds=list(fields.values()),
            tmpls=templates,
            sortf=fields[sort_ordinal],
            dict=nt,
        )

    except KeyError as err:
        key = str(err)
        raise NotetypeKeyError(key, str(nt["name"])) from err
    return notetype


@beartype
def get_models_recursively(kirepo: KiRepo, silent: bool) -> Dict[str, Notetype]:
    """
    Find and merge all `models.json` files recursively.

    Should we check for duplicates?

    Returns
    -------
    Dict[int, Notetype]
        A dictionary sending model names to Notetypes.
    """
    all_models: Dict[str, Notetype] = {}

    # Load notetypes from json files.
    bar = tqdm(F.rglob(kirepo.root, MODELS_FILE), ncols=TQDM_NUM_COLS, leave=not silent)
    bar.set_description("Models")
    for models_file in bar:

        with open(models_file, "r", encoding="UTF-8") as models_f:
            new_nts: Dict[int, Dict[str, Any]] = json.load(models_f)

        models: Dict[str, Notetype] = {}
        for _, nt in new_nts.items():
            notetype: Notetype = parse_notetype_dict(nt)
            models[notetype.name] = notetype

        # Add mappings to dictionary.
        all_models.update(models)

    return all_models


@beartype
def display_fields_health_warning(note: Note) -> int:
    """Display warnings when Anki's fields health check fails."""
    health = note.fields_check()
    if health == 1:
        warn(f"Found empty note '{note.id}'")
        warn(f"Fields health check code: {health}")
    elif health == 2:
        duplication = (
            "Failed to add note to collection. Notetype/fields of "
            + f"note '{note.id}' are duplicate of existing note."
        )
        warn(duplication)
        warn(f"First field:\n{html_to_screen(note.fields[0])}")
        warn(f"Fields health check code: {health}")
    elif health != 0:
        death = (
            f"fatal: Note '{note.id}' failed fields check with unknown "
            + f"error code: {health}"
        )
        logger.error(death)
    return health


@beartype
def parse_markdown_note(
    parser: Lark, transformer: NoteTransformer, delta: Delta
) -> DeckNote:
    """Parse with lark."""
    tree = parser.parse(delta.path.read_text(encoding="UTF-8"))
    flatnote: FlatNote = transformer.transform(tree)
    parts: Tuple[str, ...] = delta.relpath.parent.parts
    deck: str = "::".join(parts)
    return DeckNote(
        title=flatnote.title,
        nid=flatnote.nid,
        deck=deck,
        model=flatnote.model,
        tags=flatnote.tags,
        markdown=flatnote.markdown,
        fields=flatnote.fields,
    )


@beartype
def plain_to_html(plain: str) -> str:
    """Convert plain text to html"""
    # Minor clean up
    plain = plain.replace(r"&lt;", "<")
    plain = plain.replace(r"&gt;", ">")
    plain = plain.replace(r"&amp;", "&")
    plain = plain.replace(r"&nbsp;", " ")
    plain = re.sub(r"\<b\>\s*\<\/b\>", "", plain)
    plain = re.sub(r"\<i\>\s*\<\/i\>", "", plain)
    plain = re.sub(r"\<div\>\s*\<\/div\>", "", plain)

    # Strip double quotes from `src` attributes with newlines within HTML tags.
    plain = re.sub('src=\n"(\\S+)"', "src=\n\\1", plain)

    # Convert newlines to `<br>` tags.
    if not re.search(HTML_REGEX, plain):
        plain = plain.replace("\n", "<br>")

    return plain.strip()


@beartype
def update_note(
    note: Note, decknote: DeckNote, old_notetype: Notetype, new_notetype: Notetype
) -> Tuple[Note, List[Warning]]:
    """
    Change all the data of `note` to that given in `decknote`.

    This is only to be called on notes whose nid already exists in the
    database.  Creates a new deck if `decknote.deck` doesn't exist.  Assumes
    that the model has already been added to the collection, and raises an
    exception if it finds otherwise.  Changes notetype to that specified by
    `decknote.model`.  Overwrites all fields with `decknote.fields`.

    Updates:
    - tags
    - deck
    - model
    - fields
    """

    # Check that the passed argument `new_notetype` has a name consistent with
    # the model specified in `decknote`. The former should be derived from the
    # latter, and if they don't match, there is a bug in the caller.
    if decknote.model != new_notetype.name:
        raise NotetypeMismatchError(decknote, new_notetype)

    note.tags = decknote.tags
    note.flush()

    # Set the deck of the given note, and create a deck with this name if it
    # doesn't already exist. See the comments/docstrings in the implementation
    # of the `anki.decks.DeckManager.id()` method.
    newdid: int = note.col.decks.id(decknote.deck, create=True)
    cids = [c.id for c in note.cards()]

    # Set deck for all cards of this note.
    if cids:
        note.col.set_deck(cids, newdid)

    # Change notetype of note.
    fmap: Dict[str, None] = {}
    for field in old_notetype.flds:
        fmap[field.ord] = None

    # Change notetype (also clears all fields).
    if old_notetype.id != new_notetype.id:
        note.col.models.change(
            old_notetype.dict, [note.id], new_notetype.dict, fmap, None
        )
        note.load()

    # Validate field keys against notetype.
    warnings: List[Warning] = validate_decknote_fields(new_notetype, decknote)
    if len(warnings) > 0:
        return note, warnings

    # Set field values. This is correct because every field name that appears
    # in `new_notetype` is contained in `decknote.fields`, or else we would
    # have printed a warning and returned above.
    for key, field in decknote.fields.items():
        if key not in note:
            warnings.append(NoteFieldValidationWarning(note.id, key, new_notetype))
            continue
        if decknote.markdown:
            logger.warning("The 'markdown' flag is deprecated (now always 'False').")
        note[key] = plain_to_html(field)

    # Flush fields to collection object.
    note.flush()

    # Remove if unhealthy.
    health = display_fields_health_warning(note)
    if health != 0:
        note.col.remove_notes([note.id])
        warnings.append(UnhealthyNoteWarning(note.id, health))

    return note, warnings


@beartype
def validate_decknote_fields(notetype: Notetype, decknote: DeckNote) -> List[Warning]:
    """Validate that the fields given in the note match the notetype."""
    warnings: List[Warning] = []
    names: List[str] = [field.name for field in notetype.flds]

    # TODO: It might also be nice to print the path of the note in the
    # repository. This would have to be added to the `DeckNote` spec.
    if len(decknote.fields.keys()) != len(names):
        warnings.append(WrongFieldCountWarning(decknote, names))

    for x, y in zip(names, decknote.fields.keys()):
        if x != y:
            warnings.append(InconsistentFieldNamesWarning(x, y, decknote))

    return warnings


# TODO: This should use other fields when there is not enough content in the
# first field to get a unique filename, if they exist. Note that we must still
# treat the case where all fields are images!
@beartype
def get_note_path(colnote: ColNote, deck_dir: ExtantDir, card_name: str = "") -> NoFile:
    """Get note path from sort field text."""
    field_text = colnote.sortf_text

    # Construct filename, stripping HTML tags and sanitizing (quickly).
    field_text = plain_to_html(field_text)
    field_text = re.sub("<[^<]+?>", "", field_text)

    # If the HTML stripping removed all text, we just slugify the raw sort
    # field text.
    if len(field_text) == 0:
        field_text = colnote.sortf_text

    name = field_text[:MAX_FILENAME_LEN]
    slug = F.slugify(name)

    # Make it so `slug` cannot possibly be an empty string, because then we get
    # a `Path('.')` which is a bug, and causes a runtime exception.  If all
    # else fails, generate a random hex string to use as the filename.
    if len(slug) == 0:
        slug = str(colnote.n.id)
        msg = f"Slug for '{colnote.n.id}' is empty. Using nid as filename"
        logger.warning(msg)

    if card_name != "":
        slug = f"{slug}_{card_name}"
    filename: str = f"{slug}{MD}"
    note_path = F.test(deck_dir / filename, resolve=False)

    i = 1
    while not isinstance(note_path, NoFile):
        filename = f"{slug}_{i}{MD}"
        note_path = F.test(deck_dir / filename, resolve=False)
        i += 1

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
    """
    Construct path to deck directory and create it, allowing the case in which
    the directory already exists because we already created one of its
    children, in which case this function is a no-op.
    """
    # Strip leading periods so we don't get hidden folders.
    components = deck_name.split("::")
    components = [re.sub(r"^\.", r"", comp) for comp in components]
    components = [re.sub(r"/", r"-", comp) for comp in components]
    deck_path = Path(targetdir, *components)
    return F.force_mkdir(deck_path)


@beartype
def get_field_note_id(nid: int, fieldname: str) -> str:
    """A str ID that uniquely identifies field-note pairs."""
    return f"{nid}{F.slugify(fieldname)}"


@beartype
def push_decknote_to_anki(
    col: Collection, decknote: DeckNote
) -> Tuple[ColNote, List[Warning]]:
    """
    Update the Anki `Note` object in `col` corresponding to `decknote`,
    creating it if it does not already exist.

    Raises
    ------
    MissingNotetypeError
        If we can't find a notetype with the name provided in `decknote`.
    NoteFieldKeyError
        If the parsed sort field name from the notetype specified in `decknote`
        does not exist.
    """
    # Notetype/model names are privileged in Anki, so if we don't find the
    # right name, we raise an error.
    model_id: Optional[int] = col.models.id_for_name(decknote.model)
    if model_id is None:
        raise MissingNotetypeError(decknote.model)

    new = False
    note: Note
    try:
        note = col.get_note(decknote.nid)
    except NotFoundError:
        note = col.new_note(model_id)
        col.add_note(note, col.decks.id(decknote.deck, create=True))
        new = True

    # If we are updating an existing note, we need to know the old and new
    # notetypes, and then update the notetype (and the rest of the note data)
    # accordingly.
    old_notetype: Notetype = parse_notetype_dict(note.note_type())
    new_notetype: Notetype = parse_notetype_dict(col.models.get(model_id))
    note, warnings = update_note(note, decknote, old_notetype, new_notetype)

    # Get the text of the sort field for this note.
    try:
        sortf_text: str = note[new_notetype.sortf.name]
    except KeyError as err:
        raise NoteFieldKeyError(str(err), note.id) from err

    colnote = ColNote(
        n=note,
        new=new,
        deck=decknote.deck,
        title=decknote.title,
        old_nid=decknote.nid,
        markdown=decknote.markdown,
        notetype=new_notetype,
        sortf_text=sortf_text,
    )
    return colnote, warnings


@beartype
def get_colnote(col: Collection, nid: int) -> ColNote:
    """Get a dataclass representation of an Anki note."""
    try:
        note = col.get_note(nid)
    except NotFoundError as err:
        raise MissingNoteIdError(nid) from err
    notetype: Notetype = parse_notetype_dict(note.note_type())

    # Get sort field content. See comment where we subscript in the same way in
    # `push_decknote_to_anki()`.
    try:
        sortf_text: str = note[notetype.sortf.name]
    except KeyError as err:
        raise NoteFieldKeyError(str(err), nid) from err

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
    return colnote


@beartype
def get_header_lines(colnote) -> List[str]:
    """Get header of markdown representation of note."""
    lines = [
        "## Note",
        f"nid: {colnote.n.id}",
        f"model: {colnote.notetype.name}",
    ]
    tags = [tag.replace(",", "") for tag in colnote.n.tags]
    lines += [f"tags: {', '.join(tags)}"]

    # TODO: There is almost certainly a bug here, since when the sentinel
    # *does* appear, we don't add a `markdown` field at all, but it is required
    # in the note grammar, so we'll get a parsing error when we try to push. A
    # test must be written for this case.
    if not any(GENERATED_HTML_SENTINEL in field for field in colnote.n.values()):
        lines += ["markdown: false"]

    lines += [""]
    return lines


def files_in_str(
    col: Collection, string: str, include_remote: bool = False
) -> list[str]:
    """A copy of `MediaManager.files_in_str()`, but without LaTeX rendering."""
    # Extract filenames.
    files = []
    for reg in col.media.regexps:
        for match in re.finditer(reg, string):
            fname = match.group("fname")
            is_local = not re.match("(https?|ftp)://", fname.lower())
            if is_local or include_remote:
                files.append(fname)
    return files


@beartype
def copy_media_files(
    col: Collection,
    media_target_dir: EmptyDir,
    silent: bool,
) -> Tuple[Dict[int, Set[ExtantFile]], Set[Warning]]:
    """
    Get a list of extant media files used in notes and notetypes, copy those
    media files to the top-level `_media/` directory in the repository root,
    and return a map sending note ids to sets of copied media files.

    Adapted from code in `anki/pylib/anki/exporting.py`. Specifically, the
    `AnkiExporter.exportInto()` function.

    SQLite3 notes table schema
    --------------------------
    CREATE TABLE notes (
        id integer PRIMARY KEY,
        guid text NOT NULL,
        mid integer NOT NULL,
        mod integer NOT NULL,
        usn integer NOT NULL,
        tags text NOT NULL,
        flds text NOT NULL,
        -- The use of type integer for sfld is deliberate, because it means
        -- that integer values in this field will sort numerically.
        sfld integer NOT NULL,
        csum integer NOT NULL,
        flags integer NOT NULL,
        data text NOT NULL
    );

    Parameters
    ----------
    col
        Anki collection.
    silent
        Whether to display stdout.
    """

    # All note ids as a string for the SQL query.
    strnids = ids2str(list(col.find_notes(query="")))

    # This is the path to the media directory. In the original implementation
    # of `AnkiExporter.exportInto()`, there is check made of the form
    #
    #   if self.mediaDir:
    #
    # before doing path manipulation with this string.
    #
    # Examining the `__init__()` function of `MediaManager`, we can see that
    # `col.media.dir()` will only be `None` in the case where `server=True` is
    # passed to the `Collection` constructor. But since we do the construction
    # within ki, we have a guarantee that this will never be true, and thus we
    # can assume it is a nonempty string, which is all we need for the
    # following code to be safe.
    media_dir = F.test(Path(col.media.dir()))
    if not isinstance(media_dir, ExtantDir):
        raise MissingMediaDirectoryError(col.path, media_dir)

    # Find only used media files, collecting warnings for bad paths.
    media: Dict[int, Set[ExtantFile]] = {}
    warnings: Set[Warning] = set()
    query: str = "select * from notes where id in " + strnids
    rows: List[NoteDBRow] = [NoteDBRow(*row) for row in col.db.all(query)]

    bar = tqdm(rows, ncols=TQDM_NUM_COLS, leave=not silent)
    bar.set_description("Media")
    for row in bar:
        for file in files_in_str(col, row.flds):

            # Skip files in subdirs.
            if file != os.path.basename(file):
                continue
            media_file = F.test(media_dir / file)
            if isinstance(media_file, ExtantFile):
                copied_file = F.copyfile(media_file, media_target_dir, media_file.name)
                media[row.nid] = media.get(row.nid, set()) | set([copied_file])
            else:
                warnings.add(MissingMediaFileWarning(col.path, media_file))

    mids = col.db.list("select distinct mid from notes where id in " + strnids)

    # Faster version.
    _, _, files = F.shallow_walk(media_dir)
    fnames: List[Path] = [Path(file.name) for file in files]
    for fname in fnames:

        # Notetype template media files are *always* prefixed by underscores.
        if str(fname).startswith("_"):

            # Scan all models in mids for reference to fname.
            for m in col.models.all():
                if int(m["id"]) in mids and _modelHasMedia(m, str(fname)):

                    # If the path referenced by `fname` doesn't exist or is not
                    # a file, we do not display a warning or return an error.
                    # This path certainly ought to exist, since `fname` was
                    # obtained from an `os.listdir()` call.
                    media_file = F.test(media_dir / fname)
                    if isinstance(media_file, ExtantFile):
                        copied_file = F.copyfile(
                            media_file, media_target_dir, media_file.name
                        )
                        notetype_media = media.get(NOTETYPE_NID, set())
                        media[NOTETYPE_NID] = notetype_media | set([copied_file])
                    break

    return media, warnings


@beartype
def _modelHasMedia(model: NotetypeDict, fname: str) -> bool:
    """
    Check if a notetype has media.

    Adapted from `anki.exporting.AnkiExporter._modelHasMedia()`, which is an
    instance method, but does not make any use of `self`, and so could be a
    staticmethod. It is a pure function.
    """
    # First check the styling
    if fname in model["css"]:
        return True
    # If no reference to fname then check the templates as well
    for t in model["tmpls"]:
        if fname in t["qfmt"] or fname in t["afmt"]:
            return True
    return False


@beartype
def write_repository(
    col_file: ExtantFile,
    targetdir: ExtantDir,
    leaves: Leaves,
    media_target_dir: EmptyDir,
    silent: bool,
    verbose: bool,
) -> None:
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

    tidy_field_files: Dict[str, ExtantFile] = {}
    decks: Dict[str, List[ColNote]] = {}

    # Open collection using a `Maybe`.
    cwd: ExtantDir = F.cwd()
    col: Collection = M.collection(col_file)
    F.chdir(cwd)

    # ColNote-containing data structure, to be passed to `write_decks()`.
    colnotes: Dict[int, ColNote] = {}

    # Query all note ids, get the deck from each note, and construct a map
    # sending deck names to lists of notes.
    all_nids = list(col.find_notes(query=""))

    bar = tqdm(all_nids, ncols=TQDM_NUM_COLS, leave=not silent)
    bar.set_description("Notes")
    for nid in bar:
        colnote: ColNote = get_colnote(col, nid)
        colnotes[nid] = colnote
        decks[colnote.deck] = decks.get(colnote.deck, []) + [colnote]
        for field_name, field_text in colnote.n.items():
            field_text: str = html_to_screen(field_text)
            if re.search(HTML_REGEX, field_text):
                fid: str = get_field_note_id(nid, field_name)
                html_file: NoFile = F.test(root / fid)
                tidy_field_files[fid] = F.write(html_file, field_text)

    tidy_html_recursively(root, silent)

    media: Dict[int, Set[ExtantFile]]
    media, warnings = copy_media_files(col, media_target_dir, silent=silent)

    write_decks(col, targetdir, colnotes, media, tidy_field_files, silent)

    num_displayed: int = 0
    for warning in warnings:
        if verbose or type(warning) not in WARNING_IGNORE_LIST:
            click.secho(str(warning), fg="yellow")
            num_displayed += 1
    num_suppressed: int = len(warnings) - num_displayed
    echo(f"Warnings suppressed: {num_suppressed} (show with '--verbose')")

    F.rmtree(root)
    col.close(save=False)


@beartype
def write_decks(
    col: Collection,
    targetdir: ExtantDir,
    colnotes: Dict[int, ColNote],
    media: Dict[int, Set[ExtantFile]],
    tidy_field_files: Dict[str, ExtantFile],
    silent: bool,
) -> None:
    """
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
        json.dump(models_map, f, ensure_ascii=False, indent=4, sort_keys=True)

    # Implement new `ColNote`-writing procedure, using `DeckTreeNode`s.
    #
    # It must do the following for each deck:
    # - create the deck directory
    # - write the models.json file
    # - create and populate the media directory
    # - write the note payload for each note in the correct deck, exactly once
    #
    # In other words, for each deck, we need to write all of its:
    # - models
    # - media
    # - notes
    #
    # The first two are cumulative: we want the models and media of subdecks to
    # be included in their ancestors. The notes, however, should not be
    # cumulative. Indeed, we want each note to appear exactly once in the
    # entire repository, making allowances for the case where a single note's
    # cards are spread across multiple decks, in which case we must create a
    # symlink.
    #
    # And actually, both of these cases are nicely taken care of for us by the
    # `DeckManager.cids()` function, which has a `children: bool` parameter
    # which toggles whether or not to get the card ids of subdecks or not.
    root: DeckTreeNode = col.decks.deck_tree()

    @beartype
    def postorder(node: DeckTreeNode) -> List[DeckTreeNode]:
        """
        Post-order traversal. Guarantees that we won't process a node until
        we've processed all its children.
        """
        traversal: List[DeckTreeNode] = []
        for child in node.children:
            traversal += postorder(child)
        traversal += [node]
        return traversal

    @beartype
    def preorder(node: DeckTreeNode) -> List[DeckTreeNode]:
        """
        Pre-order traversal. Guarantees that we won't process a node until
        we've processed all its ancestors.
        """
        traversal: List[DeckTreeNode] = [node]
        for child in node.children:
            traversal += preorder(child)
        return traversal

    @beartype
    def map_parents(node: DeckTreeNode, col: Collection) -> Dict[str, DeckTreeNode]:
        """Map deck names to parent `DeckTreeNode`s."""
        parents: Dict[str, DeckTreeNode] = {}
        for child in node.children:
            did: int = child.deck_id
            name: str = col.decks.name(did)
            parents[name] = node
            subparents = map_parents(child, col)
            parents.update(subparents)
        return parents

    # All card ids we've already processed.
    written_cids: Set[int] = set()

    # Map nids we've already written files for to a dataclass containing:
    # - the corresponding `ExtantFile`
    # - the deck id of the card for which we wrote it
    #
    # This deck id identifies the deck corresponding to the location where all
    # the symlinks should point.
    written_notes: Dict[int, WrittenNoteFile] = {}

    nodes: List[DeckTreeNode] = postorder(root)

    bar = tqdm(nodes, ncols=TQDM_NUM_COLS, leave=not silent)
    bar.set_description("Decks")
    for node in bar:

        # The name stored in a `DeckTreeNode` object is not the full name of
        # the deck, it is just the 'basename'. The `postorder()` function
        # returns a deck with `did == 0` at the end of each call, probably
        # because this is the implicit parent deck of all top-level decks. This
        # deck empirically always has the empty string as its name. This is
        # likely an Anki implementation detail. As a result, we ignore any deck
        # with empty basename. We do this as opposed to ignoring decks with
        # `did == 0` because it seems less likely to change if the Anki devs
        # decide to mess with the implementation. Empty deck names will always
        # be reserved, but they might e.g. decide ``did == -1`` makes more
        # sense.
        did: int = node.deck_id
        basename: str = node.name
        if basename == "":
            continue
        name = col.decks.name(did)

        deck_dir: ExtantDir = create_deck_dir(name, targetdir)
        children: Set[CardId] = set(col.decks.cids(did=did, children=False))
        descendants: List[CardId] = col.decks.cids(did=did, children=True)
        descendant_nids: Set[int] = {NOTETYPE_NID}
        descendant_mids: Set[int] = set()

        # TODO: The code in this loop would be better placed in its own function.
        for cid in descendants:
            card: Card = col.get_card(cid)
            descendant_nids.add(card.nid)
            descendant_mids.add(card.note().mid)

            # Card writes should not be cumulative, so we only perform them if
            # `cid` is the card id of a card that is in the deck corresponding
            # to `node`, but not in any of its children.
            if cid not in children:
                continue

            # We only even consider writing *anything* (file or symlink) to
            # disk after checking that we haven't already processed this
            # particular card id. If we have, then we only need to bother with
            # the cumulative stuff above (nids and mids, which we keep track of
            # in order to write out models and media).
            #
            # TODO: This fixes a very serious bug. Write a test to capture the
            # issue. Use the Japanese Core 2000 deck as a template if needed.
            if cid in written_cids:
                continue
            written_cids.add(cid)

            # We only write the payload if we haven't seen this note before.
            # Otherwise, this must be a different card generated from the same
            # note, so we symlink to the location where we've already written
            # it to disk.
            colnote: ColNote = colnotes[card.nid]

            if card.nid not in written_notes:
                note_path: NoFile = get_note_path(colnote, deck_dir)
                payload: str = get_note_payload(colnote, tidy_field_files)
                note_path: ExtantFile = F.write(note_path, payload)
                written_notes[card.nid] = WrittenNoteFile(did, note_path)
            else:
                # If `card` is in the same deck as the card we wrote `written`
                # for, then there is no need to create a symlink, because the
                # note file is already there, in the correct deck directory.
                #
                # TODO: This fixes a very serious bug. Write a test to capture the
                # issue. Use the Japanese Core 2000 deck as a template if needed.
                written: WrittenNoteFile = written_notes[card.nid]
                if card.did == written.did:
                    continue

                # Get card template name.
                template: TemplateDict = card.template()
                name: str = template["name"]

                note_path: NoFile = get_note_path(colnote, deck_dir, name)
                abs_target: ExtantFile = written_notes[card.nid].file
                distance = len(note_path.parent.relative_to(targetdir).parts)
                up_path = Path("../" * distance)
                relative: Path = abs_target.relative_to(targetdir)
                target: Path = up_path / relative

                try:
                    F.symlink(note_path, target)
                except OSError as _:
                    trace = traceback.format_exc(limit=3)
                    logger.warning(f"Failed to create symlink for cid '{cid}'\n{trace}")

        # Write `models.json` for current deck.
        deck_models_map = {mid: models_map[mid] for mid in descendant_mids}
        with open(deck_dir / MODELS_FILE, "w", encoding="UTF-8") as f:
            json.dump(deck_models_map, f, ensure_ascii=False, indent=4, sort_keys=True)

    # TODO: This should be in its own function to make sure loop variables aren't being reused.

    # Chain symlinks up the deck tree into `<repo_root>/_media/`.
    media_dirs: Dict[str, ExtantDir] = {}
    parents: Dict[str, DeckTreeNode] = map_parents(root, col)
    for node in preorder(root):
        if node.name == "":
            continue
        did: int = node.deck_id
        fullname = col.decks.name(did)
        deck_dir: ExtantDir = create_deck_dir(fullname, targetdir)
        deck_media_dir: ExtantDir = F.force_mkdir(deck_dir / MEDIA)
        media_dirs[fullname] = deck_media_dir
        descendant_nids: Set[int] = {NOTETYPE_NID}
        descendants: List[CardId] = col.decks.cids(did=did, children=True)
        for cid in descendants:
            card: Card = col.get_card(cid)
            descendant_nids.add(card.nid)
        for nid in descendant_nids:
            if nid in media:
                for media_file in media[nid]:
                    parent: DeckTreeNode = parents[fullname]
                    if parent.name != "":
                        parent_did: int = parent.deck_id
                        parent_fullname: str = col.decks.name(parent_did)
                        parent_media_dir = media_dirs[parent_fullname]
                        abs_target: Symlink = F.test(
                            parent_media_dir / media_file.name, resolve=False
                        )
                    else:
                        abs_target: ExtantFile = media_file
                    path = F.test(deck_media_dir / media_file.name, resolve=False)
                    if not isinstance(path, NoFile):
                        continue
                    distance = len(path.parent.relative_to(targetdir).parts)
                    up_path = Path("../" * distance)
                    relative: Path = abs_target.relative_to(targetdir)
                    target: Path = up_path / relative

                    try:
                        F.symlink(path, target)
                    except OSError as _:
                        trace = traceback.format_exc(limit=3)
                        logger.warning(f"Failed to create symlink to media\n{trace}")


@beartype
def html_to_screen(html: str) -> str:
    """
    Convert html for a *single field* into plaintext, to be displayed within a
    markdown file.

    Does very litle (just converts HTML-escaped special characters like `<br>`
    tags or `&nbsp;`s to their UTF-8 equivalents) in the case where the
    sentinel string does not appear. This string, `GENERATED_HTML_SENTINEL`,
    indicates that a note was first created in a ki markdown file, and
    therefore its HTML structure can be safely stripped out within calls to
    `write_decks()`.
    """
    html = re.sub(r"\<style\>.*\<\/style\>", "", html, flags=re.S)
    plain = html

    # For convenience: Un-escape some common LaTeX constructs.
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

    plain = re.sub(r"\<b\>\s*\<\/b\>", "", plain)
    return plain.strip()


@beartype
def get_colnote_as_str(colnote: ColNote) -> str:
    """Return string representation of a `ColNote`."""
    lines = get_header_lines(colnote)
    for field_name, field_text in colnote.n.items():
        lines.append("### " + field_name)
        lines.append(html_to_screen(field_text))
        lines.append("")

    return "\n".join(lines)


@beartype
def get_note_payload(colnote: ColNote, tidy_field_files: Dict[str, ExtantFile]) -> str:
    """
    Return the markdown-converted contents of the Anki note represented by
    `colnote` as a string.

    Given a `ColNote`, which is a dataclass wrapper around a `Note` object
    which has been loaded from the DB, and a mapping from `fid`s (unique
    identifiers of field-note pairs) to paths, we check for each field of each
    note whether that field's `fid` is contained in `tidy_field_files`. If so,
    that means that the caller dumped the contents of this field to a file (the
    file with this path, in fact) in order to autoformat the HTML source. If
    this field was tidied/autoformatted, we read from that path to get the
    tidied source, otherwise, we use the field content present in the
    `ColNote`.
    """
    # Get tidied html if it exists.
    tidy_fields = {}
    for field_name, field_text in colnote.n.items():
        fid = get_field_note_id(colnote.n.id, field_name)
        if fid in tidy_field_files:
            # HTML5-tidy adds a newline after `<br>` in indent mode, so we
            # remove these, because `html_to_screen()` converts `<br>` tags to
            # newlines anyway.
            tidied_field_text: str = tidy_field_files[fid].read_text(encoding="UTF-8")
            tidied_field_text = tidied_field_text.replace("<br>\n", "\n")
            tidied_field_text = tidied_field_text.replace("<br/>\n", "\n")
            tidied_field_text = tidied_field_text.replace("<br />\n", "\n")
            tidy_fields[field_name] = tidied_field_text
        else:
            tidy_fields[field_name] = field_text

    lines = get_header_lines(colnote)
    for field_name, field_text in tidy_fields.items():
        lines.append("### " + field_name)
        lines.append(html_to_screen(field_text))
        lines.append("")

    return "\n".join(lines)


@beartype
def git_pull(
    remote: str,
    branch: str,
    cwd: ExtantDir,
    unrelated: bool,
    theirs: bool,
    check: bool,
    silent: bool,
) -> None:
    """Pull remote into branch using a subprocess call."""
    with F.halo(f"Pulling into '{cwd}'..."):
        args = ["git", "pull", "-v"]
        if unrelated:
            args += ["--allow-unrelated-histories"]
        if theirs:
            args += ["--strategy-option=theirs"]
        args += ["--verbose"]
        args += [remote, branch]
        p = subprocess.run(args, check=False, cwd=cwd, capture_output=True)
    echo(f"{p.stdout.decode()}", silent=silent)
    echo(f"{p.stderr.decode()}", silent=silent)
    if check and p.returncode != 0:
        click.secho(f"Error while pulling into '{cwd}'", fg="red")
        raise RuntimeError(f"Git failed with return code '{p.returncode}'.")
    echo(f"Pulling into '{cwd}'... done.", silent=silent)


@beartype
def echo(string: str, silent: bool = False) -> None:
    """Call `click.secho()` with formatting."""
    if not silent:
        click.secho(string, bold=True)


@beartype
def warn(string: str) -> None:
    """Call `click.secho()` with formatting (yellow)."""
    click.secho(f"WARNING: {string}", bold=True, fg="yellow")


@beartype
def tidy_html_recursively(root: ExtantDir, silent: bool) -> None:
    """Call html5-tidy on each file in `root`, editing in-place."""
    # Spin up subprocesses for tidying field HTML in-place.
    batches: List[List[ExtantFile]] = list(
        F.get_batches(F.rglob(root, "*"), BATCH_SIZE)
    )

    bar = tqdm(batches, ncols=TQDM_NUM_COLS, leave=not silent)
    bar.set_description("HTML")
    for batch in bar:
        # TODO: Should we fail silently here, so as to not bother user with
        # tidy warnings?
        command = [
            "tidy",
            "-q",
            "-m",
            "-i",
            "-omit",
            "-utf8",
            "--tidy-mark",
            "no",
            "--show-body-only",
            "yes",
            "--wrap",
            "68",
            "--wrap-attributes",
            "yes",
        ]
        command += batch
        try:
            subprocess.run(command, check=False, capture_output=True)
        except FileNotFoundError as err:
            raise MissingTidyExecutableError(err) from err


@beartype
def get_target(
    cwd: ExtantDir, col_file: ExtantFile, directory: str
) -> Tuple[EmptyDir, bool]:
    """Create default target directory."""
    path = F.test(Path(directory) if directory != "" else cwd / col_file.stem)
    new: bool = True
    if isinstance(path, NoPath):
        path.mkdir(parents=True)
        return M.emptydir(path), new
    if isinstance(path, EmptyDir):
        new = False
        return path, new
    raise TargetExistsError(path)


@beartype
def regenerate_note_file(colnote: ColNote, root: ExtantDir, relpath: Path) -> List[str]:
    """
    Construct the contents of a note corresponding to the arguments `colnote`,
    which itself was created from `decknote`, and then write it to disk.

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
        return []

    # Get paths to note in local repo, as distinct from staging repo.
    repo_note_path: Path = root / relpath

    # If this is not an entirely new file, remove it.
    if repo_note_path.is_file():
        repo_note_path.unlink()

    parent: ExtantDir = F.force_mkdir(repo_note_path.parent)
    new_note_path: NoFile = get_note_path(colnote, parent)
    F.write(new_note_path, get_colnote_as_str(colnote))

    # Construct a nice commit message line.
    msg = f"Reassigned nid: '{colnote.old_nid}' -> "
    msg += f"'{colnote.n.id}' in '{os.path.relpath(new_note_path, root)}'"
    return [msg]


@beartype
def echo_note_change_types(deltas: List[Delta]) -> None:
    """Write a table of git change types for notes to stdout."""
    is_change_type = lambda t: lambda d: d.status == t

    adds = list(filter(is_change_type(GitChangeType.ADDED), deltas))
    deletes = list(filter(is_change_type(GitChangeType.DELETED), deltas))
    renames = list(filter(is_change_type(GitChangeType.RENAMED), deltas))
    modifies = list(filter(is_change_type(GitChangeType.MODIFIED), deltas))
    typechanges = list(filter(is_change_type(GitChangeType.TYPECHANGED), deltas))

    WORD_PAD = 15
    COUNT_PAD = 9
    add_info: str = "ADD".ljust(WORD_PAD) + str(len(adds)).rjust(COUNT_PAD)
    delete_info: str = "DELETE".ljust(WORD_PAD) + str(len(deletes)).rjust(COUNT_PAD)
    modification_info: str = "MODIFY".ljust(WORD_PAD) + str(len(modifies)).rjust(
        COUNT_PAD
    )
    rename_info: str = "RENAME".ljust(WORD_PAD) + str(len(renames)).rjust(COUNT_PAD)
    typechange_info: str = "TYPE CHANGE".ljust(WORD_PAD) + str(len(typechanges)).rjust(
        COUNT_PAD
    )

    echo("=" * (WORD_PAD + COUNT_PAD))
    echo("Note change types")
    echo("-" * (WORD_PAD + COUNT_PAD))
    echo(add_info)
    echo(delete_info)
    echo(modification_info)
    echo(rename_info)
    echo(typechange_info)
    echo("=" * (WORD_PAD + COUNT_PAD))


@beartype
def add_models(col: Collection, models: Dict[str, Notetype]) -> None:
    """Add all new models."""

    @beartype
    def get_notetype_json(notetype: Notetype) -> str:
        dictionary: Dict[str, Any] = dataclasses.asdict(notetype)
        dictionary.pop("id")
        inner = dictionary["dict"]
        inner.pop("id")
        inner.pop("mod")
        dictionary["dict"] = inner
        return json.dumps(dictionary, sort_keys=True, indent=4)

    @beartype
    def notetype_hash_repr(notetype: Notetype) -> str:
        s = get_notetype_json(notetype)
        return f"JSON for '{pp.pformat(notetype.id)}':\n{s}"

    for model in models.values():

        # TODO: Consider waiting to parse `models` until after the
        # `add_dict()` call.
        #
        # Set the model id to `0`, and then add.
        # TODO: What happens if we try to add a model with the same name as
        # an existing model, but the two models are not the same,
        # content-wise?
        #
        # Check if a model already exists with this name, and get its `mid`.
        mid: Optional[int] = col.models.id_for_name(model.name)

        # TODO: This block is unfinished. We need to add new notetypes (and
        # rename them) only if they are 'new', where new means they are
        # different from anything else already in the DB, in the
        # content-addressed sense. If they are new, then we must indicate that
        # the notes we are adding actually have these new notetypes. For this,
        # it may make sense to use the hash of the notetype everywhere (i.e. in
        # the note file) rather than the name or mid.
        #
        # If a model already exists with this name, parse it, and check if its
        # hash is identical to the model we are trying to add.
        if mid is not None:
            nt: NotetypeDict = col.models.get(mid)

            # If we are trying to add a model that has the exact same content
            # and name as an existing model, skip it.
            existing_model: Notetype = parse_notetype_dict(nt)
            if get_notetype_json(model) == get_notetype_json(existing_model):
                continue

            logger.warning(
                f"Collision: New model '{model.name}' has same name "
                f"as existing model with mid '{mid}', but hashes differ."
            )
            logger.warning(notetype_hash_repr(model))
            logger.warning(notetype_hash_repr(existing_model))

            # If the hashes don't match, then we somehow need to update
            # `decknote.model` for the relevant notes.

            # TODO: Consider using the hash of the notetype instead of its
            # name.

        nt_copy: NotetypeDict = copy.deepcopy(model.dict)
        nt_copy["id"] = 0
        changes: OpChangesWithId = col.models.add_dict(nt_copy)
        nt: NotetypeDict = col.models.get(changes.id)
        model: Notetype = parse_notetype_dict(nt)
        echo(f"Added model '{model.name}'")


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
@click.option("--verbose", "-v", is_flag=True, help="Print more output.")
def clone(collection: str, directory: str = "", verbose: bool = False) -> None:
    """
    Clone an Anki collection into a directory.

    Parameters
    ----------
    collection : str
        The path to an `.anki2` collection file.
    directory : str, default=""
        An optional path to a directory to clone the collection into.
        Note: we check that this directory does not yet exist.
    """
    if PROFILE:
        # pylint: disable=import-outside-toplevel
        from pyinstrument import Profiler

        # pylint: enable=import-outside-toplevel
        profiler = Profiler()
        profiler.start()

    echo("Cloning.")
    col_file: ExtantFile = M.xfile(Path(collection))

    @beartype
    def cleanup(targetdir: ExtantDir, new: bool) -> Union[EmptyDir, NoPath]:
        """Cleans up after failed clone operations."""
        if new:
            return F.rmtree(targetdir)
        _, dirs, files = F.shallow_walk(targetdir)
        for directory in dirs:
            F.rmtree(directory)
        for file in files:
            os.remove(file)
        return F.test(targetdir)

    # Write all files to `targetdir`, and instantiate a `KiRepo` object.
    targetdir: EmptyDir
    new: bool
    targetdir, new = get_target(F.cwd(), col_file, directory)
    try:
        _, _ = _clone(
            col_file, targetdir, msg="Initial commit", silent=False, verbose=verbose
        )
        kirepo: KiRepo = M.kirepo(targetdir)
        F.write(kirepo.last_push_file, kirepo.repo.head.commit.hexsha)
        kirepo.repo.close()
        gc.collect()
        echo("Done.")
    except Exception as err:
        cleanup(targetdir, new)
        raise err

    if PROFILE:
        profiler.stop()
        s = profiler.output_html()
        Path("ki_clone_profile.html").resolve().write_text(s, encoding="UTF-8")


@beartype
def _clone(
    col_file: ExtantFile,
    targetdir: EmptyDir,
    msg: str,
    silent: bool,
    verbose: bool,
) -> Tuple[git.Repo, str]:
    """
    Clone an Anki collection into a directory.

    The caller expects that `targetdir` will be the root of a valid ki
    repository after this function is called, so we need to do our repo
    initialization with gitpython in here, as opposed to in `clone()`.

    Parameters
    ----------
    col_file : pathlib.Path
        The path to an `.anki2` collection file.
    targetdir : pathlib.Path
        A path to a directory to clone the collection into.
        Note: we check that this directory is empty.
    msg : str
        Message for initial commit.
    silent : bool
        Whether to suppress progress information printed to stdout.

    Returns
    -------
    repo : git.Repo
        The cloned repository.
    md5sum : str
        The hash of the Anki collection file.
    """
    echo(f"Found .anki2 file at '{col_file}'", silent=silent)

    # Create `.ki/` and `_media/`, and create empty metadata files in `.ki/`.
    # TODO: Consider writing a Maybe factory for all this.
    directories: Leaves = F.fmkleaves(targetdir, dirs={KI: KI, MEDIA: MEDIA})
    leaves: Leaves = F.fmkleaves(
        directories.dirs[KI],
        files={CONFIG_FILE: CONFIG_FILE, LAST_PUSH_FILE: LAST_PUSH_FILE},
        dirs={BACKUPS_DIR: BACKUPS_DIR},
    )

    md5sum = F.md5(col_file)
    echo(f"Computed md5sum: {md5sum}", silent)
    echo(f"Cloning into '{targetdir}'...", silent=silent)
    (targetdir / GITIGNORE_FILE).write_text(KI + "\n")

    # Write notes to disk.
    write_repository(
        col_file, targetdir, leaves, directories.dirs[MEDIA], silent, verbose
    )

    # Initialize the main repository.
    with F.halo("Initializing repository and committing contents..."):
        repo = git.Repo.init(targetdir, initial_branch=BRANCH_NAME)
        repo.git.add(all=True)
        _ = repo.index.commit(msg)

    # Store a checksum of the Anki collection file in the hashes file.
    append_md5sum(directories.dirs[KI], col_file.name, md5sum, silent)

    return repo, md5sum


@ki.command()
@beartype
def pull() -> None:
    """
    Pull from a preconfigured remote Anki collection into an existing ki
    repository.
    """
    if PROFILE:
        # pylint: disable=import-outside-toplevel
        from pyinstrument import Profiler

        # pylint: enable=import-outside-toplevel
        profiler = Profiler()
        profiler.start()

    # Check that we are inside a ki repository, and get the associated collection.
    kirepo: KiRepo = M.kirepo(F.cwd())
    con: sqlite3.Connection = lock(kirepo.col_file)
    md5sum: str = F.md5(kirepo.col_file)
    hashes: List[str] = kirepo.hashes_file.read_text(encoding="UTF-8").split("\n")
    hashes = list(filter(lambda l: l != "", hashes))
    if md5sum in hashes[-1]:
        echo("ki pull: up to date.")
        return

    _pull(kirepo, silent=False)
    unlock(con)

    if PROFILE:
        profiler.stop()
        s = profiler.output_html()
        Path("ki_pull_profile.html").resolve().write_text(s, encoding="UTF-8")


@beartype
def _pull(kirepo: KiRepo, silent: bool) -> None:
    """
    Pull into `kirepo` without checking if we are already up-to-date.

    Load the git repository at `anki_remote_root`, force pull (preferring
    'theirs', i.e. the new stuff from the sqlite3 database) changes from that
    repository (which is cloned straight from the collection, which in general
    may have new changes) into `last_push_repo`, and then pull `last_push_repo`
    into the main repository.

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

    Parameters
    ----------
    kirepo : KiRepo
        A dataclass representing the Ki repository in the cwd.
    silent : bool
        Whether to suppress progress information printed to stdout.

    Raises
    ------
    CollectionChecksumError
        If the Anki collection file was modified while pulling changes. This is
        very unlikely, since the caller acquires a lock on the SQLite3
        database.
    """
    md5sum: str = F.md5(kirepo.col_file)
    echo(f"Pulling from '{kirepo.col_file}'", silent)
    echo(f"Computed md5sum: {md5sum}", silent)

    # Copy `repo` into a temp directory and `reset --hard` at ref of last
    # successful `push()`.
    sha: str = kirepo.last_push_file.read_text(encoding="UTF-8")
    with F.halo(text=f"Checking out repo at '{sha}'..."):
        ref: RepoRef = M.repo_ref(kirepo.repo, sha=sha)
        last_push_repo: git.Repo = copy_repo(ref, f"{LOCAL_SUFFIX}-{md5sum}")
        unsub_repo: git.Repo = copy_repo(ref, f"unsub-{LOCAL_SUFFIX}-{md5sum}")

    # Ki clone collection into a temp directory at `anki_remote_root`.
    anki_remote_root: EmptyDir = F.mksubdir(F.mkdtemp(), REMOTE_SUFFIX / md5sum)
    msg = f"Fetch changes from DB at `{kirepo.col_file}` with md5sum `{md5sum}`"
    remote_repo, _ = _clone(
        kirepo.col_file,
        anki_remote_root,
        msg,
        silent=silent,
        verbose=False,
    )

    # Create git remote pointing to `remote_repo`, which represents the current
    # state of the Anki SQLite3 database, and pull it into `last_push_repo`.
    anki_remote = last_push_repo.create_remote(REMOTE_NAME, remote_repo.git_dir)
    unsub_remote = unsub_repo.create_remote(REMOTE_NAME, remote_repo.git_dir)
    last_push_root: ExtantDir = F.working_dir(last_push_repo)

    # =================== NEW PULL ARCHITECTURE ====================
    # Update all submodules in `unsub_repo`. This is critically important,
    # because it essentially 'rolls-back' commits made in submodules since the
    # last successful ki push in the main repository. Our `copy_repo()` call
    # does a `reset --hard` to the commit of the last push, but this does *not*
    # do an equivalent rollback for submodules. So they may contain new local
    # changes that we don't want. Calling `git submodule update` here checks
    # out the commit that *was* recorded in the submodule file at the ref of
    # the last push.
    unsub_root = F.working_dir(unsub_repo)
    with F.halo(text="Updating submodules in stage repositories..."):
        unsub_repo.git.submodule("update")
        last_push_repo.git.submodule("update")
    with F.halo(text=f"Unsubmoduling repository at '{unsub_root}'..."):
        unsub_repo = unsubmodule_repo(unsub_repo)
    patches_dir: ExtantDir = F.mkdtemp()
    with F.halo(text=f"Fetching from remote at '{remote_repo.working_dir}'..."):
        anki_remote.fetch()
        unsub_remote.fetch()
    with F.halo(text=f"Diffing 'HEAD' ~ 'FETCH_HEAD' in '{unsub_root}'..."):
        raw_unified_patch = unsub_repo.git.diff(["HEAD", "FETCH_HEAD"], binary=True)

    @beartype
    def unquote_diff_path(path: str) -> str:
        if len(path) <= 4:
            return path
        if path[0] == '"' and path[-1] == '"':
            path = path.lstrip('"').rstrip('"')
        if path[:2] in ("a/", "b/"):
            path = path[2:]
        return path

    # Construct patches for every file in the flattenened/unsubmoduled checkout
    # of the revision of the last successful `ki push`. Each patch is the diff
    # between the relevant file in the flattened (all submodules converted to
    # ordinary directories) repository and the same file in the Anki remote (a
    # fresh `ki clone` of the current database).
    patches: List[Patch] = []
    f = io.StringIO()
    with F.halo(text="Generating submodule patches..."):
        with redirect_stdout(f):
            for diff in whatthepatch.parse_patch(raw_unified_patch):
                a_path = unquote_diff_path(diff.header.old_path)
                b_path = unquote_diff_path(diff.header.new_path)
                if a_path == DEV_NULL:
                    a_path = b_path
                if b_path == DEV_NULL:
                    b_path = a_path
                patch = Patch(Path(a_path), Path(b_path), diff)
                patches.append(patch)

    # Construct a map that sends submodule relative roots, that is, the
    # relative path of a submodule root directory to the top-level root
    # directory of the ki repository, to `git.Repo` objects for each submodule.
    subrepos: Dict[Path, git.Repo] = {}
    for sm in last_push_repo.submodules:
        if sm.exists() and sm.module_exists():
            sm_repo: git.Repo = sm.module()
            sm_root: ExtantDir = F.working_dir(sm_repo)

            # Get submodule root relative to ki repository root.
            sm_rel_root: Path = sm_root.relative_to(last_push_root)
            subrepos[sm_rel_root] = sm_repo

            # Remove submodules directories from remote repo.
            halotext = f"Removing submodule directory '{sm_rel_root}' from remote..."
            if os.path.isdir(anki_remote_root / sm_rel_root):
                with F.halo(text=halotext):
                    remote_repo.git.rm(["-r", str(sm_rel_root)])

    if len(last_push_repo.submodules) > 0:
        with F.halo(text="Committing submodule directory removals..."):
            remote_repo.git.add(all=True)
            remote_repo.index.commit("Remove submodule directories.")

    # Apply patches within submodules.
    msg = "Applying patches:\n\n"
    patches_bar = tqdm(patches, ncols=TQDM_NUM_COLS)
    patches_bar.set_description("Patches")
    patched_submodules: Set[Path] = set()
    for patch in patches_bar:
        for sm_rel_root, sm_repo in subrepos.items():

            # TODO: We must also treat case where we moved a file into or out
            # of a submodule, but we just do this for now. In this case, we may
            # have `patch.a` not be relative to the submodule root (if we moved
            # a file into the sm dir), or vice-versa.
            a_in_submodule: bool = patch.a.is_relative_to(sm_rel_root)
            b_in_submodule: bool = patch.b.is_relative_to(sm_rel_root)

            if a_in_submodule and b_in_submodule:

                # Ignore the submodule 'file' itself.
                if sm_rel_root in (patch.a, patch.b):
                    continue

                # Hash the patch to use as a filename.
                blake2 = hashlib.blake2s()
                blake2.update(patch.diff.text.encode())
                patch_hash: str = blake2.hexdigest()
                patch_path: NoFile = F.test(patches_dir / patch_hash)

                # Strip trailing linefeeds from each line so that `git apply`
                # is happy on Windows (equivalent to running `dos2unix`).
                patch_path: ExtantFile = F.write(patch_path, patch.diff.text)
                if sys.platform == "win32":
                    with open(patch_path, "rb") as f:
                        with open(patch_path, "wb") as g:
                            g.write(f.read().replace(b"\r\n", b"\n"))

                # Number of leading path components to drop from diff paths.
                num_parts = len(sm_rel_root.parts) + 1

                # TODO: More tests are needed to make sure that the `git apply`
                # call is not flaky. In particular, we must treat new and
                # deleted files.
                #
                # Note that it is unnecessary to use `--3way` here, because
                # this submodule is supposed to represent a fast-forward from
                # the last successful push to the current state of the remote.
                # There should be no nontrivial merging involved.
                #
                # Then -p<n> flag tells `git apply` to drop the first n leading
                # path components from both diff paths. So if n is 2, we map
                # `a/dog/cat` -> `cat`.
                sm_repo.git.apply(
                    patch_path,
                    p=str(num_parts),
                    allow_empty=True,
                    verbose=True,
                )
                patched_submodules.add(sm_rel_root)
                msg += f"  `{patch.a}`\n"

    echo(f"Applied {len(patched_submodules)} patches within submodules.")
    for sm_rel_root in patched_submodules:
        echo(f"  Patched '{sm_rel_root}'")

    # Commit patches in submodules.
    with F.halo(text="Committing applied patches to submodules..."):
        for sm_repo in subrepos.values():
            sm_repo.git.add(all=True)
            sm_repo.index.commit(msg)

    # TODO: What if a submodule was deleted (or added) entirely?
    #
    # New commits in submodules within `last_push_repo` are be pulled into the
    # submodules within `kirepo.repo`. This is done by adding a remote pointing
    # to the patched submodule in each corresponding submodule in the main
    # repository, and then pulling from that remote. Then the remote is
    # deleted.
    for sm in kirepo.repo.submodules:
        if sm.exists() and sm.module_exists():
            sm_repo: git.Repo = sm.module()
            sm_rel_root: Path = F.working_dir(sm_repo).relative_to(kirepo.root)

            # Note that `subrepos` are the submodules of `last_push_repo`.
            if sm_rel_root in subrepos:
                remote_sm: git.Repo = subrepos[sm_rel_root]

                # TODO: What is contained in this branch that isn't already in
                # `BRANCH_NAME`?
                remote_sm.git.branch("upstream")

                # Simulate a `git merge --strategy=theirs upstream`.
                remote_sm.git.checkout(["-b", "tmp", "upstream"])
                remote_sm.git.merge(["-s", "ours", BRANCH_NAME])
                remote_sm.git.checkout(BRANCH_NAME)
                remote_sm.git.merge("tmp")
                remote_sm.git.branch(["-D", "tmp"])

                remote_target: ExtantDir = F.git_dir(remote_sm)
                sm_remote = sm_repo.create_remote(REMOTE_NAME, remote_target)
                git_pull(
                    REMOTE_NAME,
                    BRANCH_NAME,
                    F.working_dir(sm_repo),
                    False,
                    False,
                    False,
                    silent,
                )
                sm_repo.delete_remote(sm_remote)
                remote_sm.close()
            sm_repo.close()

    # Commit new submodules commits in `last_push_repo`.
    if len(patched_submodules) > 0:
        with F.halo(text=f"Committing new submodule commits to '{last_push_root}'..."):
            last_push_repo.git.add(all=True)
            last_push_repo.index.commit(msg)

    # Handle deleted files, preferring `theirs`.
    deletes = 0
    del_msg = "Remove files deleted in remote.\n\n"
    fetch_head = last_push_repo.commit("FETCH_HEAD")
    diff_index = last_push_repo.commit("HEAD").diff(fetch_head)
    for diff in diff_index.iter_change_type(GitChangeType.DELETED.value):

        # Don't remove gitmodules.
        if diff.a_path == GITMODULES_FILE:
            continue

        a_path: Path = F.test(last_push_root / diff.a_path)
        if isinstance(a_path, ExtantFile):
            last_push_repo.git.rm(diff.a_path)
            del_msg += f"Remove '{a_path}'\n"
            deletes += 1

    if deletes > 0:
        with F.halo(text=f"Committing remote deletions to '{last_push_root}'..."):
            last_push_repo.git.add(all=True)
            last_push_repo.index.commit(del_msg)

    # =================== NEW PULL ARCHITECTURE ====================

    echo(f"Copying blobs at HEAD='{sha}' to stage in '{last_push_root}'...")
    git_copy = F.copytree(F.git_dir(last_push_repo), F.test(F.mkdtemp() / "GIT"))
    last_push_repo.close()
    last_push_root: NoFile = F.rmtree(F.working_dir(last_push_repo))
    del last_push_repo
    remote_root: ExtantDir = F.working_dir(remote_repo)
    last_push_root: ExtantDir = F.copytree(remote_root, last_push_root)

    echo(f"Copying git history from '{sha}' to stage...")
    last_push_repo: git.Repo = M.repo(last_push_root)
    git_dir: NoPath = F.rmtree(F.git_dir(last_push_repo))
    del last_push_repo
    F.copytree(git_copy, F.test(git_dir))

    echo(f"Committing stage repository contents in '{last_push_root}'...")
    last_push_repo: git.Repo = M.repo(last_push_root)
    last_push_repo.git.add(all=True)
    last_push_repo.index.commit(f"Pull changes from repository at `{kirepo.root}`")

    # Create remote pointing to `last_push_repo` and pull into `repo`. Note
    # that this `git pull` may not always create a merge commit, because a
    # fast-forward only updates the branch pointer.
    last_push_remote = kirepo.repo.create_remote(REMOTE_NAME, last_push_repo.git_dir)
    kirepo.repo.git.config("pull.rebase", "false")
    git_pull(REMOTE_NAME, BRANCH_NAME, kirepo.root, False, False, False, silent)
    kirepo.repo.delete_remote(last_push_remote)

    # Append the hash of the collection to the hashes file, and raise an error
    # if the collection was modified while we were pulling changes.
    append_md5sum(kirepo.ki_dir, kirepo.col_file.name, md5sum, silent=True)
    if F.md5(kirepo.col_file) != md5sum:
        raise CollectionChecksumError(kirepo.col_file)


# PUSH


@ki.command()
@click.option("--verbose", "-v", is_flag=True, help="Print more output.")
@beartype
def push(verbose: bool = False) -> PushResult:
    """
    Push a ki repository into a .anki2 file.

    Consists of the following operations:

        - Clone collection into a temp directory
        - Delete all files and folders except `.git/`
        - Copy into this temp directory all files and folders from the current
          repository checked out at HEAD (excluding `.git*` stuff).
        - Unsubmodule repo
        - Add and commit everything
        - Take diff from `HEAD~1` to `HEAD`.

    Returns
    -------
    PushResult
        An `Enum` indicating whether the push was nontrivial.

    Raises
    ------
    UpdatesRejectedError
        If the user needs to pull remote changes first.
    """
    if PROFILE:
        # pylint: disable=import-outside-toplevel
        from pyinstrument import Profiler

        # pylint: enable=import-outside-toplevel
        profiler = Profiler()
        profiler.start()

    pp.install_extras(exclude=["ipython", "django", "ipython_repr_pretty"])

    # Check that we are inside a ki repository, and load collection.
    cwd: ExtantDir = F.cwd()
    kirepo: KiRepo = M.kirepo(cwd)
    con: sqlite3.Connection = lock(kirepo.col_file)

    md5sum: str = F.md5(kirepo.col_file)
    hashes: List[str] = kirepo.hashes_file.read_text(encoding="UTF-8").split("\n")
    hashes = list(filter(lambda l: l != "", hashes))
    if md5sum not in hashes[-1]:
        raise UpdatesRejectedError(kirepo.col_file)

    # =================== NEW PUSH ARCHITECTURE ====================
    with F.halo("Initializing stage repository..."):
        head: KiRepoRef = M.head_kirepo_ref(kirepo)
        head_kirepo: KiRepo = copy_kirepo(head, f"{HEAD_SUFFIX}-{md5sum}")
        remote_root: EmptyDir = F.mksubdir(F.mkdtemp(), REMOTE_SUFFIX / md5sum)

    msg = f"Fetch changes from collection '{kirepo.col_file}' with md5sum '{md5sum}'"
    remote_repo, _ = _clone(
        kirepo.col_file,
        remote_root,
        msg,
        silent=True,
        verbose=verbose,
    )

    with F.halo(f"Copying blobs at HEAD='{head.sha}' to stage in '{remote_root}'..."):
        git_copy = F.copytree(F.git_dir(remote_repo), F.test(F.mkdtemp() / "GIT"))
        remote_repo.close()
        remote_root: NoFile = F.rmtree(F.working_dir(remote_repo))
        del remote_repo
        remote_root: ExtantDir = F.copytree(head_kirepo.root, remote_root)

    with F.halo(f"Flattening stage repository submodules in '{remote_root}'..."):
        remote_repo: git.Repo = unsubmodule_repo(M.repo(remote_root))
        git_dir: NoPath = F.rmtree(F.git_dir(remote_repo))
        del remote_repo
        F.copytree(git_copy, F.test(git_dir))

    with F.halo(f"Committing stage repository contents in '{remote_root}'..."):
        remote_repo: git.Repo = M.repo(remote_root)
        remote_repo.git.add(all=True)
        remote_repo.index.commit(f"Pull changes from repository at `{kirepo.root}`")
    # =================== NEW PUSH ARCHITECTURE ====================

    # Read grammar.
    # TODO:! Should we assume this always exists? A nice error message should
    # be printed on initialization if the grammar file is missing. No
    # computation should be done, and none of the click commands should work.
    grammar_path = Path(__file__).resolve().parent / "grammar.lark"
    grammar = grammar_path.read_text(encoding="UTF-8")

    # Instantiate parser.
    parser = Lark(grammar, start="note", parser="lalr")
    transformer = NoteTransformer()

    deltas: List[Union[Delta, Warning]] = diff2(remote_repo, parser, transformer)

    # Map model names to models.
    models: Dict[str, Notetype] = get_models_recursively(head_kirepo, silent=True)

    result: PushResult = push_deltas(
        deltas,
        models,
        kirepo,
        md5sum,
        parser,
        transformer,
        head_kirepo,
        con,
        verbose,
    )

    if PROFILE:
        profiler.stop()
        s = profiler.output_html()
        Path("ki_push_profile.html").resolve().write_text(s, encoding="UTF-8")

    return result


@beartype
def push_deltas(
    deltas: List[Union[Delta, Warning]],
    models: Dict[str, Notetype],
    kirepo: KiRepo,
    md5sum: str,
    parser: Lark,
    transformer: NoteTransformer,
    head_kirepo: KiRepo,
    con: sqlite3.Connection,
    verbose: bool,
) -> PushResult:
    """Push a list of `Delta`s to an Anki collection."""
    warnings: List[Warning] = [delta for delta in deltas if isinstance(delta, Warning)]
    deltas: List[Delta] = [delta for delta in deltas if isinstance(delta, Delta)]

    # Display warnings from diff procedure.
    for warning in warnings:
        if verbose or type(warning) not in WARNING_IGNORE_LIST:
            click.secho(str(warning), fg="yellow")
    warnings = []

    # If there are no changes, quit.
    if len(set(deltas)) == 0:
        echo("ki push: up to date.")
        return PushResult.UP_TO_DATE

    echo(f"Pushing to '{kirepo.col_file}'")
    echo(f"Computed md5sum: {md5sum}")
    echo(f"Verified md5sum matches latest hash in '{kirepo.hashes_file}'")

    # Copy collection to a temp directory.
    temp_col_dir: ExtantDir = F.mkdtemp()
    new_col_file = temp_col_dir / kirepo.col_file.name
    col_name: str = kirepo.col_file.name
    new_col_file: ExtantFile = F.copyfile(kirepo.col_file, temp_col_dir, col_name)

    head: RepoRef = M.head_repo_ref(kirepo.repo)
    echo(f"Generating local .anki2 file from latest commit: {head.sha}")
    echo(f"Writing changes to '{new_col_file}'...")

    # Open collection, holding cwd constant (otherwise Anki changes it).
    cwd: ExtantDir = F.cwd()
    col: Collection = M.collection(new_col_file)
    F.chdir(cwd)

    # Add new models to the collection.
    add_models(col, models)

    # Gather logging statements to display.
    log: List[str] = []

    # Stash both unstaged and staged files (including untracked).
    kirepo.repo.git.stash(include_untracked=True, keep_index=True)
    kirepo.repo.git.reset("HEAD", hard=True)

    # Display table of note change type counts.
    echo_note_change_types(deltas)

    is_delete = lambda d: d.status == GitChangeType.DELETED

    bar = tqdm(deltas, ncols=TQDM_NUM_COLS)
    bar.set_description("Deltas")
    for delta in bar:

        # Parse the file at `delta.path` into a `DeckNote`, and
        # add/edit/delete in collection.
        decknote = parse_markdown_note(parser, transformer, delta)

        if is_delete(delta):
            col.remove_notes([decknote.nid])
            continue

        # TODO: If relevant prefix of sort field has changed, we should
        # regenerate the file. Recall that the sort field is used to determine
        # the filename. If the content of the sort field has changed, then we
        # may need to update the filename.
        colnote, note_warnings = push_decknote_to_anki(col, decknote)
        log += regenerate_note_file(colnote, kirepo.root, delta.relpath)
        warnings += note_warnings

    if verbose:
        for msg in log:
            click.secho(str(msg), fg="yellow")

    num_displayed: int = 0
    for warning in warnings:
        if verbose or type(warning) not in WARNING_IGNORE_LIST:
            click.secho(str(warning), fg="yellow")
            num_displayed += 1
    num_suppressed: int = len(warnings) - num_displayed
    echo(f"Warnings suppressed: {num_suppressed} (show with '--verbose')")

    # Commit nid reassignments.
    echo(f"Reassigned {len(log)} nids.")
    if len(log) > 0:
        msg = "Generated new nid(s).\n\n" + "\n".join(log)

        # Commit in all submodules (doesn't support recursing yet).
        for sm in kirepo.repo.submodules:
            subrepo: git.Repo = sm.module()
            subrepo.git.add(all=True)
            subrepo.index.commit(msg)

        # Commit in main repository.
        kirepo.repo.git.add(all=True)
        _ = kirepo.repo.index.commit(msg)

    # It is always safe to save changes to the DB, since the DB is a copy.
    col.close(save=True)

    # Backup collection file and overwrite collection.
    backup(kirepo)
    new_col_file = F.copyfile(new_col_file, F.parent(kirepo.col_file), col_name)
    echo(f"Overwrote '{kirepo.col_file}'")

    # Add media files to collection.
    col: Collection = M.collection(kirepo.col_file)
    media_files = F.rglob(head_kirepo.root, MEDIA_FILE_RECURSIVE_PATTERN)

    # TODO: Write an analogue of `Anki2Importer._mungeMedia()` that does
    # deduplication, and fixes fields for notes that reference that media.
    bar = tqdm(media_files, ncols=TQDM_NUM_COLS, disable=False)
    bar.set_description("Media")
    for media_file in bar:

        # Add (and possibly rename) media paths.
        _: str = col.media.add_file(media_file)

    col.close(save=True)

    # Append to hashes file.
    new_md5sum = F.md5(new_col_file)
    append_md5sum(kirepo.ki_dir, new_col_file.name, new_md5sum, silent=False)

    # Update the commit SHA of most recent successful PUSH.
    head: RepoRef = M.head_repo_ref(kirepo.repo)
    kirepo.last_push_file.write_text(head.sha)

    # Unlock Anki SQLite DB.
    unlock(con)
    return PushResult.NONTRIVIAL
