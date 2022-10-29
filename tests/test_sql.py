"""Tests for SQLite Lark grammar."""
from __future__ import annotations

import time
import shutil
import tempfile
from pathlib import Path

import prettyprinter as pp
from loguru import logger

from beartype import beartype
from beartype.typing import Iterable, Sequence, List

import hypothesis.strategies as st
from hypothesis import given, settings
from hypothesis.stateful import (
    RuleBasedStateMachine,
    Bundle,
    rule,
    initialize,
    multiple,
    consumes,
)

import anki.collection
from anki.decks import DeckNameId, DeckTreeNode
from anki.models import NotetypeNameId, NotetypeDict
from anki.collection import Collection, Note

import ki.functional as F
from ki.sqlite import SQLiteTransformer
from tests.test_parser import get_parser, debug_lark_error


# pylint: disable=too-many-lines, missing-function-docstring

BAD_ASCII_CONTROLS = ["\0", "\a", "\b", "\v", "\f"]

DECK_CHAR = r"(?!::)[^\0\x07\x08\x0b\x0c\"\r\n]"
DECK_NON_SPACE_CHAR = r"(?!::)[^\0\x07\x08\x0b\x0c\"\s]"
DECK_COMPONENT_NAME = (
    f"({DECK_NON_SPACE_CHAR})+(({DECK_CHAR})+({DECK_NON_SPACE_CHAR})+){0,}"
)

ROOT_DID = 1

tempd: Path = Path(tempfile.mkdtemp())
emptyd: Path = Path(tempfile.mkdtemp())
PATH = tempd / "collection.anki2"
EMPTY = emptyd / "empty.anki2"
COL = Collection(PATH)
COL.close()
shutil.copyfile(PATH, EMPTY)


@st.composite
def new_deck_fullnames(draw, collections: Bundle, parents: st.SearchStrategy[int]) -> st.SearchStrategy[str]:
    col = draw(collections)
    parent = draw(parents)
    root: DeckTreeNode = col.decks.deck_tree()
    node = col.decks.find_deck_in_tree(root, parent)
    names: Iterable[str] = map(lambda c: c.name, node.children)
    pattern: str = f"^(?!{'$|.join(names)'}$)"
    name = draw(st.from_regex(pattern))
    return f"{node.name}::{name}"


class AnkiCollection(RuleBasedStateMachine):
    """
    A state machine for testing `sqldiff` output parsing.

    Operation classes
    =================
    * notes
    * cards
    * decks
    * notetypes
    * tags
    * media

    Notes
    -----
    Add, edit, change notetype, move, delete

    Cards
    -----
    Move

    Decks
    -----
    Add, rename, move, delete

    Notetypes
    ---------
    Add, edit (see below), delete

    Notetype fields
    ---------------
    Add, delete, reposition, rename, set sort index

    Notetype templates
    ------------------
    Add, delete, reposition

    Tags
    ----
    Add, rename, reparent, delete

    Media
    -----
    Add, delete
    """
    # pylint: disable=no-self-use

    def __init__(self):
        t1 = time.time()

        super().__init__()
        self.col = COL
        self.col.reopen()

        logger.debug(time.time() - t1)

    dids = Bundle("dids")
    notes = Bundle("notes")
    notetypes = Bundle("notetypes")
    collections = Bundle("collections")

    @initialize(target=collections)
    @beartype
    def init_collection(self) -> Collection:
        return self.col

    @initialize(target=notetypes)
    @beartype
    def init_notetypes(self) -> Iterable[NotetypeDict]:
        name_ids: Sequence[NotetypeNameId] = self.col.models.all_names_and_ids()
        models = list(map(lambda m: self.col.models.get(m.id), name_ids))
        return multiple(*models)

    @initialize(target=dids)
    @beartype
    def init_decks(self) -> Iterable[int]:
        name_ids: Sequence[DeckNameId] = self.col.decks.all_names_and_ids()
        dids = list(filter(lambda d: d != ROOT_DID, map(lambda m: m.id, name_ids)))
        return multiple(*dids)

    @rule(notetype=notetypes, did=dids, target=notes)
    @beartype
    def add_note(self, notetype: NotetypeDict, did: int) -> Note:
        note: Note = self.col.new_note(notetype)
        self.col.add_note(note, did)
        return note

    @rule(note=notes, fields=st.lists(st.text(), min_size=100, max_size=100))
    @beartype
    def edit_note(self, note: Note, fields: List[str]) -> None:
        n = len(note.fields)
        note.fields = fields[:n]

    @rule(note=consumes(notes))
    @beartype
    def delete_note(self, note: Note) -> None:
        self.col.remove_notes([note.id])

    @rule(note=notes, did=dids)
    @beartype
    def move_note(self, note: Note, did: int) -> None:
        self.col.set_deck(note.card_ids(), did)

    @rule(
        fullname=new_deck_fullnames(
            collections=collections,
            parents=st.one_of(dids, st.just(ROOT_DID)),
        ),
        target=dids,
    )
    @beartype
    def add_deck(self, fullname: str) -> int:
        return self.col.decks.id(fullname, create=True)

    @rule(dids=st.lists(dids, min_size=2, max_size=2, unique=True))
    @beartype
    def reparent_deck(self, dids: int) -> None:
        did, parent = dids
        self.col.decks.reparent([did], parent)

    @rule(did=consumes(dids))
    @beartype
    def remove_deck(self, did: int) -> None:
        self.col.decks.remove([did])

    def teardown(self) -> None:
        try:
            self.col.close()
        finally:
            shutil.copyfile(EMPTY, PATH)


# AnkiCollection.TestCase.settings = settings(max_examples=100)
TestAnkiCollection = AnkiCollection.TestCase

BLOCK = r"""
DELETE FROM notes WHERE id=1645010162168;
DELETE FROM notes WHERE id=1645027705329;
INSERT INTO notes(id,guid,mid,mod,usn,tags,flds,sfld,csum,flags,data) VALUES(1651202347045,'D(l.-iAXR1',1651202298367,1651202347,-1,'','a'||X'1f'
        ||'b[sound:1sec.mp3]','a',2264392759,0,'');
INSERT INTO notes(id,guid,mid,mod,usn,tags,flds,sfld,csum,flags,data) VALUES(1651232832286,'H0%>O*C~M!',1651232485378,1651232832,-1,'','Who introduced the notion of a direct integral in functional analysis?'||X'1f'
        ||'John von Neumann','Who introduced the notion of a direct integral in functional analysis?',2707929287,0,'');
"""

UPDATE = r"""
UPDATE notes SET mod=1645221606, flds='aa'||X'1f'
||'bb', sfld='aa', csum=3771269976 WHERE id=1645010162168;
DELETE FROM notes WHERE id=1645027705329;
INSERT INTO notes(id,guid,mid,mod,usn,tags,flds,sfld,csum,flags,data) VALUES(1645222430007,'f}:^>jzMjG',1645010146011,1645222430,-1,'','f'||X'1f'
||'g','f',1242175777,0,'');
"""


def test_basic():
    parser = get_parser(filename="sqlite.lark", start="diff")
    parser.parse(BLOCK)


def test_transformer():
    parser = get_parser(filename="sqlite.lark", start="diff")
    tree = parser.parse(BLOCK)
    transformer = SQLiteTransformer()
    _ = transformer.transform(tree)


def test_transformer_on_update_commands():
    parser = get_parser(filename="sqlite.lark", start="diff")
    tree = parser.parse(UPDATE)
    transformer = SQLiteTransformer()
    _ = transformer.transform(tree)
