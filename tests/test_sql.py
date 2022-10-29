"""Tests for SQLite Lark grammar."""
from __future__ import annotations

import time
import shutil
import tempfile
import subprocess
from pathlib import Path

import prettyprinter as pp
from loguru import logger

from beartype import beartype
from beartype.typing import Iterable, Sequence, List

import hypothesis.strategies as st
from hypothesis import given, settings, assume, Verbosity
from hypothesis.stateful import (
    rule,
    multiple,
    consumes,
    initialize,
    precondition,
    Bundle,
    RuleBasedStateMachine,
)
from hypothesis.strategies import composite

import anki.collection
from anki.decks import DeckNameId, DeckTreeNode
from anki.models import NotetypeNameId, NotetypeDict
from anki.collection import Collection, Note

import ki.functional as F
from ki.sqlite import SQLiteTransformer
from tests.test_parser import get_parser, debug_lark_error
from tests.test_ki import get_test_collection


# pylint: disable=too-many-lines, missing-function-docstring

BAD_ASCII_CONTROLS = ["\0", "\a", "\b", "\v", "\f"]

DECK_CHAR = r"(?!::)[^\0\x07\x08\x0b\x0c\"\r\n]"
DECK_NON_SPACE_CHAR = r"(?!::)[^\0\x07\x08\x0b\x0c\"\s]"
DECK_COMPONENT_NAME = (
    f"({DECK_NON_SPACE_CHAR})+(({DECK_CHAR})+({DECK_NON_SPACE_CHAR})+){0,}"
)

DEFAULT_DID = 1

EMPTY = get_test_collection("empty")
logger.debug(F.md5(EMPTY.col_file))


@composite
def new_deck_fullnames(
    draw, collections: Bundle, parents: st.SearchStrategy[int]
) -> st.SearchStrategy[str]:
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
        super().__init__()
        self.tempd = Path(tempfile.mkdtemp())
        self.path = F.chk(self.tempd / "collection.anki2")
        self.path = F.copyfile(EMPTY.col_file, self.path)
        self.col = Collection(self.path)
        if not self.col.db:
            self.col.reopen()

    @precondition(lambda self: len(list(self.col.decks.all_names_and_ids())) >= 1)
    @rule(data=st.data())
    def add_note(self, data: st.DataObject) -> None:
        """Add a new note with random fields."""
        nt: NotetypeDict = data.draw(st.sampled_from(self.col.models.all()))
        note: Note = self.col.new_note(nt)
        n: int = len(self.col.models.field_names(nt))
        note.fields = data.draw(st.lists(st.text(), min_size=n, max_size=n))
        dids = list(map(lambda d: d.id, self.col.decks.all_names_and_ids()))
        did: int = data.draw(st.sampled_from(dids))
        self.col.add_note(note, did)

    @rule(data=st.data())
    def add_deck(self, data: st.DataObject) -> None:
        """Add a new deck by creating a child node."""
        dids = list(map(lambda d: d.id, self.col.decks.all_names_and_ids()))
        root: DeckTreeNode = self.col.decks.deck_tree()
        parent_did: int = data.draw(st.sampled_from(dids))
        node = self.col.decks.find_deck_in_tree(root, parent_did)
        names: Set[str] = set(map(lambda c: c.name, node.children))
        name = data.draw(st.text(min_size=1).filter(lambda s: s not in names))
        _ = self.col.decks.id(f"{node.name}::{name}", create=True)

    @precondition(lambda self: len(list(self.col.decks.all_names_and_ids())) >= 2)
    @rule(data=st.data())
    def remove_deck(self, data: st.DataObject) -> None:
        """Remove a deck if one exists."""
        dids: Set[int] = set(map(lambda d: d.id, self.col.decks.all_names_and_ids()))
        dids -= {DEFAULT_DID}
        did: int = data.draw(st.sampled_from(list(dids)))
        _ = self.col.decks.remove([did])

    def teardown(self) -> None:
        """Cleanup the state of the system."""
        did = self.col.decks.id("dummy", create=True)
        self.col.decks.remove([did])
        self.col.save()
        self.col.close(save=True)
        assert str(self.path) != str(EMPTY.col_file)
        assert F.md5(self.path) != F.md5(EMPTY.col_file)
        p = subprocess.run(
            ["sqldiff", "--table", "notes", str(EMPTY.col_file), str(self.path)],
            capture_output=True,
        )
        logger.debug(p.stdout.decode())
        logger.debug(p.stderr.decode())
        shutil.rmtree(self.tempd)


AnkiCollection.TestCase.settings = settings(max_examples=20, verbosity=Verbosity.debug)
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
