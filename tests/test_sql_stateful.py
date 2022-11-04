"""Tests for SQLite Lark grammar."""
from __future__ import annotations

import shutil
import random
import tempfile
import subprocess
from pathlib import Path

import prettyprinter as pp
from loguru import logger
from libfaketime import fake_time, reexec_if_needed

from beartype import beartype
from beartype.typing import Set, List

import hypothesis.strategies as st
from hypothesis import settings, Verbosity
from hypothesis.stateful import (
    rule,
    precondition,
    RuleBasedStateMachine,
)

# pylint: disable=unused-import
import anki.collection

# pylint: enable=unused-import
from anki.decks import DeckNameId
from anki.models import NotetypeDict
from anki.collection import Collection, Note

import ki.functional as F
from ki.sqlite import SQLiteTransformer
from tests.test_parser import get_parser
from tests.test_ki import get_test_collection


# pylint: disable=too-many-lines, missing-function-docstring

reexec_if_needed()

DEFAULT_DID = 1

EMPTY = get_test_collection("empty")
logger.debug(F.md5(EMPTY.col_file))


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
    k = 0

    def __init__(self):
        super().__init__()
        logger.debug(f"Starting test {AnkiCollection.k}...")
        AnkiCollection.k += 1
        random.seed(0)
        self.freeze = True
        if self.freeze:
            self.freezer = fake_time("2022-05-01 00:00:00")
            self.freezer.start()
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
        nt: NotetypeDict = data.draw(st.sampled_from(self.col.models.all()), label="nt")
        note: Note = self.col.new_note(nt)
        n: int = len(self.col.models.field_names(nt))
        characters: st.SearchStrategy = st.characters(
            blacklist_characters=["\x1f"], blacklist_categories=["Cs"]
        )
        fields: st.SearchStrategy = st.text(alphabet=characters)
        fieldlists: st.SearchStrategy = st.lists(fields, min_size=n, max_size=n)
        note.fields = data.draw(fieldlists, label="add note: fields")
        dids = list(map(lambda d: d.id, self.col.decks.all_names_and_ids()))
        did: int = data.draw(st.sampled_from(dids), label="add note: did")
        self.col.add_note(note, did)

    @precondition(lambda self: len(set(self.col.find_notes(query=""))) > 0)
    @rule(data=st.data())
    def remove_note(self, data: st.DataObject) -> None:
        """Remove a note randomly selected note from the collection."""
        nids = list(self.col.find_notes(query=""))
        nid = data.draw(st.sampled_from(nids), label="rm note: nid")
        self.col.remove_notes([nid])

    @rule(data=st.data())
    def add_deck(self, data: st.DataObject) -> None:
        """Add a new deck by creating a child node."""
        deck_name_ids: List[DeckNameId] = list(self.col.decks.all_names_and_ids())
        parent: DeckNameId = data.draw(st.sampled_from(deck_name_ids), label="parent")
        names = set(map(lambda x: x[0], self.col.decks.children(parent.id)))
        names_st = st.text(min_size=1).filter(lambda s: s not in names)
        name = data.draw(names_st, label="add deck: deckname")
        if self.freeze:
            self.freezer.tick()
        _ = self.col.decks.id(f"{parent.name}::{name}", create=True)

    @precondition(lambda self: len(list(self.col.decks.all_names_and_ids())) >= 2)
    @rule(data=st.data())
    def remove_deck(self, data: st.DataObject) -> None:
        """Remove a deck if one exists."""
        dids: Set[int] = set(map(lambda d: d.id, self.col.decks.all_names_and_ids()))
        dids -= {DEFAULT_DID}
        did: int = data.draw(st.sampled_from(list(dids)), "rm deck: did")
        _ = self.col.decks.remove([did])

    def teardown(self) -> None:
        """Cleanup the state of the system."""
        did = self.col.decks.id("dummy", create=True)
        self.col.decks.remove([did])
        self.col.close(save=True)
        assert str(self.path) != str(EMPTY.col_file)
        assert F.md5(self.path) != F.md5(EMPTY.col_file)
        p = subprocess.run(
            ["sqldiff", "--table", "notes", str(EMPTY.col_file), str(self.path)],
            capture_output=True,
            check=True,
        )
        block = p.stdout.decode()
        logger.debug(block)
        parser = get_parser(filename="sqlite.lark", start="diff")
        transformer = SQLiteTransformer()
        tree = parser.parse(block)
        stmts = transformer.transform(tree)
        # logger.debug(pp.pformat(stmts))

        shutil.rmtree(self.tempd)
        if self.freeze:
            self.freezer.stop()


AnkiCollection.TestCase.settings = settings(
    max_examples=100,
    stateful_step_count=30,
    verbosity=Verbosity.normal,
)
TestAnkiCollection = AnkiCollection.TestCase
