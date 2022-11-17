"""Tests for SQLite Lark grammar."""
import shutil
import random
import tempfile
import subprocess
from pathlib import Path
from functools import reduce, partial
from itertools import starmap
from collections import namedtuple

import prettyprinter as pp
from loguru import logger
from libfaketime import fake_time, reexec_if_needed
from pyinstrument import Profiler

from beartype import beartype
from beartype.typing import Set, List, TypeVar, Iterable, Callable

import hypothesis.strategies as st
from hypothesis import settings, Verbosity
from hypothesis.stateful import (
    rule,
    precondition,
    RuleBasedStateMachine,
)
from hypothesis.strategies import composite, SearchStrategy

# pylint: disable=unused-import
import anki.collection

# pylint: enable=unused-import
from anki.decks import DeckNameId
from anki.errors import CardTypeError
from anki.models import ModelManager, NotetypeDict, TemplateDict
from anki.collection import Collection, Note

import ki.functional as F
from ki.types import File
from ki.sqlite import SQLiteTransformer, Delete, Update, Table
from tests.schema import checkpoint_anki2
from tests.test_ki import get_test_collection
from tests.test_parser import get_parser

T = TypeVar("T")

# pylint: disable=too-many-lines, missing-function-docstring, too-many-locals

reexec_if_needed()

ROOT_DID = 0
DEFAULT_DID = 1

EMPTY = get_test_collection("empty")
INITIAL = checkpoint_anki2(EMPTY.col_file, "initial")

pp.install_extras(exclude=["django", "ipython", "ipython_repr_pretty"])

PARSER = get_parser(filename="sqlite.lark", start="diff")
TRANSFORMER = SQLiteTransformer()

@composite
@beartype
def fnames(draw: Callable[[SearchStrategy[T]], T]) -> str:
    """Field names."""
    fchars = st.characters(
        blacklist_characters=[":", "{", "}", '"'],
        blacklist_categories=["Cc", "Cs", "Lo", "Lm", "Sk", "Po", "So", "Sm", "Mn"],
        max_codepoint=0x7F,
    )

    # First chars for field names.
    endchars = st.characters(
        blacklist_characters=["^", "/", "#", ":", "{", "}", '"'],
        blacklist_categories=["Zs", "Zl", "Zp", "Cc", "Cs", "Lo"],
        max_codepoint=0x7F,
    )
    pith = draw(st.text(alphabet=fchars, min_size=0), "add nt: fname pith")
    end = draw(st.one_of(st.just(""), endchars), "add nt: fname end")
    start = draw(endchars, "add nt: fname start")
    return start + pith + end


@beartype
def fmts(data: st.DataObject, fieldnames: List[str], text: str) -> str:
    """A card side template with optional embedded fields."""
    # Placeholders for field contents to be embedded in the front or
    # back of card templates (called ``qfmt`` and ``afmt``).
    frepls: List[str] = list(map(lambda s: "{{" + s + "}}", fieldnames))

    # Sample some of the fields to insert into ``text``.
    replsets = st.sets(st.sampled_from(frepls), min_size=1)
    placeholders = data.draw(replsets, "fmts: placeholders")

    # Sample locations at which to insert the placeholders.
    locs = st.integers(min_value=0, max_value=len(text))
    n = len(placeholders)
    loclists = st.lists(locs, min_size=n, max_size=n)
    ks: Iterable[int] = reversed(sorted(data.draw(loclists, "fmts: frepl locs")))

    # Pair each field placeholder with a location and interpolate.
    Placeholder = namedtuple("Placeholder", ["loc", "text"])
    xs: List[Placeholder] = starmap(Placeholder, zip(ks, placeholders))
    result = reduce(lambda s, x: s[: x.loc] + x.text + s[x.loc :], xs, text)
    assert len(result) > 0
    assert "{{" in result
    assert "}}" in result
    return result


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
        self.profile = False
        self.freeze = True
        if self.profile:
            self.profiler = Profiler()
            self.profiler.start()
        super().__init__()
        logger.debug(f"Starting test {AnkiCollection.k}...")
        AnkiCollection.k += 1
        random.seed(0)
        if self.freeze:
            self.freezer = fake_time("2022-05-01 00:00:00")
            self.freezer.start()
        self.tempd = Path(tempfile.mkdtemp())
        self.path = F.chk(self.tempd / "collection.anki2")
        self.path = F.copyfile(EMPTY.col_file, self.path)
        self.checkpoint = None
        self.col = Collection(self.path)
        if not self.col.db:
            self.col.reopen()

        characters: SearchStrategy = st.characters(
            blacklist_characters=["\x1f"],
            blacklist_categories=["Cs"],
        )
        self.fields: SearchStrategy = st.text(alphabet=characters)
        self.saved = False

    @precondition(lambda self: len(list(self.col.decks.all_names_and_ids())) >= 1)
    @rule(data=st.data())
    def add_note(self, data: st.DataObject) -> None:
        """Add a new note with random fields."""
        nt: NotetypeDict = data.draw(st.sampled_from(self.col.models.all()), "nt")
        note: Note = self.col.new_note(nt)
        n: int = len(self.col.models.field_names(nt))
        fieldlists: SearchStrategy = st.lists(self.fields, min_size=n, max_size=n)
        note.fields = data.draw(fieldlists, "add note: fields")
        dids = list(map(lambda d: d.id, self.col.decks.all_names_and_ids()))
        did: int = data.draw(st.sampled_from(dids), "add note: did")
        self.col.add_note(note, did)

    @precondition(lambda self: self.col.note_count() >= 1)
    @rule(data=st.data())
    def edit_note(self, data: st.DataObject) -> None:
        """Edit a note's fields."""
        nids = list(self.col.find_notes(query=""))
        nid = data.draw(st.sampled_from(nids), "edit note: nid")
        note: Note = self.col.get_note(nid)
        n: int = len(self.col.models.field_names(note.note_type()))
        fieldlists: SearchStrategy = st.lists(self.fields, min_size=n, max_size=n)
        note.fields = data.draw(fieldlists, "edit note: fields")

    @precondition(lambda self: self.col.note_count() >= 1)
    @rule(data=st.data())
    def change_notetype(self, data: st.DataObject) -> None:
        """Change a note's notetype."""
        nids = list(self.col.find_notes(query=""))
        nid = data.draw(st.sampled_from(nids), "chg nt: nid")
        note: Note = self.col.get_note(nid)
        nt: NotetypeDict = data.draw(st.sampled_from(self.col.models.all()), "nt")
        old: NotetypeDict = note.note_type()
        items = map(lambda x: (x[0], None), self.col.models.field_map(nt).values())
        self.col.models.change(old, [nid], nt, fmap=dict(items), cmap=None)

    @precondition(lambda self: len(list(self.col.decks.all_names_and_ids())) >= 3)
    @precondition(lambda self: self.col.card_count() >= 1)
    def move_card(self, data: st.DataObject) -> None:
        """Move a card to a (possibly) different deck."""
        cids = list(self.col.find_notes(query=""))
        cid = data.draw(st.sampled_from(cids), "mv card: cid")
        old: int = self.col.decks.for_card_ids([cid])[0]
        dids: Set[int] = set(map(lambda d: d.id, self.col.decks.all_names_and_ids()))
        dids -= {DEFAULT_DID, old}
        new: int = data.draw(st.sampled_from(list(dids)), "mv card: did")
        self.col.set_deck([cid], deck_id=new)

    @precondition(lambda self: self.col.note_count() >= 1)
    @rule(data=st.data())
    def remove_note(self, data: st.DataObject) -> None:
        """Remove a note randomly selected note from the collection."""
        nids = list(self.col.find_notes(query=""))
        nid = data.draw(st.sampled_from(nids), "rm note: nid")
        self.col.remove_notes([nid])

    @rule(data=st.data())
    def add_deck(self, data: st.DataObject) -> None:
        """Add a new deck by creating a child node."""
        deck_name_ids: List[DeckNameId] = list(self.col.decks.all_names_and_ids())
        parent: DeckNameId = data.draw(st.sampled_from(deck_name_ids), "parent")
        names = set(map(lambda x: x[0], self.col.decks.children(parent.id)))
        names_st = st.text(min_size=1).filter(lambda s: s not in names)
        name = data.draw(names_st, "add deck: deckname")
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

    @precondition(lambda self: len(list(self.col.decks.all_names_and_ids())) >= 2)
    @rule(data=st.data())
    def rename_deck(self, data: st.DataObject) -> None:
        """Rename a deck."""
        dids: Set[int] = set(map(lambda d: d.id, self.col.decks.all_names_and_ids()))
        dids -= {DEFAULT_DID}
        did: int = data.draw(st.sampled_from(list(dids)), "rename deck: did")
        name: str = data.draw(st.text(), "rename deck: name")
        self.col.decks.rename(did, name)

    @precondition(lambda self: len(list(self.col.decks.all_names_and_ids())) >= 2)
    @rule(data=st.data())
    def reparent_deck(self, data: st.DataObject) -> None:
        """Move a deck."""
        dids: Set[int] = set(map(lambda d: d.id, self.col.decks.all_names_and_ids()))
        srcs = dids - {DEFAULT_DID}
        did: int = data.draw(st.sampled_from(list(srcs)), "mv deck: did")
        dsts = {ROOT_DID} | dids - {did}
        dst: int = data.draw(st.sampled_from(list(dsts)), "mv deck: dst")
        self.col.decks.reparent([did], dst)

    @rule(data=st.data())
    def add_notetype(self, data: st.DataObject) -> None:
        """Add a new notetype."""
        mm: ModelManager = self.col.models
        nchars = st.characters(blacklist_characters=['"'], blacklist_categories=["Cs"])
        name: str = data.draw(st.text(min_size=1, alphabet=nchars), "add nt: name")
        nt: NotetypeDict = mm.new(name)
        field_name_lists = st.lists(fnames(), min_size=1, unique_by=lambda s: s.lower())
        fieldnames = data.draw(field_name_lists, "add nt: fieldnames")
        nt["flds"] = list(map(mm.new_field, fieldnames))

        tmplnames = st.lists(st.text(alphabet=nchars, min_size=1), min_size=1)
        tnames: List[str] = data.draw(tmplnames, "add nt: tnames")
        n = len(tnames)
        txts = st.text(
            alphabet=st.characters(
                blacklist_characters=["{", "}"], blacklist_categories=["Cs"]
            )
        )
        textlists = st.lists(txts, min_size=n, max_size=n, unique=True)
        qtxts = data.draw(textlists, "add nt: qtxts")
        atxts = data.draw(textlists, "add nt: atxts")
        qfmts = list(map(partial(fmts, data, fieldnames), qtxts))
        afmts = list(map(partial(fmts, data, fieldnames), atxts))
        tmpls: List[TemplateDict] = list(map(mm.new_template, tnames))
        triples = zip(tmpls, qfmts, afmts)
        tmpls = list(starmap(lambda t, q, a: t | {"qfmt": q, "afmt": a}, triples))
        nt["tmpls"] = tmpls
        try:
            mm.add_dict(nt)
        except CardTypeError as err:
            logger.debug(err)
            logger.debug(err.help_page)
            raise err

    @rule(data=st.data())
    def remove_notetype(self, data: st.DataObject) -> None:
        """Remove a notetype."""
        mm: ModelManager = self.col.models
        mids: List[int] = list(map(lambda m: m.id, mm.all_names_and_ids()))
        mid: int = data.draw(st.sampled_from(mids), "rm nt: mid")
        mm.remove(mid)

    @precondition(lambda self: not self.saved)
    @rule()
    def save(self) -> None:
        """Checkpoint the database."""
        self.checkpoint: File = checkpoint_anki2(self.path, "checkpoint")
        self.saved = True

    def teardown(self) -> None:
        """Cleanup the state of the system."""
        did = self.col.decks.id("dummy", create=True)
        self.col.decks.remove([did])
        self.col.save()
        self.col.db.commit()
        final: File = checkpoint_anki2(self.path, "final")
        if not self.saved:
            self.checkpoint = INITIAL
        assert str(final) != str(EMPTY.col_file)
        assert str(final) != str(self.checkpoint)
        assert F.md5(final) != F.md5(EMPTY.col_file)
        # assert F.md5(final) != F.md5(self.checkpoint)
        p = subprocess.run(
            ["sqldiff", str(self.checkpoint), str(final)],
            capture_output=True,
            check=True,
        )
        block = p.stdout.decode()
        print("\n".join(filter(lambda l: "col" not in l, block.split("\n"))))
        tree = PARSER.parse(block)
        stmts = TRANSFORMER.transform(tree)
        stmts = list(
            filter(
                lambda s: not (isinstance(s, Update) and s.table == Table.Collection),
                stmts,
            )
        )
        logger.debug(pp.pformat(stmts))

        shutil.rmtree(self.tempd)
        if self.freeze:
            self.freezer.stop()
        if self.profile:
            self.profiler.stop()
            Path("profile.html").write_text(self.profiler.output_html())


AnkiCollection.TestCase.settings = settings(
    max_examples=50,
    stateful_step_count=50,
    verbosity=Verbosity.normal,
    deadline=None,
)
TestAnkiCollection = AnkiCollection.TestCase
