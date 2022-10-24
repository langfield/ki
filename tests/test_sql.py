"""Tests for SQLite Lark grammar."""
from __future__ import annotations

import os
import time
import tempfile
from pathlib import Path

from loguru import logger

from beartype import beartype
from beartype.typing import Iterable, Sequence

from lark import Lark
from lark.exceptions import (
    UnexpectedToken,
    UnexpectedInput,
    UnexpectedCharacters,
    VisitError,
)

import hypothesis.strategies as st
from hypothesis import given, settings
from hypothesis.stateful import (
    RuleBasedStateMachine,
    Bundle,
    rule,
    initialize,
    multiple,
)

import anki.collection
from anki._backend import RustBackend
from anki.models import NotetypeNameId, NotetypeDict
from anki.collection import Collection, Note

import ki
import ki.functional as F
from ki.types import Dir
from ki.sqlite import SQLiteTransformer
from tests.test_parser import get_parser, debug_lark_error

from pyinstrument import Profiler

# pylint: disable=too-many-lines, missing-function-docstring

BAD_ASCII_CONTROLS = ["\0", "\a", "\b", "\v", "\f"]

tempd: Path = Path(tempfile.mkdtemp())
COL = Collection(tempd / "collection.anki2")


class AnkiCollection(RuleBasedStateMachine):
    def __init__(self):
        t1 = time.time()
        profiler = Profiler()
        profiler.start()

        super().__init__()
        self.col = COL

        profiler.stop()
        html = profiler.output_html()
        Path("hyp.html").write_text(html)
        logger.debug(time.time() - t1)

    notetypes = Bundle("notetypes")

    @initialize(target=notetypes)
    @beartype
    def init_collection(self) -> Iterable[NotetypeDict]:
        name_ids: Sequence[NotetypeNameId] = self.col.models.all_names_and_ids()
        models = list(map(lambda m: self.col.models.get(m.id), name_ids))
        return multiple(*models)

    @rule(notetype=notetypes)
    @beartype
    def add_note(self, notetype: NotetypeDict) -> None:
        note: Note = self.col.new_note(notetype)

AnkiCollection.TestCase.settings = settings(max_examples=100)
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
    block = transformer.transform(tree)


def test_transformer_on_update_commands():
    parser = get_parser(filename="sqlite.lark", start="diff")
    tree = parser.parse(UPDATE)
    transformer = SQLiteTransformer()
    block = transformer.transform(tree)
