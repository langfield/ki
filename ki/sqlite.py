#!/usr/bin/env python3
"""A Lark transformer for the ki note grammar."""
import json
from enum import Enum
from itertools import starmap
from dataclasses import dataclass

from dacite import from_dict

from lark import Transformer
from lark.lexer import Token

from beartype import beartype
from beartype.typing import (
    List,
    Dict,
    Union,
    Tuple,
    Any,
    Optional,
)

from ki.types import SQLNote, SQLCard, SQLDeck, SQLField, SQLNotetype, SQLTemplate

from loguru import logger

# pylint: disable=invalid-name, no-self-use, too-few-public-methods
# pylint: disable=missing-class-docstring, too-many-instance-attributes
# pylint: disable=too-many-public-methods


class Table(Enum):
    Notes = "notes"
    Cards = "cards"
    Decks = "decks"
    Fields = "fields"
    Notetypes = "notetypes"
    Templates = "templates"
    Collection = "col"


@beartype
@dataclass(frozen=True, eq=True)
class Field:
    font: str
    name: str
    ord: int
    rtl: bool
    size: int
    sticky: bool
    media: Optional[List[str]]


@beartype
@dataclass(frozen=True, eq=True)
class Template:
    afmt: str
    bafmt: str
    bqfmt: str
    did: Union[str, None]
    name: str
    ord: int
    qfmt: str


@beartype
@dataclass(frozen=True, eq=True)
class Model:
    css: str
    did: Union[int, None]
    flds: List[Field]
    id: int
    latexPost: str
    latexPre: str
    mod: int
    name: str
    req: Any
    sortf: int
    tmpls: List[Template]
    type: int
    usn: int
    vers: Optional[int]
    tags: Optional[List[str]]


@beartype
@dataclass(frozen=True, eq=True)
class Deck:
    name: str
    id: int
    mod: int
    usn: int
    dyn: int
    desc: str


Row, Column = int, str
NtidRow = int
OrdRow = int
FieldText = str
Value = Union[int, str, List[FieldText], Dict[int, Model], Dict[int, Deck], None]
AssignmentMap = Dict[Column, Value]

DeckName = Tuple[str, ...]
Fields = Tuple[str, ...]

NoteValue = Union[Table, Fields, int, str]
CardValue = Union[Table, int, str]
DeckValue = Union[Table, DeckName, int, str]
FieldValue = Union[Table, int, str]
NotetypeValue = Union[Table, int, str]
TemplateValue = Union[Table, int, str]

InsertionValue = Union[SQLNote, SQLCard, SQLDeck, SQLField, SQLNotetype, SQLTemplate]


@beartype
@dataclass(frozen=True, eq=True)
class Insert:
    table: Table
    data: InsertionValue


@beartype
@dataclass(frozen=True, eq=True)
class Update:
    table: Table
    assignments: AssignmentMap
    row: int


@beartype
@dataclass(frozen=True, eq=True)
class Delete:
    table: Table
    row: int

@beartype
@dataclass(frozen=True, eq=True)
class NtDelete:
    table: Table
    ntid: int
    ord: int


Statement = Union[Insert, Update, Delete, NtDelete]


class SQLiteTransformer(Transformer):
    """Parse SQL block."""

    # pylint: disable=missing-function-docstring

    @beartype
    def diff(self, xs: List[Union[Statement, None]]) -> List[Statement]:
        return list(filter(lambda x: x is not None, xs))

    @beartype
    def stmt(self, xs: List[Union[Statement, None]]) -> Union[Statement, None]:
        assert len(xs) == 1
        return xs[0]

    @beartype
    def insert(self, xs: List[InsertionValue]) -> Insert:
        assert len(xs) == 1
        x = xs[0]
        if isinstance(x, SQLNote):
            return Insert(table=Table.Notes, data=x)
        if isinstance(x, SQLCard):
            return Insert(table=Table.Cards, data=x)
        if isinstance(x, SQLDeck):
            return Insert(table=Table.Decks, data=x)
        if isinstance(x, SQLField):
            return Insert(table=Table.Fields, data=x)
        if isinstance(x, SQLNotetype):
            return Insert(table=Table.Notetypes, data=x)
        return Insert(table=Table.Templates, data=x)


    @beartype
    def update(self, xs: List[Union[Table, AssignmentMap, Row]]) -> Update:
        assert len(xs) == 3
        return Update(table=xs[0], assignments=xs[1], row=xs[2])

    @beartype
    def delete(self, xs: List[Union[Table, Row]]) -> Delete:
        assert len(xs) == 2
        return Delete(table=xs[0], row=xs[1])

    @beartype
    def nt_delete(self, xs: List[Union[Table, NtidRow, OrdRow]]) -> NtDelete:
        assert len(xs) == 3
        return NtDelete(table=xs[0], ntid=xs[1], ord=xs[2])

    @beartype
    def bad(self, _: Any) -> None:
        return None

    @beartype
    def insertion(self, xs: List[InsertionValue]) -> InsertionValue:
        assert len(xs) == 1
        return xs[0]

    @beartype
    def note(self, xs: List[NoteValue]) -> SQLNote:
        _, _, guid, mid, _, _, tags, flds, _, _, _, _ = xs
        return SQLNote(mid=mid, guid=guid, tags=tags, flds=flds)

    @beartype
    def card(self, xs: List[CardValue]) -> SQLCard:
        _, cid, nid, did, ord = xs[:5]
        return SQLCard(cid=cid, nid=nid, did=did, ord=ord)

    @beartype
    def deck(self, xs: List[DeckValue]) -> SQLDeck:
        assert len(xs) == 7
        _, did, deckname = xs[:3]
        return SQLDeck(did=did, deckname=deckname)

    @beartype
    def field(self, xs: List[FieldValue]) -> SQLField:
        assert len(xs) == 5
        _, ntid, ord, fieldname = xs[:4]
        return SQLField(ntid=ntid, ord=ord, fieldname=fieldname)

    @beartype
    def notetype(self, xs: List[NotetypeValue]) -> SQLNotetype:
        assert len(xs) == 6
        _, ntid, ntname = xs[:3]
        return SQLNotetype(ntid=ntid, ntname=ntname)

    @beartype
    def template(self, xs: List[TemplateValue]) -> SQLTemplate:
        assert len(xs) == 7
        _, ntid, ord, tmplname = xs[:4]
        return SQLTemplate(ntid=ntid, ord=ord, tmplname=tmplname)

    @beartype
    def NOTES_SCHEMA(self, _: Token) -> Table:
        return Table.Notes

    @beartype
    def CARDS_SCHEMA(self, _: Token) -> Table:
        return Table.Cards

    @beartype
    def DECKS_SCHEMA(self, _: Token) -> Table:
        return Table.Decks

    @beartype
    def FIELDS_SCHEMA(self, _: Token) -> Table:
        return Table.Fields

    @beartype
    def NOTETYPES_SCHEMA(self, _: Token) -> Table:
        return Table.Notetypes

    @beartype
    def TEMPLATES_SCHEMA(self, _: Token) -> Table:
        return Table.Templates

    @beartype
    def assignments(self, xs: List[Tuple[str, Value]]) -> AssignmentMap:
        return dict(filter(lambda x: x[1] is not None, xs))

    @beartype
    def row(self, xs: List[int]) -> int:
        assert len(xs) == 1
        return xs[0]

    @beartype
    def ord_row(self, xs: List[int]) -> int:
        assert len(xs) == 1
        return xs[0]

    @beartype
    def ntid_row(self, xs: List[int]) -> int:
        assert len(xs) == 1
        return xs[0]

    @beartype
    def TABLE(self, t: Token) -> Table:
        s = str(t)
        if s == "notes":
            return Table.Notes
        if s == "cards":
            return Table.Cards
        if s == "col":
            return Table.Collection
        if s == "decks":
            return Table.Decks
        if s == "fields":
            return Table.Fields
        if s == "notetypes":
            return Table.Notetypes
        if s == "templates":
            return Table.Templates
        raise ValueError(f"Invalid table: {s}")

    @beartype
    def assignment(self, ts: List[Union[Token, Value]]) -> Tuple[str, Value]:
        assert len(ts) == 2
        column, val = ts
        if column in ("models", "decks"):
            val = val.lstrip("'")
            val = val.rstrip("'")
            val = json.loads(val)
            if column == "models":
                val = dict(
                    starmap(lambda k, v: (int(k), from_dict(Model, v)), val.items())
                )
            elif column == "decks":
                val = dict(
                    starmap(lambda k, v: (int(k), from_dict(Deck, v)), val.items())
                )
        if column in ("conf",):
            return str(column), None
        return str(column), val

    @beartype
    def value(self, xs: List[Value]) -> Value:
        return xs[0]

    @beartype
    def flds(self, xs: List[str]) -> Tuple[str, ...]:
        # pylint: disable=unidiomatic-typecheck
        ys = map(lambda x: x if type(x) == str else str(x), xs)
        s = "".join(ys)
        return tuple(s.split("\x1f"))

    @beartype
    def deckname(self, xs: List[str]) -> Tuple[str, ...]:
        # pylint: disable=unidiomatic-typecheck
        ys = map(lambda x: x if type(x) == str else str(x), xs)
        s = "".join(ys)
        return tuple(s.split("\x1f"))

    @beartype
    def bytestring(self, xs: List[str]) -> str:
        # pylint: disable=unidiomatic-typecheck
        ys = map(lambda x: x if type(x) == str else str(x), xs)
        return "".join(ys)

    @beartype
    def sfld(self, xs: Union[List[int], List[str]]) -> Union[int, str]:
        assert len(xs) == 1
        x = xs[0]
        return x

    @beartype
    def seq(self, xs: List[str]) -> str:
        return xs[0]

    @beartype
    def bytes(self, xs: List[Token]) -> str:
        return bytes.fromhex("".join(list(map(str, xs)))).decode(encoding="UTF-8")

    @beartype
    def cid(self, xs: List[int]) -> int:
        assert len(xs) == 1
        return xs[0]

    @beartype
    def nid(self, xs: List[int]) -> int:
        assert len(xs) == 1
        return xs[0]

    @beartype
    def did(self, xs: List[int]) -> int:
        assert len(xs) == 1
        return xs[0]

    @beartype
    def ord(self, xs: List[int]) -> int:
        assert len(xs) == 1
        return xs[0]

    @beartype
    def mod(self, xs: List[int]) -> int:
        assert len(xs) == 1
        return xs[0]

    @beartype
    def usn(self, xs: List[int]) -> int:
        assert len(xs) == 1
        return xs[0]

    @beartype
    def type(self, xs: List[int]) -> int:
        assert len(xs) == 1
        return xs[0]

    @beartype
    def queue(self, xs: List[int]) -> int:
        assert len(xs) == 1
        return xs[0]

    @beartype
    def due(self, xs: List[int]) -> int:
        assert len(xs) == 1
        return xs[0]

    @beartype
    def ivl(self, xs: List[int]) -> int:
        assert len(xs) == 1
        return xs[0]

    @beartype
    def factor(self, xs: List[int]) -> int:
        assert len(xs) == 1
        return xs[0]

    @beartype
    def reps(self, xs: List[int]) -> int:
        assert len(xs) == 1
        return xs[0]

    @beartype
    def lapses(self, xs: List[int]) -> int:
        assert len(xs) == 1
        return xs[0]

    @beartype
    def left(self, xs: List[int]) -> int:
        assert len(xs) == 1
        return xs[0]

    @beartype
    def odue(self, xs: List[int]) -> int:
        assert len(xs) == 1
        return xs[0]

    @beartype
    def odid(self, xs: List[int]) -> int:
        assert len(xs) == 1
        return xs[0]

    @beartype
    def flags(self, xs: List[int]) -> int:
        assert len(xs) == 1
        return xs[0]

    @beartype
    def data(self, xs: List[str]) -> str:
        assert len(xs) == 1
        return xs[0]

    @beartype
    def guid(self, xs: List[str]) -> str:
        assert len(xs) == 1
        s = xs[0]
        return s

    @beartype
    def mid(self, xs: List[int]) -> int:
        assert len(xs) == 1
        return xs[0]

    @beartype
    def tags(self, xs: List[str]) -> Tuple[str, ...]:
        assert len(xs) == 1
        s = xs[0]
        s = s.lstrip()
        s = s.rstrip()
        return tuple(s.split())

    @beartype
    def csum(self, xs: List[int]) -> int:
        assert len(xs) == 1
        return xs[0]

    @beartype
    def mtime_secs(self, xs: List[int]) -> int:
        assert len(xs) == 1
        return xs[0]

    @beartype
    def common(self, xs: List[str]) -> str:
        assert len(xs) == 1
        return xs[0]

    @beartype
    def kind(self, xs: List[str]) -> str:
        assert len(xs) == 1
        return xs[0]

    @beartype
    def ntid(self, xs: List[int]) -> int:
        assert len(xs) == 1
        return xs[0]

    @beartype
    def fieldname(self, xs: List[str]) -> str:
        assert len(xs) == 1
        return xs[0]

    @beartype
    def config(self, xs: List[str]) -> str:
        assert len(xs) == 1
        return xs[0]

    @beartype
    def ntname(self, xs: List[str]) -> str:
        assert len(xs) == 1
        return xs[0]

    @beartype
    def tmplname(self, xs: List[str]) -> str:
        assert len(xs) == 1
        return xs[0]

    @beartype
    def BLOB(self, t: Token) -> str:
        s = str(t)
        s = s.removeprefix("x'")
        s = s.removesuffix("'")
        return s

    @beartype
    def STRING(self, t: Token) -> str:
        s = str(t)
        s = s.removeprefix("'")
        s = s.removesuffix("'")
        s = s.replace("''", "'")
        return s

    @beartype
    def INT(self, t: Token) -> int:
        return int(str(t))

    @beartype
    def SIGNED_INT(self, t: Token) -> int:
        return int(str(t))
