#!/usr/bin/env python3
"""A Lark transformer for the ki note grammar."""
from enum import Enum
from dataclasses import dataclass

import prettyprinter as pp
from loguru import logger
from google.protobuf.json_format import MessageToDict

from lark import Transformer
from lark.lexer import Token

from beartype import beartype
from beartype.door import is_bearable, die_if_unbearable
from beartype.typing import (
    List,
    Dict,
    Union,
    Tuple,
    Any,
    Literal,
)
from anki.notetypes_pb2 import Notetype as NotetypePb2

from ki.types import SQLNote, SQLCard, SQLDeck, SQLField, SQLNotetype, SQLTemplate, Notetype


# pylint: disable=invalid-name, no-self-use, too-few-public-methods
# pylint: disable=missing-class-docstring, too-many-instance-attributes
# pylint: disable=too-many-public-methods, redefined-builtin


class Table(Enum):
    Notes = "notes"
    Cards = "cards"
    Decks = "decks"
    Fields = "fields"
    Notetypes = "notetypes"
    Templates = "templates"
    Collection = "col"

Model, Deck, Template = int, int, int

Row = int
Ntid = int
Ord = int
NtidRow = int
OrdRow = int
FieldText = str
Value = Union[int, str, List[FieldText], Dict[int, Model], Dict[int, Deck], None]

# Can we make this just a tuple somehow, by doing more complex parsing?
DeckName = Union[str, Tuple[str, ...]]
NoteFields = Tuple[str, ...]
Tags = Tuple[str, ...]

# Sum types for insert column values.
NoteValue = Union[Table, NoteFields, int, str]
CardValue = Union[Table, int, str]
DeckValue = Union[Table, DeckName, int, str]
FieldValue = Union[Table, int, str]
NotetypeValue = Union[Table, int, str]
TemplateValue = Union[Table, int, str]

class Column(Enum):
    id = "id"
    guid = "guid"
    mid = "mid"
    mod = "mod"
    usn = "usn"
    tags = "tags"
    flds = "flds"
    sfld = "sfld"
    csum = "csum"
    flags = "flags"
    data = "data"
    nid = "nid"
    did = "did"
    ord = "ord"
    type = "type"
    queue = "queue"
    due = "due"
    ivl = "ivl"
    factor = "factor"
    reps = "reps"
    lapses = "lapses"
    left = "left"
    odue = "odue"
    odid = "odid"
    crt = "crt"
    scm = "scm"
    ver = "ver"
    ls = "ls"
    conf = "conf"
    models = "models"
    decks = "decks"
    dconf = "dconf"
    name = "name"
    mtime_secs = "mtime_secs"
    common = "common"
    kind = "kind"
    config = "config"

COLUMN_NAME_MAP = dict(Column.__members__)

# This should be a dataclass.
NotetypeConfig = Dict[str, str]
NotetypeFieldConfig = Dict[str, str]
NotetypeTemplateConfig = Dict[str, str]

# Key-value pairs for updates.
NoteKeyValue = Union[
    Tuple[Literal[Column.mid], int],
    Tuple[Literal[Column.guid], str],
    Tuple[Literal[Column.tags], Tags],
    Tuple[Literal[Column.flds], NoteFields],
]
CardKeyValue = Union[
    Tuple[Literal[Column.did], int],
]
DeckKeyValue = Union[
    Tuple[Literal[Column.name], DeckName],
]
FieldKeyValue = Union[
    Tuple[Literal[Column.name], str],
    Tuple[Literal[Column.config], NotetypeFieldConfig],
]
NotetypeKeyValue = Union[
    Tuple[Literal[Column.name], str],
    Tuple[Literal[Column.config], NotetypeConfig],
]
TemplateKeyValue = Union[
    Tuple[Literal[Column.name], str],
    Tuple[Literal[Column.config], NotetypeTemplateConfig],
]
UnparsedProtoBufConfigKeyValue = Tuple[Literal[Column.config], bytes]

USED = {Column.mid, Column.guid, Column.tags, Column.flds, Column.did, Column.name, Column.config}


KeyValue = Union[
    NoteKeyValue,
    CardKeyValue,
    DeckKeyValue,
    FieldKeyValue,
    NotetypeKeyValue,
    TemplateKeyValue,
    UnparsedProtoBufConfigKeyValue,
]
KeyValues = Tuple[KeyValue, ...]

# Foldables of key-value pairs, i.e. our type-safe, immutable, hashable
# versions of dicts, since @beartype doesn't support them yet :(
NoteKeyValues = Tuple[NoteKeyValue, ...]
CardKeyValues = Tuple[CardKeyValue, ...]
DeckKeyValues = Tuple[DeckKeyValue, ...]
FieldKeyValues = Tuple[FieldKeyValue, ...]
NotetypeKeyValues = Tuple[NotetypeKeyValue, ...]
TemplateKeyValues = Tuple[TemplateKeyValue, ...]


# Update types for each table.

@beartype
@dataclass(frozen=True, eq=True)
class NotesUpdate:
    updates: NoteKeyValues
    row: Row

@beartype
@dataclass(frozen=True, eq=True)
class CardsUpdate:
    updates: CardKeyValues
    row: Row

@beartype
@dataclass(frozen=True, eq=True)
class DecksUpdate:
    updates: DeckKeyValues
    row: Row

@beartype
@dataclass(frozen=True, eq=True)
class FieldsUpdate:
    updates: FieldKeyValues
    ntid: Ntid
    ord: Ord

@beartype
@dataclass(frozen=True, eq=True)
class NotetypesUpdate:
    updates: NotetypeKeyValues
    row: Row

@beartype
@dataclass(frozen=True, eq=True)
class TemplatesUpdate:
    updates: TemplateKeyValues
    ntid: Ntid
    ord: Ord

# Our new `Update` type.
Update = Union[
    NotesUpdate,
    CardsUpdate,
    DecksUpdate,
    FieldsUpdate,
    NotetypesUpdate,
    TemplatesUpdate,
]

# Delete types for each table.

@beartype
@dataclass(frozen=True, eq=True)
class NotesDelete:
    row: Row

@beartype
@dataclass(frozen=True, eq=True)
class CardsDelete:
    row: Row

@beartype
@dataclass(frozen=True, eq=True)
class DecksDelete:
    row: Row

@beartype
@dataclass(frozen=True, eq=True)
class FieldsDelete:
    ntid: Ntid
    ord: Ord

@beartype
@dataclass(frozen=True, eq=True)
class NotetypesDelete:
    row: Row

@beartype
@dataclass(frozen=True, eq=True)
class TemplatesDelete:
    ntid: Ntid
    ord: Ord

# Our new `Delete` type.
Delete = Union[
    NotesDelete,
    CardsDelete,
    DecksDelete,
    FieldsDelete,
    NotetypesDelete,
    TemplatesDelete,
]

# Our new `Insert` type.
Insert = Union[SQLNote, SQLCard, SQLDeck, SQLField, SQLNotetype, SQLTemplate]

Statement = Union[Insert, Update, Delete]


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
    def insert(self, xs: List[Insert]) -> Insert:
        assert len(xs) == 1
        return xs[0]


    @beartype
    def update(self, xs: List[Union[Table, KeyValues, Row, Tuple[Ntid, Ord]]]) -> Union[Update, None]:
        assert len(xs) == 3
        table, updates, row = xs
        assert isinstance(table, Table)
        assert is_bearable(updates, KeyValues)
        if len(updates) == 0:
            return None
        if table == Table.Notes:
            return NotesUpdate(updates=updates, row=row)
        if table == Table.Cards:
            return CardsUpdate(updates=updates, row=row)
        if table == Table.Decks:
            return DecksUpdate(updates=updates, row=row)
        if table == Table.Fields:
            assert len(row) == 2
            ntid, ord = row
            updates = dict(updates)
            if Column.config in updates:
                msg = NotetypePb2.Field.Config()
                msg.ParseFromString(updates[Column.config])
                updates |= {Column.config: MessageToDict(msg)}
            updates = tuple(updates.items())
            return FieldsUpdate(updates=updates, ntid=ntid, ord=ord)
        if table == Table.Notetypes:
            updates = dict(updates)
            if Column.config in updates:
                msg = NotetypePb2.Config()
                msg.ParseFromString(updates[Column.config])
                updates |= {Column.config: MessageToDict(msg)}
            updates = tuple(updates.items())
            return NotetypesUpdate(updates=updates, row=row)
        if table == Table.Templates:
            assert len(row) == 2
            ntid, ord = row
            updates = dict(updates)
            if Column.config in updates:
                msg = NotetypePb2.Template.Config()
                msg.ParseFromString(updates[Column.config])
                updates |= {Column.config: MessageToDict(msg)}
            updates = tuple(updates.items())
            return TemplatesUpdate(updates=updates, ntid=ntid, ord=ord)
        raise ValueError(f"Bad table: {table}")

    @beartype
    def delete(self, xs: List[Union[Table, Row, Tuple[Ntid, Ord]]]) -> Delete:
        assert len(xs) == 2
        table, row = xs
        assert isinstance(table, Table)
        if table == Table.Notes:
            return NotesDelete(row=row)
        if table == Table.Cards:
            return CardsDelete(row=row)
        if table == Table.Decks:
            return DecksDelete(row=row)
        if table == Table.Fields:
            assert len(row) == 2
            ntid, ord = row
            return FieldsDelete(ntid=ntid, ord=ord)
        if table == Table.Notetypes:
            return NotetypesDelete(row=row)
        if table == Table.Templates:
            assert len(row) == 2
            ntid, ord = row
            return TemplatesDelete(ntid=ntid, ord=ord)
        raise ValueError(f"Bad table: {table}")

    @beartype
    def bad(self, _: Any) -> None:
        return None

    @beartype
    def insertion(self, xs: List[Insert]) -> Insert:
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
    def assignments(self, xs: List[Union[KeyValue, None]]) -> KeyValues:
        return tuple(filter(lambda x: x is not None, xs))

    @beartype
    def assignment(self, ts: List[Union[Token, Value]]) -> Union[KeyValue, None]:
        assert len(ts) == 2
        name, val = ts
        if name not in COLUMN_NAME_MAP:
            raise ValueError(f"Invalid column name: {name}")
        column = COLUMN_NAME_MAP[name]
        if column not in USED:
            return None
        if column == Column.flds:
            die_if_unbearable(val, str)
            val = tuple(val.split("\x1f"))
        if column == Column.config:
            val = bytes.fromhex(val)
        return (column, val)

    @beartype
    def row(self, xs: List[int]) -> Union[Row, Tuple[Ntid, Ord]]:
        assert len(xs) in (1, 2)
        if len(xs) == 1:
            return xs[0]
        if len(xs) == 2:
            return tuple(xs)
        raise ValueError(f"Row specified must be length 1 or 2: {xs}")

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
