#!/usr/bin/env python3
"""A Lark transformer for the ki note grammar."""
from __future__ import annotations

from enum import Enum
from dataclasses import dataclass

from lark import Transformer
from lark.lexer import Token

from beartype import beartype
from beartype.typing import (
    List,
    Dict,
    Union,
    Tuple,
    Any,
)

from loguru import logger

# pylint: disable=invalid-name, no-self-use, too-few-public-methods
# pylint: disable=missing-class-docstring


class Table(Enum):
    Notes = "notes"
    Collection = "col"


Row, Column = int, str
Field = str
Value = Union[int, str, List[Field]]
AssignmentMap = Dict[Column, Value]


@beartype
@dataclass(frozen=True, eq=True)
class Insert:
    table: Table
    values: List[Value]


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
    def insert(self, xs: List[Union[Table, Token, List[Value]]]) -> Insert:
        assert len(xs) == 3
        table, _, values = xs
        return Insert(table=table, values=values)

    @beartype
    def update(self, xs: List[Union[Table, AssignmentMap, Row]]) -> Update:
        assert len(xs) == 3
        return Update(table=xs[0], assignments=xs[1], row=xs[2])

    @beartype
    def delete(self, xs: List[Union[Table, Row]]) -> Delete:
        assert len(xs) == 2
        return Delete(table=xs[0], row=xs[1])

    @beartype
    def bad(self, _: Any) -> None:
        return None

    @beartype
    def assignments(self, xs: List[Tuple[str, Value]]) -> AssignmentMap:
        return dict(xs)

    @beartype
    def row(self, xs: List[int]) -> int:
        assert len(xs) == 1
        return xs[0]

    @beartype
    def values(self, xs: List[Value]) -> List[Value]:
        return xs

    @beartype
    def TABLE(self, t: Token) -> Table:
        s = str(t)
        if s == "notes":
            return Table.Notes
        if s == "col":
            return Table.Collection
        raise ValueError(f"Invalid table: {s}")

    @beartype
    def assignment(self, ts: List[Union[Token, Value]]) -> Tuple[str, Value]:
        assert len(ts) == 2
        column, val = ts
        return str(column), val

    @beartype
    def value(self, xs: List[Value]) -> Value:
        return xs[0]

    @beartype
    def fields(self, xs: List[str]) -> List[str]:
        ys = map(lambda x: x if type(x) == str else str(x), xs)
        s = "".join(xs)
        return s.split("\x1f")

    @beartype
    def bytestring(self, xs: List[str]) -> str:
        ys = map(lambda x: x if type(x) == str else str(x), xs)
        return "".join(xs)

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
    def STRING(self, t: Token) -> str:
        return str(t)

    @beartype
    def INT(self, t: Token) -> int:
        return int(str(t))

    @beartype
    def SIGNED_INT(self, t: Token) -> int:
        return int(str(t))

    @beartype
    def NUMBER(self, t: Token) -> Union[int, str]:
        x = float(str(t))
        return int(x) if x.is_integer() else str(x)
