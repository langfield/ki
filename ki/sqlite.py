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
)

from loguru import logger

# pylint: disable=invalid-name, no-self-use, too-few-public-methods
# pylint: disable=missing-class-docstring


class Table(Enum):
    Notes = "notes"


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
    def diff(self, xs: List[Statement]) -> List[Statement]:
        return xs

    @beartype
    def stmt(self, xs: List[Statement]) -> Statement:
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
    def assignments(self, xs: List[Tuple[str, str]]) -> AssignmentMap:
        return dict(xs)

    @beartype
    def row(self, xs: List[int]) -> int:
        assert len(xs) == 1
        return xs[0]

    @beartype
    def values(self, xs: List[Value]) -> List[Value]:
        print("values")
        return xs

    @beartype
    def TABLE(self, t: Token) -> Table:
        print("TABLE")
        s = str(t)
        if s == "notes":
            return Table.Notes
        raise ValueError(f"Invalid table: {s}")

    @beartype
    def assignment(self, t: Token) -> Tuple[str, str]:
        logger.debug("ASSGN")
        s = str(t)
        sections = s.split("=")
        assert len(sections) == 2
        col, val = sections
        return col, val

    @beartype
    def value(self, xs: List[Value]) -> Value:
        return xs[0]

    @beartype
    def note_fields(self, xs: List[str]) -> List[str]:
        ys = map(lambda x: x if type(x) == str else str(x), xs)
        s = "".join(xs)
        return s.split("\x1f")

    @beartype
    def field_sequence(self, xs: List[str]) -> str:
        return xs[0]

    @beartype
    def string_sequence(self, ts: List[str]) -> str:
        assert len(ts) == 1
        return str(ts[0])

    @beartype
    def byte_sequence(self, xs: List[Token]) -> str:
        return bytes.fromhex("".join(list(map(str, xs)))).decode(encoding="UTF-8")

    @beartype
    def SINGLE_QUOTED_STRING(self, t: Token) -> str:
        logger.debug("SINGLE QUOTED")
        return str(t)

    @beartype
    def SINGLE_QUOTED_STRING_NOT_FIELD(self, t: Token) -> str:
        print("SINGLE QUOTED STRING NOT")
        return str(t)

    @beartype
    def INT(self, t: Token) -> int:
        return int(str(t))
