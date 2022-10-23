#!/usr/bin/env python3
"""A Lark transformer for the ki note grammar."""
from __future__ import annotations

import re
from dataclasses import dataclass

from lark import Transformer
from lark.lexer import Token

from beartype import beartype
from beartype.typing import (
    List,
    Dict,
    Optional,
    Union,
    Tuple,
)

# pylint: disable=invalid-name, no-self-use, too-few-public-methods


Row = int
Field = str
Value = Union[int, str]
Table = str
IndexedTable = str
AssignmentMap = Dict[Field, Value]

Insert = Tuple[IndexedTable, List[Value]]
Update = Tuple[Table, AssignmentMap, Row]
Delete = Tuple[Table, Row]
Bad = str
Statement = Union[Insert, Update, Delete, Bad]


class SQLiteTransformer(Transformer):
    """Parse SQL block."""
    # pylint: disable=missing-function-docstring

    @beartype
    def diff(self, xs: List[Statement]) -> List[Statement]:
        return xs

    @beartype
    def stmt(self, x: Statement) -> Statement:
        return x

    @beartype
    def insert(self, xs: List[Union[IndexedTable, List[Value]]]) -> Insert:
        assert len(xs) == 2
        return (xs[0], xs[1])

    @beartype
    def update(self, xs: List[Union[Table, AssignmentMap, Row]]) -> Update:
        assert len(xs) == 3
        return tuple(xs[:3])
    
    @beartype
    def delete(self, xs: List[Union[Table, Row]]) -> Delete:
        assert len(xs) == 2
        return tuple(xs[:2])

    @beartype
    def assignments(self, xs: List[Tuple[str, str]]) -> List[Value]:
        return xs

    @beartype
    def ASSIGNMENT(self, xs: List[str]) -> Tuple[str, str]:
        assert len(xs) == 2
        return xs[0], xs[1]

    @beartype
    def FIELD(self, t: Token) -> str:
        return str(t)

    @beartype
    def FIELD(self, t: Token) -> str:
        return str(t)

    @beartype
    def VALUE(self, x: Value) -> Value:
        return x

    @beartype
    def NOTE_FIELD(self, t: Token) -> str:
        return t

    @beartype
    def ESCAPED_STRING(self, t: Token) -> str:
        return str(t)

    @beartype
    def STRING_VALUE(self, t: Token) -> str:
        return str(t)

    @beartype
    def INT(self, t: Token) -> int:
        return int(str(t))
