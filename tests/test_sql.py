"""Tests for SQLite Lark grammar."""
from __future__ import annotations

from pathlib import Path

import pytest

from loguru import logger
from beartype import beartype

from lark import Lark
from lark.exceptions import (
    UnexpectedToken,
    UnexpectedInput,
    UnexpectedCharacters,
    VisitError,
)

import ki
from ki.sqlite import SQLiteTransformer

# pylint: disable=too-many-lines, missing-function-docstring

BAD_ASCII_CONTROLS = ["\0", "\a", "\b", "\v", "\f"]


def get_parser():
    """Return a parser."""
    # Read grammar.
    grammar_path = Path(ki.__file__).resolve().parent / "sqlite.lark"
    grammar = grammar_path.read_text(encoding="UTF-8")

    # Instantiate parser.
    parser = Lark(grammar, start="diff", parser="lalr")

    return parser


@beartype
def debug_lark_error(note: str, err: UnexpectedInput) -> None:
    """Print an exception."""
    logger.warning(f"\n{note}")
    logger.error(f"accepts: {err.accepts}")
    logger.error(f"column: {err.column}")
    logger.error(f"expected: {err.expected}")
    logger.error(f"line: {err.line}")
    logger.error(f"pos_in_stream: {err.pos_in_stream}")
    logger.error(f"token: {err.token}")
    logger.error(f"token_history: {err.token_history}")
    logger.error(f"\n{err}")


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
    parser = get_parser()
    parser.parse(BLOCK)


def test_transformer():
    parser = get_parser()
    tree = parser.parse(BLOCK)
    transformer = SQLiteTransformer()
    block = transformer.transform(tree)
    import prettyprinter as pp
    logger.debug(pp.pformat(block))


def test_transformer_on_update_commands():
    parser = get_parser()
    tree = parser.parse(UPDATE)
    transformer = SQLiteTransformer()
    block = transformer.transform(tree)
    import prettyprinter as pp
    logger.debug(pp.pformat(block))
