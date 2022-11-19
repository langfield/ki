"""Tests for SQLite Lark grammar."""
from pathlib import Path

import pytest
import prettyprinter as pp
from beartype import beartype
from beartype.typing import List

import ki
from ki.types import SQLNote, SQLCard, SQLDeck
from ki.sqlite import SQLiteTransformer, Table, Insert, Statement, Update, Delete, NtDelete
from tests.test_parser import get_parser, debug_lark_error


# pylint: disable=too-many-lines, missing-function-docstring


@beartype
def transform(sql: str) -> List[Statement]:
    parser = get_parser(filename="sqlite.lark", start="diff")
    try:
        tree = parser.parse(sql)
    except Exception as err:
        debug_lark_error(err)
        raise err
    transformer = SQLiteTransformer()
    return transformer.transform(tree)


BLOCK = r"""
DELETE FROM notes WHERE id=1645010162168;
DELETE FROM notes WHERE id=1645027705329;
INSERT INTO notes(id,guid,mid,mod,usn,tags,flds,sfld,csum,flags,data) VALUES(1651202347045,'D(l.-iAXR1',1651202298367,1651202347,-1,'','a'||X'1f'
        ||'b[sound:1sec.mp3]','a',2264392759,0,'');
INSERT INTO notes(id,guid,mid,mod,usn,tags,flds,sfld,csum,flags,data) VALUES(1651232832286,'H0%>O*C~M!',1651232485378,1651232832,-1,'','Who introduced the notion of a direct integral in functional analysis?'||X'1f'
        ||'John von Neumann','Who introduced the notion of a direct integral in functional analysis?',2707929287,0,'');
"""


def test_transformer():
    assert transform(BLOCK) == [
        Delete(table=Table.Notes, row=1645010162168),
        Delete(table=Table.Notes, row=1645027705329),
        Insert(table=Table.Notes, data=SQLNote(mid=1651202298367, guid="D(l.-iAXR1", tags=(), flds=("a", "b[sound:1sec.mp3]"))),
        Insert(table=Table.Notes, data=SQLNote(mid=1651232485378, guid="H0%>O*C~M!", tags=(), flds=("Who introduced the notion of a direct integral in functional analysis?", "John von Neumann"))),
    ]


UPDATE = r"""
UPDATE notes SET mod=1645221606, flds='aa'||X'1f'
||'bb', sfld='aa', csum=3771269976 WHERE id=1645010162168;
DELETE FROM notes WHERE id=1645027705329;
INSERT INTO notes(id,guid,mid,mod,usn,tags,flds,sfld,csum,flags,data) VALUES(1645222430007,'f}:^>jzMjG',1645010146011,1645222430,-1,'','f'||X'1f'
||'g','f',1242175777,0,'');
"""


def test_transformer_on_update_commands():
    assert transform(UPDATE) == [
        Update(table=Table.Notes, assignments={"mod": 1645221606, "flds": "aa\x1fbb", "sfld": "aa", "csum": 3771269976}, row=1645010162168),
        Delete(table=Table.Notes, row=1645027705329),
        Insert(table=Table.Notes, data=SQLNote(mid=1645010146011, guid="f}:^>jzMjG", tags=(), flds=("f", "g"))),
    ]


EMPTYISH = r"""
INSERT INTO notes(id,guid,mid,mod,usn,tags,flds,sfld,csum,flags,data) VALUES(1651363200000,'ku}V%9e9,l',1667061149792,1651363200,-1,'',''||X'1f','',3661210606,0,'');
"""


def test_transformer_on_empty_insert():
    assert transform(EMPTYISH) == [Insert(table=Table.Notes, data=SQLNote(mid=1667061149792, guid="ku}V%9e9,l", tags=(), flds=("", "")))]


QUOTES = r"""
INSERT INTO notes(id,guid,mid,mod,usn,tags,flds,sfld,csum,flags,data) VALUES(1651363200000,'|o/qdllw(',1667061149792,1651363200,-1,'',''||X'1f'
||'''','',3661210606,0,'');
"""


def test_transformer_on_single_quotes():
    assert transform(QUOTES) == [Insert(table=Table.Notes, data=SQLNote(mid=1667061149792, guid="|o/qdllw(", tags=(), flds=("", "'")))]


THREE_FIELDS = r"""
INSERT INTO notes(id,guid,mid,mod,usn,tags,flds,sfld,csum,flags,data) VALUES(1651363200001,'roH<$&er7G',1667061149794,1651363200,-1,'',''||X'1f1f','',3661210606,0,'');
"""


def test_transformer_on_three_fields():
    assert transform(THREE_FIELDS) == [Insert(table=Table.Notes, data=SQLNote(mid=1667061149794, guid="roH<$&er7G", tags=(), flds=("", "", "")))]


TRAILING_EMPTY_FIELD = r"""
INSERT INTO notes(id,guid,mid,mod,usn,tags,flds,sfld,csum,flags,data) VALUES(1651363200001,'r#){>b5q^b',1667061149794,1651363200,-1,'',''||X'1f'
||'0'||X'1f','',3661210606,0,'');
"""


def test_transformer_on_trailing_empty_field():
    assert transform(TRAILING_EMPTY_FIELD) == [Insert(table=Table.Notes, data=SQLNote(mid=1667061149794, guid="r#){>b5q^b", tags=(), flds=("", "0", "")))]


ESCAPED_SINGLE_QUOTE_AS_SORT_FIELD = r"""
INSERT INTO notes(id,guid,mid,mod,usn,tags,flds,sfld,csum,flags,data) VALUES(1651363200000,'OJl<Xj<>{H',1667061149792,1651363200,-1,'',''''||X'1f','',3143146758,0,'');
"""


def test_transformer_on_single_quote_in_sort_field():
    assert transform(ESCAPED_SINGLE_QUOTE_AS_SORT_FIELD) == [Insert(table=Table.Notes, data=SQLNote(mid=1667061149792, guid="OJl<Xj<>{H", tags=(), flds=("'", "")))]


INTEGER_SORT_FIELD = r"""
INSERT INTO notes(id,guid,mid,mod,usn,tags,flds,sfld,csum,flags,data) VALUES(1651363200000,'F,y8PO5.KP',1667061149792,1651363200,-1,'','0'||X'1f',0,3059261382,0,'');
"""


def test_transformer_on_integer_sort_field():
    assert transform(INTEGER_SORT_FIELD) == [Insert(table=Table.Notes, data=SQLNote(mid=1667061149792, guid="F,y8PO5.KP", tags=(), flds=("0", "")))]


RAW_BYTES_IN_SORT_FIELD = r"""
INSERT INTO notes(id,guid,mid,mod,usn,tags,flds,sfld,csum,flags,data) VALUES(1651363200001,'O#4+2LG`3{',1667061149792,1651363200,-1,'',''||X'0a1f',''||X'0a',2915580697,0,'');
"""


def test_transformer_on_raw_bytes_in_sort_field():
    assert transform(RAW_BYTES_IN_SORT_FIELD) == [Insert(table=Table.Notes, data=SQLNote(mid=1667061149792, guid="O#4+2LG`3{", tags=(), flds=("\x0a", "")))]


SURROGATE_UTF_BYTES = r"""
INSERT INTO notes(id,guid,mid,mod,usn,tags,flds,sfld,csum,flags,data) VALUES(1651363217005,'c5zvCfp,h%',1667061149796,1651363217,-1,'',',-򁿈AÜ'||X'1f',',-򁿈AÜ',3950683052,0,'');
"""


def test_transformer_on_surrogate_utf_bytes():
    assert transform(SURROGATE_UTF_BYTES) == [Insert(table=Table.Notes, data=SQLNote(mid=1667061149796, guid="c5zvCfp,h%", tags=(), flds=(",-򁿈AÜ", "")))]


DECIMALS_IN_SORT_FIELD = r"""
INSERT INTO notes(id,guid,mid,mod,usn,tags,flds,sfld,csum,flags,data) VALUES(1651363200000,'q6qBOHAT=6',1667061149792,1651363200,-1,'','.1'||X'1f',0.1,162134904,0,'');
"""


def test_transformer_on_decimals_in_sort_field():
    assert transform(DECIMALS_IN_SORT_FIELD) == [Insert(table=Table.Notes, data=SQLNote(mid=1667061149792, guid="q6qBOHAT=6", tags=(), flds=(".1", "")))]


with open(Path(ki.__file__).parent.parent / "tests" / "update.sql", "r", encoding="UTF-8") as f:
    COL_UPDATE = f.read()


def test_transformer_on_col_update_queries():
    transform(COL_UPDATE)


CARDS = r"""
INSERT INTO cards(id,nid,did,ord,mod,usn,type,queue,due,ivl,factor,reps,lapses,"left",odue,odid,flags,data) VALUES(1651363200000,1651363200000,1,0,1651363200,-1,0,0,1,0,0,0,0,0,0,0,0,'{}');
"""


def test_transformer_on_card_inserts():
    assert transform(CARDS) == [Insert(table=Table.Cards, data=SQLCard(cid=1651363200000,nid=1651363200000,did=1,ord=0))]


DECKS = r"""
INSERT INTO decks(id,name,mtime_secs,usn,common,kind) VALUES(1651363201000,'Default'||X'1f'
||'0',1651363201,-1,x'08011001',x'0a020801');
INSERT INTO graves(oid,type,usn) VALUES(1651363201001,2,-1);
"""


def test_transformer_on_decks_from_v18_schema():
    assert transform(DECKS) == [Insert(table=Table.Decks, data=SQLDeck(did=1651363201000,deckname=("Default", "0")))]


FIELDS = r"""
UPDATE config SET mtime_secs=1651363200, val=x'31363637303631313439373933' WHERE "KEY"='curModel';
DELETE FROM fields WHERE ntid=1667061149792 AND ord=0;
DELETE FROM fields WHERE ntid=1667061149792 AND ord=1;
INSERT INTO graves(oid,type,usn) VALUES(1651363200000,2,-1);
DELETE FROM notetypes WHERE id=1667061149792;
DELETE FROM templates WHERE ntid=1667061149792 AND ord=0;
"""


def test_transformer_on_field_deletions_from_v18_schema():
    assert transform(FIELDS) == [
        NtDelete(table=Table.Fields, ntid=1667061149792, ord=0),
        NtDelete(table=Table.Fields, ntid=1667061149792, ord=1),
        Delete(table=Table.Notetypes, row=1667061149792),
        NtDelete(table=Table.Templates, ntid=1667061149792, ord=0),
    ]
