"""Tests for SQLite Lark grammar."""
from pathlib import Path
from dataclasses import dataclass

import pytest
import prettyprinter as pp
from beartype import beartype
from beartype.typing import List

import ki
from ki.types import SQLNote, SQLCard, SQLDeck, SQLField, SQLNotetype, SQLTemplate
from ki.sqlite import SQLiteTransformer, Statement, CardsUpdate, DecksUpdate, Column, FieldsDelete, NotesDelete, NotetypesDelete, TemplatesDelete, NotesUpdate, FieldsUpdate
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
        NotesDelete(row=1645010162168),
        NotesDelete(row=1645027705329),
        SQLNote(mid=1651202298367, guid="D(l.-iAXR1", tags=(), flds=("a", "b[sound:1sec.mp3]")),
        SQLNote(mid=1651232485378, guid="H0%>O*C~M!", tags=(), flds=("Who introduced the notion of a direct integral in functional analysis?", "John von Neumann")),
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
        NotesUpdate(updates=((Column.flds, ("aa", "bb")),), row=1645010162168),
        NotesDelete(row=1645027705329),
        SQLNote(mid=1645010146011, guid="f}:^>jzMjG", tags=(), flds=("f", "g")),
    ]


EMPTYISH = r"""
INSERT INTO notes(id,guid,mid,mod,usn,tags,flds,sfld,csum,flags,data) VALUES(1651363200000,'ku}V%9e9,l',1667061149792,1651363200,-1,'',''||X'1f','',3661210606,0,'');
"""


def test_transformer_on_empty_insert():
    assert transform(EMPTYISH) == [SQLNote(mid=1667061149792, guid="ku}V%9e9,l", tags=(), flds=("", ""))]


QUOTES = r"""
INSERT INTO notes(id,guid,mid,mod,usn,tags,flds,sfld,csum,flags,data) VALUES(1651363200000,'|o/qdllw(',1667061149792,1651363200,-1,'',''||X'1f'
||'''','',3661210606,0,'');
"""


def test_transformer_on_single_quotes():
    assert transform(QUOTES) == [SQLNote(mid=1667061149792, guid="|o/qdllw(", tags=(), flds=("", "'"))]


THREE_FIELDS = r"""
INSERT INTO notes(id,guid,mid,mod,usn,tags,flds,sfld,csum,flags,data) VALUES(1651363200001,'roH<$&er7G',1667061149794,1651363200,-1,'',''||X'1f1f','',3661210606,0,'');
"""


def test_transformer_on_three_fields():
    assert transform(THREE_FIELDS) == [SQLNote(mid=1667061149794, guid="roH<$&er7G", tags=(), flds=("", "", ""))]


TRAILING_EMPTY_FIELD = r"""
INSERT INTO notes(id,guid,mid,mod,usn,tags,flds,sfld,csum,flags,data) VALUES(1651363200001,'r#){>b5q^b',1667061149794,1651363200,-1,'',''||X'1f'
||'0'||X'1f','',3661210606,0,'');
"""


def test_transformer_on_trailing_empty_field():
    assert transform(TRAILING_EMPTY_FIELD) == [SQLNote(mid=1667061149794, guid="r#){>b5q^b", tags=(), flds=("", "0", ""))]


ESCAPED_SINGLE_QUOTE_AS_SORT_FIELD = r"""
INSERT INTO notes(id,guid,mid,mod,usn,tags,flds,sfld,csum,flags,data) VALUES(1651363200000,'OJl<Xj<>{H',1667061149792,1651363200,-1,'',''''||X'1f','',3143146758,0,'');
"""


def test_transformer_on_single_quote_in_sort_field():
    assert transform(ESCAPED_SINGLE_QUOTE_AS_SORT_FIELD) == [SQLNote(mid=1667061149792, guid="OJl<Xj<>{H", tags=(), flds=("'", ""))]


INTEGER_SORT_FIELD = r"""
INSERT INTO notes(id,guid,mid,mod,usn,tags,flds,sfld,csum,flags,data) VALUES(1651363200000,'F,y8PO5.KP',1667061149792,1651363200,-1,'','0'||X'1f',0,3059261382,0,'');
"""


def test_transformer_on_integer_sort_field():
    assert transform(INTEGER_SORT_FIELD) == [SQLNote(mid=1667061149792, guid="F,y8PO5.KP", tags=(), flds=("0", ""))]


RAW_BYTES_IN_SORT_FIELD = r"""
INSERT INTO notes(id,guid,mid,mod,usn,tags,flds,sfld,csum,flags,data) VALUES(1651363200001,'O#4+2LG`3{',1667061149792,1651363200,-1,'',''||X'0a1f',''||X'0a',2915580697,0,'');
"""


def test_transformer_on_raw_bytes_in_sort_field():
    assert transform(RAW_BYTES_IN_SORT_FIELD) == [SQLNote(mid=1667061149792, guid="O#4+2LG`3{", tags=(), flds=("\x0a", ""))]


SURROGATE_UTF_BYTES = r"""
INSERT INTO notes(id,guid,mid,mod,usn,tags,flds,sfld,csum,flags,data) VALUES(1651363217005,'c5zvCfp,h%',1667061149796,1651363217,-1,'',',-򁿈AÜ'||X'1f',',-򁿈AÜ',3950683052,0,'');
"""


def test_transformer_on_surrogate_utf_bytes():
    assert transform(SURROGATE_UTF_BYTES) == [SQLNote(mid=1667061149796, guid="c5zvCfp,h%", tags=(), flds=(",-򁿈AÜ", ""))]


DECIMALS_IN_SORT_FIELD = r"""
INSERT INTO notes(id,guid,mid,mod,usn,tags,flds,sfld,csum,flags,data) VALUES(1651363200000,'q6qBOHAT=6',1667061149792,1651363200,-1,'','.1'||X'1f',0.1,162134904,0,'');
"""


def test_transformer_on_decimals_in_sort_field():
    assert transform(DECIMALS_IN_SORT_FIELD) == [SQLNote(mid=1667061149792, guid="q6qBOHAT=6", tags=(), flds=(".1", ""))]


with open(Path(ki.__file__).parent.parent / "tests" / "update.sql", "r", encoding="UTF-8") as f:
    COL_UPDATE = f.read()


def test_transformer_on_col_update_queries():
    transform(COL_UPDATE)


CARDS = r"""
INSERT INTO cards(id,nid,did,ord,mod,usn,type,queue,due,ivl,factor,reps,lapses,"left",odue,odid,flags,data) VALUES(1651363200000,1651363200000,1,0,1651363200,-1,0,0,1,0,0,0,0,0,0,0,0,'{}');
"""


def test_transformer_on_card_inserts():
    assert transform(CARDS) == [SQLCard(cid=1651363200000,nid=1651363200000,did=1,ord=0)]


DECKS = r"""
INSERT INTO decks(id,name,mtime_secs,usn,common,kind) VALUES(1651363201000,'Default'||X'1f'
||'0',1651363201,-1,x'08011001',x'0a020801');
INSERT INTO graves(oid,type,usn) VALUES(1651363201001,2,-1);
"""


def test_transformer_on_decks_from_v18_schema():
    assert transform(DECKS) == [SQLDeck(did=1651363201000,deckname=("Default", "0"))]


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
        FieldsDelete(ntid=1667061149792, ord=0),
        FieldsDelete(ntid=1667061149792, ord=1),
        NotetypesDelete(row=1667061149792),
        TemplatesDelete(ntid=1667061149792, ord=0),
    ]


TEMPLATES = r"""
UPDATE config SET mtime_secs=1651363200, val=x'31363531333633323030303030' WHERE "KEY"='curModel';
INSERT INTO fields(ntid,ord,name,config) VALUES(1651363200000,0,'0',x'1a05417269616c2014');
INSERT INTO graves(oid,type,usn) VALUES(1651363200000,2,-1);
INSERT INTO notetypes(id,name,mtime_secs,usn,config) VALUES(1651363200000,'0',1651363200,-1,x'1a7e2e63617264207b0a20202020666f6e742d66616d696c793a20617269616c3b0a20202020666f6e742d73697a653a20323070783b0a20202020746578742d616c69676e3a2063656e7465723b0a20202020636f6c6f723a20626c61636b3b0a202020206261636b67726f756e642d636f6c6f723a2077686974653b0a7d0a2ab2015c646f63756d656e74636c6173735b313270745d7b61727469636c657d0a5c7370656369616c7b706170657273697a653d33696e2c35696e7d0a5c7573657061636b6167655b757466385d7b696e707574656e637d0a5c7573657061636b6167657b616d7373796d622c616d736d6174687d0a5c706167657374796c657b656d7074797d0a5c7365746c656e6774687b5c706172696e64656e747d7b30696e7d0a5c626567696e7b646f63756d656e747d0a320e5c656e647b646f63756d656e747d420510011a0100');
INSERT INTO templates(ntid,ord,name,mtime_secs,usn,config) VALUES(1651363200000,0,'0',0,0,x'0a057b7b307d7d12057b7b307d7d');
"""


def test_transformer_on_schema_18_template_insertions():
    assert transform(TEMPLATES) == [
        SQLField(ntid=1651363200000, ord=0, fieldname="0"),
        SQLNotetype(ntid=1651363200000, ntname='0'),
        SQLTemplate(ntid=1651363200000, ord=0, tmplname='0'),
    ]


DECK_UPDATE = r"""
UPDATE decks SET name='blank', mtime_secs=1651363202 WHERE id=1651363201000;
INSERT INTO graves(oid,type,usn) VALUES(1651363202000,2,-1);
"""


def test_transformer_on_schema_18_deck_updates():
    assert transform(DECK_UPDATE) == [
        DecksUpdate(updates=((Column.name, "blank"),), row=1651363201000),
    ]


CARD_INSERT = r"""
INSERT INTO cards(id,nid,did,ord,mod,usn,type,queue,due,ivl,factor,reps,lapses,"left",odue,odid,flags,data) VALUES(1651363200000,1651363200000,1,0,1651363200,-1,0,0,1,0,0,0,0,0,0,0,0,'{}');
INSERT INTO config("KEY",usn,mtime_secs,val) VALUES('_deck_1_lastNotetype',-1,1651363200,x'31363531333633323030303030');
INSERT INTO config("KEY",usn,mtime_secs,val) VALUES('_nt_1651363200000_lastDeck',-1,1651363200,x'31');
UPDATE config SET mtime_secs=1651363200, val=x'31363531333633323030303030' WHERE "KEY"='curModel';
UPDATE config SET usn=-1, mtime_secs=1651363200, val=x'32' WHERE "KEY"='nextPos';
INSERT INTO fields(ntid,ord,name,config) VALUES(1651363200000,0,'0',x'1a05417269616c2014');
INSERT INTO graves(oid,type,usn) VALUES(1651363200000,2,-1);
INSERT INTO notes(id,guid,mid,mod,usn,tags,flds,sfld,csum,flags,data) VALUES(1651363200000,'wC&{B}s3O|',1651363200000,1651363200,-1,'','','',3661210606,0,'');
INSERT INTO notetypes(id,name,mtime_secs,usn,config) VALUES(1651363200000,'0',1651363200,-1,x'1a7e2e63617264207b0a20202020666f6e742d66616d696c793a20617269616c3b0a20202020666f6e742d73697a653a20323070783b0a20202020746578742d616c69676e3a2063656e7465723b0a20202020636f6c6f723a20626c61636b3b0a202020206261636b67726f756e642d636f6c6f723a2077686974653b0a7d0a2ab2015c646f63756d656e74636c6173735b313270745d7b61727469636c657d0a5c7370656369616c7b706170657273697a653d33696e2c35696e7d0a5c7573657061636b6167655b757466385d7b696e707574656e637d0a5c7573657061636b6167657b616d7373796d622c616d736d6174687d0a5c706167657374796c657b656d7074797d0a5c7365746c656e6774687b5c706172696e64656e747d7b30696e7d0a5c626567696e7b646f63756d656e747d0a320e5c656e647b646f63756d656e747d420510011a0100');
INSERT INTO templates(ntid,ord,name,mtime_secs,usn,config) VALUES(1651363200000,0,'0',0,0,x'0a057b7b307d7d12057b7b307d7d');
"""


def test_transformer_on_schema_18_card_insert_redux():
    assert transform(CARD_INSERT) == [
        SQLCard(cid=1651363200000,nid=1651363200000,did=1,ord=0),
        SQLField(ntid=1651363200000,ord=0,fieldname="0"),
        SQLNote(guid="wC&{B}s3O|", mid=1651363200000, tags=(), flds=("",)),
        SQLNotetype(ntid=1651363200000, ntname='0'),
        SQLTemplate(ntid=1651363200000, ord=0, tmplname='0'),
    ]


NOTE_SET_DUE = r"""
UPDATE cards SET due=2 WHERE id=1651363200000;
UPDATE config SET val=x'33' WHERE "KEY"='nextPos';
INSERT INTO graves(oid,type,usn) VALUES(1651363200000,0,-1);
INSERT INTO graves(oid,type,usn) VALUES(1651363200000,1,-1);
UPDATE notes SET guid='g;a&:e3_^~' WHERE id=1651363200000;
"""


def test_transformer_on_schema_18_note_set_due():
    assert transform(NOTE_SET_DUE) == [
        NotesUpdate(updates=((Column.guid, "g;a&:e3_^~"),), row=1651363200000),
    ]


NOTETYPE_INSERT = r"""
INSERT INTO notetypes(id,name,mtime_secs,usn,config) VALUES(1651363200000,''||X'1f',1651363200,-1,x'1a7e2e63617264207b0a20202020666f6e742d66616d696c793a20617269616c3b0a20202020666f6e742d73697a653a20323070783b0a20202020746578742d616c69676e3a2063656e7465723b0a20202020636f6c6f723a20626c61636b3b0a202020206261636b67726f756e642d636f6c6f723a2077686974653b0a7d0a2ab2015c646f63756d656e74636c6173735b313270745d7b61727469636c657d0a5c7370656369616c7b706170657273697a653d33696e2c35696e7d0a5c7573657061636b6167655b757466385d7b696e707574656e637d0a5c7573657061636b6167657b616d7373796d622c616d736d6174687d0a5c706167657374796c657b656d7074797d0a5c7365746c656e6774687b5c706172696e64656e747d7b30696e7d0a5c626567696e7b646f63756d656e747d0a320e5c656e647b646f63756d656e747d420510011a0100');
"""


def test_transformer_on_schema_18_notetype_insert():
    assert transform(NOTETYPE_INSERT) == [
        SQLNotetype(ntid=1651363200000, ntname="\x1f")
    ]


UPDATE_COMPOSITE_PRIMARY_KEYS = r"""
UPDATE fields SET name='656' WHERE ntid=1667061149796 AND ord=0;
"""


def test_transformer_on_schema_18_composite_primary_keys():
    assert transform(UPDATE_COMPOSITE_PRIMARY_KEYS) == [
        FieldsUpdate(((Column.name, "656"),), ntid=1667061149796, ord=0)
    ]
