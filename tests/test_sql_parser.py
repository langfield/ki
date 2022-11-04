"""Tests for SQLite Lark grammar."""

import prettyprinter as pp
from ki.sqlite import SQLiteTransformer
from tests.test_parser import get_parser, debug_lark_error


# pylint: disable=too-many-lines, missing-function-docstring


BLOCK = r"""
DELETE FROM notes WHERE id=1645010162168;
DELETE FROM notes WHERE id=1645027705329;
INSERT INTO notes(id,guid,mid,mod,usn,tags,flds,sfld,csum,flags,data) VALUES(1651202347045,'D(l.-iAXR1',1651202298367,1651202347,-1,'','a'||X'1f'
        ||'b[sound:1sec.mp3]','a',2264392759,0,'');
INSERT INTO notes(id,guid,mid,mod,usn,tags,flds,sfld,csum,flags,data) VALUES(1651232832286,'H0%>O*C~M!',1651232485378,1651232832,-1,'','Who introduced the notion of a direct integral in functional analysis?'||X'1f'
        ||'John von Neumann','Who introduced the notion of a direct integral in functional analysis?',2707929287,0,'');
"""


def test_basic():
    parser = get_parser(filename="sqlite.lark", start="diff")
    parser.parse(BLOCK)


def test_transformer():
    parser = get_parser(filename="sqlite.lark", start="diff")
    tree = parser.parse(BLOCK)
    transformer = SQLiteTransformer()
    _ = transformer.transform(tree)


UPDATE = r"""
UPDATE notes SET mod=1645221606, flds='aa'||X'1f'
||'bb', sfld='aa', csum=3771269976 WHERE id=1645010162168;
DELETE FROM notes WHERE id=1645027705329;
INSERT INTO notes(id,guid,mid,mod,usn,tags,flds,sfld,csum,flags,data) VALUES(1645222430007,'f}:^>jzMjG',1645010146011,1645222430,-1,'','f'||X'1f'
||'g','f',1242175777,0,'');
"""


def test_transformer_on_update_commands():
    parser = get_parser(filename="sqlite.lark", start="diff")
    tree = parser.parse(UPDATE)
    transformer = SQLiteTransformer()
    _ = transformer.transform(tree)


EMPTYISH = r"""
INSERT INTO notes(id,guid,mid,mod,usn,tags,flds,sfld,csum,flags,data) VALUES(1651363200000,'ku}V%9e9,l',1667061149792,1651363200,-1,'',''||X'1f','',3661210606,0,'');
"""


def test_transformer_on_empty_insert():
    parser = get_parser(filename="sqlite.lark", start="diff")
    tree = parser.parse(EMPTYISH)
    transformer = SQLiteTransformer()
    _ = transformer.transform(tree)


QUOTES = r"""
INSERT INTO notes(id,guid,mid,mod,usn,tags,flds,sfld,csum,flags,data) VALUES(1651363200000,'|o/qdllw(',1667061149792,1651363200,-1,'',''||X'1f'
||'''','',3661210606,0,'');
"""


def test_transformer_on_single_quotes():
    parser = get_parser(filename="sqlite.lark", start="diff")
    tree = parser.parse(QUOTES)
    transformer = SQLiteTransformer()
    _ = transformer.transform(tree)


THREE_FIELDS = r"""
INSERT INTO notes(id,guid,mid,mod,usn,tags,flds,sfld,csum,flags,data) VALUES(1651363200001,'roH<$&er7G',1667061149794,1651363200,-1,'',''||X'1f1f','',3661210606,0,'');
"""


def test_transformer_on_three_fields():
    parser = get_parser(filename="sqlite.lark", start="diff")
    tree = parser.parse(THREE_FIELDS)
    transformer = SQLiteTransformer()
    _ = transformer.transform(tree)


TRAILING_EMPTY_FIELD = r"""
INSERT INTO notes(id,guid,mid,mod,usn,tags,flds,sfld,csum,flags,data) VALUES(1651363200001,'r#){>b5q^b',1667061149794,1651363200,-1,'',''||X'1f'
||'0'||X'1f','',3661210606,0,'');
"""


def test_transformer_on_trailing_empty_field():
    parser = get_parser(filename="sqlite.lark", start="diff")
    tree = parser.parse(TRAILING_EMPTY_FIELD)
    transformer = SQLiteTransformer()
    out = transformer.transform(tree)
    pp.pprint(out)


ESCAPED_SINGLE_QUOTE_AS_SORT_FIELD = r"""
INSERT INTO notes(id,guid,mid,mod,usn,tags,flds,sfld,csum,flags,data) VALUES(1651363200000,'OJl<Xj<>{H',1667061149792,1651363200,-1,'',''''||X'1f','',3143146758,0,'');
"""


def test_transformer_on_single_quote_in_sort_field():
    parser = get_parser(filename="sqlite.lark", start="diff")
    try:
        tree = parser.parse(ESCAPED_SINGLE_QUOTE_AS_SORT_FIELD)
    except Exception as err:
        debug_lark_error(err)
        raise err
    transformer = SQLiteTransformer()
    out = transformer.transform(tree)
    pp.pprint(out)


INTEGER_SORT_FIELD = r"""
INSERT INTO notes(id,guid,mid,mod,usn,tags,flds,sfld,csum,flags,data) VALUES(1651363200000,'F,y8PO5.KP',1667061149792,1651363200,-1,'','0'||X'1f',0,3059261382,0,'');
"""


def test_transformer_on_integer_sort_field():
    parser = get_parser(filename="sqlite.lark", start="diff")
    try:
        tree = parser.parse(INTEGER_SORT_FIELD)
    except Exception as err:
        debug_lark_error(err)
        raise err
    transformer = SQLiteTransformer()
    out = transformer.transform(tree)
    pp.pprint(out)


RAW_BYTES_IN_SORT_FIELD = r"""
INSERT INTO notes(id,guid,mid,mod,usn,tags,flds,sfld,csum,flags,data) VALUES(1651363200001,'O#4+2LG`3{',1667061149792,1651363200,-1,'',''||X'0a1f',''||X'0a',2915580697,0,'');
"""


def test_transformer_on_raw_bytes_in_sort_field():
    parser = get_parser(filename="sqlite.lark", start="diff")
    try:
        tree = parser.parse(RAW_BYTES_IN_SORT_FIELD)
    except Exception as err:
        debug_lark_error(err)
        raise err
    transformer = SQLiteTransformer()
    out = transformer.transform(tree)
    pp.pprint(out)


SURROGATE_UTF_BYTES = r"""
INSERT INTO notes(id,guid,mid,mod,usn,tags,flds,sfld,csum,flags,data) VALUES(1651363217005,'c5zvCfp,h%',1667061149796,1651363217,-1,'',',-򁿈AÜ'||X'1f',',-򁿈AÜ',3950683052,0,'');
"""


def test_transformer_on_surrogate_utf_bytes():
    parser = get_parser(filename="sqlite.lark", start="diff")
    try:
        tree = parser.parse(SURROGATE_UTF_BYTES)
    except Exception as err:
        debug_lark_error(err)
        raise err
    transformer = SQLiteTransformer()
    out = transformer.transform(tree)
    pp.pprint(out)


DECIMALS_IN_SORT_FIELD = r"""
INSERT INTO notes(id,guid,mid,mod,usn,tags,flds,sfld,csum,flags,data) VALUES(1651363200000,'q6qBOHAT=6',1667061149792,1651363200,-1,'','.1'||X'1f',0.1,162134904,0,'');
"""


def test_transformer_on_decimals_in_sort_field():
    parser = get_parser(filename="sqlite.lark", start="diff")
    try:
        tree = parser.parse(DECIMALS_IN_SORT_FIELD)
    except Exception as err:
        debug_lark_error(err)
        raise err
    transformer = SQLiteTransformer()
    out = transformer.transform(tree)
    pp.pprint(out)
