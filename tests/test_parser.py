from pathlib import Path

import pytest
from tqdm import tqdm
from lark import Lark
from loguru import logger
from beartype import beartype
from lark.exceptions import UnexpectedToken, UnexpectedCharacters, UnexpectedInput

ASCII_CONTROLS = ["\0", "\a", "\b", "\t", "\n", "\v", "\f", "\r"]


def get_parser():
    """Return a parser."""
    # Read grammar.
    grammar_path = Path(__file__).resolve().parent.parent / "grammar.lark"
    grammar = grammar_path.read_text()

    # Instantiate parser.
    parser = Lark(grammar, start="file", parser="lalr")

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


TOO_MANY_HASHES_TITLE = r"""
### Note
nid: 123412341234
model: Basic
deck: a
tags:
markdown: false

### Front
r

### Back
s
"""


def test_too_many_hashes_for_title():
    """Do too many hashes in title cause parse error?"""
    note = TOO_MANY_HASHES_TITLE
    parser = get_parser()
    with pytest.raises(UnexpectedToken) as exc:
        parser.parse(note)
    err = exc.value
    assert err.line == 2
    assert err.column == 1
    assert err.token == "###"
    assert err.token_history == None


TOO_FEW_HASHES_TITLE = r"""
# Note
nid: 123412341234
model: Basic
deck: a
tags:
markdown: false

### Front
r

### Back
s
"""


def test_too_few_hashes_for_title():
    """Do too few hashes in title cause parse error?"""
    note = TOO_FEW_HASHES_TITLE
    parser = get_parser()
    with pytest.raises(UnexpectedToken) as exc:
        parser.parse(note)
    err = exc.value
    assert err.line == 2
    assert err.column == 1
    assert err.token == "# Note\n"
    assert err.token_history == None


TOO_FEW_HASHES_FIELDNAME = r"""
## Note
nid: 123412341234
model: Basic
deck: a
tags:
markdown: false

## Front
r

### Back
s
"""


def test_too_few_hashes_for_fieldname():
    """Do too many hashes in fieldname cause parse error?"""
    note = TOO_FEW_HASHES_FIELDNAME
    parser = get_parser()
    with pytest.raises(UnexpectedToken) as exc:
        parser.parse(note)
    err = exc.value
    assert err.line == 9
    assert err.column == 1
    assert err.token == "##"
    assert err.expected == set(["FIELDSENTINEL"])
    assert len(err.token_history) == 1
    prev = err.token_history.pop()
    assert str(prev) == "markdown: false\n\n"


TOO_MANY_HASHES_FIELDNAME = r"""
## Note
nid: 123412341234
model: Basic
deck: a
tags:
markdown: false

#### Front
r

### Back
s
"""


def test_too_many_hashes_for_fieldname():
    """Do too many hashes in fieldname cause parse error?"""
    note = TOO_MANY_HASHES_FIELDNAME
    parser = get_parser()
    with pytest.raises(UnexpectedToken) as exc:
        parser.parse(note)
    err = exc.value
    assert err.line == 9
    assert err.column == 4
    assert err.token == "# Front\n"
    assert err.expected == set(["ANKINAME"])
    assert len(err.token_history) == 1
    prev = err.token_history.pop()
    assert str(prev) == "###"


MISSING_FIELDNAME = r"""
## Note
nid: 123412341234
model: Basic
deck: a
tags:
markdown: false

###    
r

### Back
s
"""


def test_missing_fieldname():
    """Does a missing fieldname raise a parse error?"""
    note = MISSING_FIELDNAME
    parser = get_parser()
    with pytest.raises(UnexpectedToken) as exc:
        parser.parse(note)
    err = exc.value
    assert err.line == 9
    assert err.column == 8
    assert err.token == "\n"
    assert err.expected == set(["ANKINAME"])
    assert len(err.token_history) == 1
    prev = err.token_history.pop()
    assert str(prev) == "###"


MISSING_TITLE = r"""
##
nid: 123412341234
model: Basic
deck: a
tags:
markdown: false

### a
r

### b
s
"""


def test_missing_title():
    """Does a missing title raise a parse error?"""
    note = MISSING_TITLE
    parser = get_parser()
    with pytest.raises(UnexpectedToken) as exc:
        parser.parse(note)
    err = exc.value
    assert err.line == 2
    assert err.column == 3
    assert err.token == "\n"
    assert err.expected == set(["TITLENAME"])
    assert len(err.token_history) == 1
    prev = err.token_history.pop()
    assert str(prev) == "##"


MISSING_MODEL = r"""
##a
nid: 123412341234
model:
deck: a
tags:
markdown: false

### a
r

### b
s
"""


def test_missing_model():
    """Does a missing model raise a parse error?"""
    note = MISSING_MODEL
    parser = get_parser()
    with pytest.raises(UnexpectedToken) as exc:
        parser.parse(note)
    err = exc.value
    assert err.line == 4
    assert err.column == 1
    assert err.token == "model"
    assert err.expected == set(["MODEL"])
    assert len(err.token_history) == 1
    prev = err.token_history.pop()
    assert str(prev) == "nid: 123412341234\n"


WHITESPACE_MODEL = r"""
##a
nid: 123412341234
model:          	
deck: a
tags:
markdown: false

### a
r

### b
s
"""


def test_whitespace_model():
    """Does a whitespace model raise a parse error?"""
    note = WHITESPACE_MODEL
    parser = get_parser()
    with pytest.raises(UnexpectedToken) as exc:
        parser.parse(note)
    err = exc.value
    assert err.line == 4
    assert err.column == 1
    assert err.token == "model"
    assert err.expected == set(["MODEL"])
    assert len(err.token_history) == 1
    prev = err.token_history.pop()
    assert str(prev) == "nid: 123412341234\n"


FIELDNAME_VALIDATION = r"""
## a
nid: 123412341234
model: a
deck: a
tags:
markdown: false

### @@@@@
r

### b
s
"""

BAD_FIELDNAME_CHARS = [":", "{", "}", '"']


def test_bad_field_single_char_name_validation():
    """Do invalid fieldname characters raise an error?"""
    template = FIELDNAME_VALIDATION
    parser = get_parser()
    for fieldname in BAD_FIELDNAME_CHARS:
        note = template.replace("@@@@@", fieldname)
        with pytest.raises(UnexpectedInput) as exc:
            parser.parse(note)
        err = exc.value
        assert err.line == 9
        assert err.column == 5
        assert err.token == fieldname + "\n"
        assert err.expected == set(["ANKINAME"])
        assert len(err.token_history) == 1
        prev = err.token_history.pop()
        assert str(prev) == "###"


BAD_FIELDNAMES = ["aa:aa", "aa{aa", "aa}aa", 'aa"aa']


def test_bad_field_multi_char_name_validation():
    """Do invalid fieldname characters raise an error?"""
    template = FIELDNAME_VALIDATION
    parser = get_parser()
    for fieldname in BAD_FIELDNAMES:
        note = template.replace("@@@@@", fieldname)
        with pytest.raises(UnexpectedInput) as exc:
            parser.parse(note)
        err = exc.value
        assert err.line == 9
        assert err.column == 7
        assert err.token == fieldname[2:] + "\n"
        assert err.expected == set(["NEWLINE"])
        assert len(err.token_history) == 1
        prev = err.token_history.pop()
        assert str(prev) == fieldname[:2]


BAD_START_FIELDNAME_CHARS = ["#", "/", "^"]


def test_fieldname_start_validation():
    """Do bad start characters in fieldnames raise an error?"""
    template = FIELDNAME_VALIDATION
    parser = get_parser()
    for char in BAD_START_FIELDNAME_CHARS:
        fieldname = char + "a"
        note = template.replace("@@@@@", fieldname)
        with pytest.raises(UnexpectedInput) as exc:
            parser.parse(note)
        err = exc.value
        assert err.line == 9
        assert err.column == 5
        assert err.token == fieldname + "\n"
        assert err.expected == set(["ANKINAME"])
        assert len(err.token_history) == 1
        prev = err.token_history.pop()
        assert str(prev) == "###"


def test_parser_goods():
    """Try all good note examples."""
    parser = get_parser()
    goods = Path("tests/data/notes/good.md").read_text().split("---\n")
    logger.info(f"Length goods: {len(goods)}")
    for good in goods:
        try:
            parser.parse(good)
        except UnexpectedToken as err:
            logger.error(f"\n{good}")
            raise err


def main():
    parser = get_parser()

    # Read example note.
    note = Path("tests/data/notes/note123412341234.md").read_text()

    # Parse.
    tree = parser.parse(note)
    logger.debug(tree.pretty())

    # Parse all notes in a collection.
    for path in tqdm(set((Path.home() / "collection").iterdir())):
        if path.suffix == ".md":
            note = path.read_text()
            parser.parse(note)


if __name__ == "__main__":
    main()
