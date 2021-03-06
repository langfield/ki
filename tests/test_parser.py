"""Tests for markdown note Lark grammar."""
from pathlib import Path

import pytest

from tqdm import tqdm
from loguru import logger
from beartype import beartype

from lark import Lark
from lark.exceptions import UnexpectedToken, UnexpectedInput, UnexpectedCharacters

import ki
from ki import NoteTransformer

# pylint: disable=too-many-lines

BAD_ASCII_CONTROLS = ["\0", "\a", "\b", "\v", "\f"]


def get_parser():
    """Return a parser."""
    # Read grammar.
    grammar_path = Path(ki.__file__).resolve().parent / "grammar.lark"
    grammar = grammar_path.read_text()

    # Instantiate parser.
    parser = Lark(grammar, start="note", parser="lalr")

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


TOO_MANY_HASHES_TITLE = r"""### Note
nid: 123412341234
model: Basic
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
    assert err.line == 1
    assert err.column == 3
    assert err.token == "# Note\n"
    assert len(err.token_history) == 1
    prev = err.token_history.pop()
    assert str(prev) == "##"


TOO_FEW_HASHES_TITLE = r"""# Note
nid: 123412341234
model: Basic
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
    assert err.line == 1
    assert err.column == 1
    assert err.token == "# Note\n"
    assert len(err.token_history) == 1
    assert err.token_history.pop() is None


TOO_FEW_HASHES_FIELDNAME = r"""## Note
nid: 123412341234
model: Basic
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
    assert err.line == 7
    assert err.column == 1
    assert err.token == "##"
    assert err.expected == set(["FIELDSENTINEL"])
    assert len(err.token_history) == 1
    prev = err.token_history.pop()
    assert str(prev) == "markdown: false\n\n"


TOO_MANY_HASHES_FIELDNAME = r"""## Note
nid: 123412341234
model: Basic
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
    assert err.line == 7
    assert err.column == 4
    assert err.token == "# Front\n"
    assert err.expected == set(["ANKINAME"])
    assert len(err.token_history) == 1
    prev = err.token_history.pop()
    assert str(prev) == "###"


MISSING_FIELDNAME = r"""## Note
nid: 123412341234
model: Basic
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
    assert err.line == 7
    assert err.column == 8
    assert err.token == "\n"
    assert err.expected == set(["ANKINAME"])
    assert len(err.token_history) == 1
    prev = err.token_history.pop()
    assert str(prev) == "###"


MISSING_TITLE = r"""##
nid: 123412341234
model: Basic
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
    assert err.line == 1
    assert err.column == 3
    assert err.token == "\n"
    assert err.expected == set(["TITLENAME"])
    assert len(err.token_history) == 1
    prev = err.token_history.pop()
    assert str(prev) == "##"


MISSING_MODEL = r"""##a
nid: 123412341234
model:
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
    assert err.line == 3
    assert err.column == 1
    assert err.token == "model"
    assert err.expected == set(["MODEL"])
    assert len(err.token_history) == 1
    prev = err.token_history.pop()
    assert str(prev) == "nid: 123412341234\n"


WHITESPACE_MODEL = r"""##a
nid: 123412341234
model:          	
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
    assert err.line == 3
    assert err.column == 1
    assert err.token == "model"
    assert err.expected == set(["MODEL"])
    assert len(err.token_history) == 1
    prev = err.token_history.pop()
    assert str(prev) == "nid: 123412341234\n"


FIELDNAME_VALIDATION = r"""## a
nid: 123412341234
model: a
tags:
markdown: false

### @@@@@
r

### b
s
"""

BAD_FIELDNAME_CHARS = [":", "{", "}", '"'] + BAD_ASCII_CONTROLS


def test_bad_field_single_char_name_validation():
    """Do invalid fieldname characters raise an error?"""
    template = FIELDNAME_VALIDATION
    parser = get_parser()
    for char in BAD_FIELDNAME_CHARS:
        note = template.replace("@@@@@", char)
        with pytest.raises(UnexpectedInput) as exc:
            parser.parse(note)
        err = exc.value

        assert err.line == 7
        assert err.column == 5
        assert len(err.token_history) == 1
        prev = err.token_history.pop()
        assert str(prev) == "###"
        if isinstance(err, UnexpectedToken):
            assert err.token in char + "\n"
            assert err.expected == set(["ANKINAME"])
        if isinstance(err, UnexpectedCharacters):
            assert err.char == char


def test_bad_field_multi_char_name_validation():
    """Do invalid fieldname characters raise an error?"""
    template = FIELDNAME_VALIDATION
    parser = get_parser()
    for char in BAD_FIELDNAME_CHARS:
        fieldname = "aa" + char + "aa"
        note = template.replace("@@@@@", fieldname)
        with pytest.raises(UnexpectedInput) as exc:
            parser.parse(note)
        err = exc.value
        assert err.line == 7
        assert err.column == 7
        assert len(err.token_history) == 1
        prev = err.token_history.pop()
        assert str(prev) == fieldname[:2]
        if isinstance(err, UnexpectedToken):
            assert err.token in fieldname[2:] + "\n"
            assert err.expected == set(["NEWLINE"])
        if isinstance(err, UnexpectedCharacters):
            assert err.char == char


BAD_START_FIELDNAME_CHARS = ["#", "/", "^"] + BAD_FIELDNAME_CHARS


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
        assert err.line == 7
        assert err.column == 5
        assert len(err.token_history) == 1
        prev = err.token_history.pop()
        assert str(prev) == "###"
        if isinstance(err, UnexpectedToken):
            assert err.token in fieldname + "\n"
            assert err.expected == set(["ANKINAME"])
        if isinstance(err, UnexpectedCharacters):
            assert err.char == char


FIELD_CONTENT_VALIDATION = r"""## a
nid: 123412341234
model: a
tags:
markdown: false

### a
@@@@@

### b
s
"""


def test_field_content_validation():
    """Do ascii control characters in fields raise an error?"""
    template = FIELD_CONTENT_VALIDATION
    parser = get_parser()
    for char in BAD_ASCII_CONTROLS:
        field = char + "a"
        note = template.replace("@@@@@", field)
        with pytest.raises(UnexpectedCharacters) as exc:
            parser.parse(note)
        err = exc.value
        assert err.line == 8
        assert err.column == 1
        assert err.char == char
        assert len(err.token_history) == 1
        prev = err.token_history.pop()
        assert str(prev) == "\n"


TAG_VALIDATION = r"""## a
nid: 123412341234
model: 0a
tags: @@@@@
markdown: false

### a
r

### b
s
"""

BAD_TAG_CHARS = ['"', "\u3000", " "] + BAD_ASCII_CONTROLS


def test_tag_validation():
    """Do ascii control characters and quotes in tag names raise an error?"""
    template = TAG_VALIDATION
    parser = get_parser()
    for char in BAD_TAG_CHARS:
        tags = f"subtle, {char}, heimdall"
        note = template.replace("@@@@@", tags)
        with pytest.raises(UnexpectedInput) as exc:
            parser.parse(note)
        err = exc.value
        assert err.line == 4
        assert err.column in (15, 16)
        assert len(err.token_history) == 1
        prev = err.token_history.pop()
        assert str(prev) == ","
        if isinstance(err, UnexpectedToken):
            remainder = ",".join(tags.split(",")[1:]) + "\n"
            assert err.token in remainder
            assert err.expected == set(["TAGNAME"])
        if isinstance(err, UnexpectedCharacters):
            assert err.char == char


def test_parser_goods():
    """Try all good note examples."""
    parser = get_parser()
    goods = Path("tests/data/notes/good.md").read_text(encoding="UTF-8").split("---\n")
    for good in goods:
        try:
            parser.parse(good)
        except UnexpectedToken as err:
            logger.error(f"\n{good}")
            raise err


def test_transformer():
    """Try out transformer."""
    parser = get_parser()
    note = Path("tests/data/notes/noteLARK.md").read_text(encoding="UTF-8")
    tree = parser.parse(note)
    transformer = NoteTransformer()
    transformer.transform(tree)


def test_transformer_goods():
    """Try all good note examples."""
    parser = get_parser()
    transformer = NoteTransformer()
    goods = Path("tests/data/notes/good.md").read_text(encoding="UTF-8").split("---\n")
    for good in goods:
        try:
            tree = parser.parse(good)
            transformer.transform(tree)
        except UnexpectedToken as err:
            logger.error(f"\n{good}")
            raise err


def main():
    """Parse all notes in main collection."""
    parse_collection()


def parse_collection():
    """Parse all notes in a collection."""
    transformer = NoteTransformer()
    grammar_path = Path(ki.__file__).resolve().parent / "grammar.lark"
    grammar = grammar_path.read_text()
    parser = Lark(grammar, start="file", parser="lalr", transformer=transformer)
    for path in tqdm(set((Path.home() / "collection").iterdir())):
        if path.suffix == ".md":
            note = path.read_text()
            parser.parse(note)


if __name__ == "__main__":
    main()
