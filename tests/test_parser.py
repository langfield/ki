from pathlib import Path
from tqdm import tqdm
from lark import Lark
from lark.exceptions import UnexpectedToken, UnexpectedCharacters
from loguru import logger
import pytest

def get_parser():
    """Return a parser."""
    # Read grammar.
    grammar_path = Path(__file__).resolve().parent.parent / "grammar.lark"
    grammar = grammar_path.read_text()

    # Instantiate parser.
    parser = Lark(grammar, start="file", parser="lalr")

    return parser


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


def test_parser_bads():
    """Try all bad note examples."""
    parser = get_parser()
    bads = Path("tests/data/notes/bad.md").read_text().split("---\n")
    logger.info(f"Length bads: {len(bads)}")
    for bad in bads:
        try:
            parser.parse(bad)
            logger.error(f"\n{bad}")
            assert False, "Should raise an error"
        except (UnexpectedToken, UnexpectedCharacters) as err:
            logger.info(err)
            continue
