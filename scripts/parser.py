from pathlib import Path
from lark import Lark
from loguru import logger

grammar_path = Path(__file__).resolve().parent.parent / "grammar.lark"
grammar = grammar_path.read_text()
logger.debug(grammar)
note = Path("tests/data/notes/note123412341234.md").read_text()
parser = Lark(grammar, start="file")
tree = parser.parse(note)
logger.debug(tree.pretty())

for path in (Path.home() / "collection").iterdir():
    logger.debug(path)
    note = path.read_text()
    parser.parse(note)
