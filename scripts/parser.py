from pathlib import Path
from lark import Lark
from loguru import logger

grammar = Path("grammar.lark").read_text()
logger.debug(grammar)
note = Path("tests/data/notes/note123412341234.md").read_text()
parser = Lark(grammar, start="file")
tree = parser.parse(note)
logger.debug(tree.pretty())
