from pathlib import Path
from tqdm import tqdm
from lark import Lark
from loguru import logger

grammar_path = Path(__file__).resolve().parent.parent / "grammar.lark"
grammar = grammar_path.read_text()
logger.debug(grammar)
note = Path("tests/data/notes/note123412341234.md").read_text()
parser = Lark(grammar, start="file")
tree = parser.parse(note)
logger.debug(tree.pretty())

for path in tqdm(set((Path.home() / "collection").iterdir())):
    if path.suffix == ".md":
        note = path.read_text()
        parser.parse(note)
