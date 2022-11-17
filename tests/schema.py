"""A function for removing custom collations from a sqlite file."""
import sqlite3
from beartype import beartype

import ki.functional as F
from ki.types import File

DECK_CONFIG = r"""CREATE TABLE deck_config (
  id integer PRIMARY KEY NOT NULL,
  name text NOT NULL,
  mtime_secs integer NOT NULL,
  usn integer NOT NULL,
  config blob NOT NULL
)"""

CONFIG = r"""CREATE TABLE config (
  KEY text NOT NULL PRIMARY KEY,
  usn integer NOT NULL,
  mtime_secs integer NOT NULL,
  val blob NOT NULL
) without rowid"""

FIELDS = r"""CREATE TABLE fields (
  ntid integer NOT NULL,
  ord integer NOT NULL,
  name text NOT NULL,
  config blob NOT NULL,
  PRIMARY KEY (ntid, ord)
) without rowid"""

TEMPLATES = r"""CREATE TABLE templates (
  ntid integer NOT NULL,
  ord integer NOT NULL,
  name text NOT NULL,
  mtime_secs integer NOT NULL,
  usn integer NOT NULL,
  config blob NOT NULL,
  PRIMARY KEY (ntid, ord)
) without rowid"""

NOTETYPES = r"""CREATE TABLE notetypes (
  id integer NOT NULL PRIMARY KEY,
  name text NOT NULL,
  mtime_secs integer NOT NULL,
  usn integer NOT NULL,
  config blob NOT NULL
)"""

DECKS = r"""CREATE TABLE decks (
  id integer PRIMARY KEY NOT NULL,
  name text NOT NULL,
  mtime_secs integer NOT NULL,
  usn integer NOT NULL,
  common blob NOT NULL,
  kind blob NOT NULL
)"""

TAGS = r"""CREATE TABLE tags (
  tag text NOT NULL PRIMARY KEY,
  usn integer NOT NULL,
  collapsed boolean NOT NULL,
  config blob NULL
) without rowid"""


@beartype
def checkpoint_anki2(file: File, label: str) -> File:
    """Checkpoint an Anki collection database file."""
    copy = F.copyfile(file, F.chk(F.mkdtemp() / f"{label}.anki2"))
    con = sqlite3.connect(copy)
    cur = con.cursor()
    tables = {
        "deck_config": DECK_CONFIG,
        "config": CONFIG,
        "fields": FIELDS,
        "templates": TEMPLATES,
        "notetypes": NOTETYPES,
        "decks": DECKS,
        "tags": TAGS,
    }
    cur.execute("PRAGMA writable_schema=1;")
    for name, sql in tables.items():
        cur.execute(
            f"UPDATE sqlite_master SET sql='{sql}' WHERE type='table' AND name='{name}';"
        )
    con.commit()
    con.close()
    con = sqlite3.connect(copy)
    cur = con.cursor()
    cur.execute("REINDEX;")
    con.commit()
    con.close()
    return copy
