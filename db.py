import shutil
from functools import reduce
from itertools import starmap
from dataclasses import dataclass

import mysql.connector
import prettyprinter as pp
from beartype import beartype
from beartype.typing import List, Union, Iterable, Dict
from anki.collection import Collection

import ki.functional as F


Row = List[Union[str, int]]
Table = List[Row]

FILE = "tests/data/collections/original.anki2"
TABLES = ["cards", "col", "graves", "notes", "revlog"]
TYPEMAP = {"INTEGER": "BIGINT", "TEXT": "TEXT"}


@beartype
@dataclass(frozen=True, eq=True)
class SQLiteField:
    name: str
    t: str
    pkey: bool


@beartype
@dataclass(frozen=True, eq=True)
class MySQLField:
    name: str
    t: str
    pkey: bool


@beartype
def get_sqlite_fields(col: Collection) -> Iterable[SQLiteField]:
    xss = map(lambda x: col.db.execute(f"PRAGMA TABLE_INFO('{x}')"), TABLES)
    field_fn = lambda f: SQLiteField(name=f[1], t=f[2], pkey=f[1] in ("id", "oid"))
    ts = map(lambda fs: map(field_fn, fs), xss)
    return ts


def main() -> None:
    """Convert collection to MySQL database."""
    # Connect to the MySQL DB.
    db = mysql.connector.connect(
        host="localhost", user="root", password="", database=""
    )
    cursor = db.cursor()

    # Open the collection and select all tables.
    col = Collection(shutil.copyfile(FILE, F.mkdtemp() / "col.anki2"))
    tables: List[Table] = list(map(lambda n: col.db.all(f"SELECT * FROM {n}"), TABLES))
    assert len(tables) == len(TABLES)
    maps = starmap(lambda data, name: {name: data}, zip(tables, TABLES))
    tablemap: Dict[str, Table] = reduce(lambda a, b: a | b, maps)

    # Get schemas and convert to MySQL.
    convert_fn = lambda f: MySQLField(name=f.name, t=TYPEMAP[f.t], pkey=f.pkey)
    show_fn = lambda f: f"`{f.name}` {f.t}" + (" PRIMARY KEY" if f.pkey else "")
    msqts = map(lambda fs: map(convert_fn, fs), get_sqlite_fields(col))
    args = map(lambda fs: "(" + ", ".join(map(show_fn, fs)) + ")", msqts)

    cursor.execute(f"DROP DATABASE root")
    cursor.execute(f"CREATE DATABASE root")
    cursor.execute(f"USE root")
    for name, arg in zip(TABLES, args):
        print(arg)
        cursor.execute(f"CREATE TABLE {name} {arg}")
        dummy = "(" + ", ".join(["%s"] * 18) + ")"
        print(len(dummy))
        for row in tablemap[name]:
            print(len(row))
            cursor.execute(f"INSERT INTO {name} VALUES {dummy}", tuple(row))


if __name__ == "__main__":
    pp.install_extras(exclude=["ipython", "django", "ipython_repr_pretty"])
    main()
