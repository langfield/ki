import time
import shutil
from itertools import starmap
from functools import reduce, partial
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
FILE = "/home/mal/collection.anki2"
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
def get_sqlite_fields(col: Collection, table: str) -> List[SQLiteField]:
    field_fn = lambda f: SQLiteField(name=f[1], t=f[2], pkey=f[1] in ("id", "oid"))
    return list(map(field_fn, col.db.execute(f"PRAGMA TABLE_INFO('{table}')")))


@beartype
def create_table(
    cursor: mysql.connector.cursor.MySQLCursor,
    tablemap: Dict[str, Table],
    table: str,
    ncols: int,
    arg: str,
) -> None:
    cursor.execute(f"CREATE TABLE {table} {arg}")
    dummy = "(" + ", ".join(["%s"] * ncols) + ")"
    cursor.executemany(f"INSERT INTO {table} VALUES {dummy}", tablemap[table])


def main() -> None:
    """Convert collection to MySQL database."""
    # Connect to the MySQL DB.
    t = time.time()
    db = mysql.connector.connect(
        host="localhost", user="root", password="", database=""
    )
    cursor = db.cursor()

    # Open the collection and select all tables.
    col = Collection(shutil.copyfile(FILE, F.mkdtemp() / "col.anki2"))
    tables: List[Table] = list(map(lambda n: col.db.all(f"SELECT * FROM {n}"), TABLES))

    maps = starmap(lambda data, name: {name: data}, zip(tables, TABLES))
    tablemap: Dict[str, Table] = reduce(lambda a, b: a | b, maps)

    # Get schemas and convert to MySQL.
    type_fn = lambda n, t: "TEXT" if n == "sfld" else TYPEMAP[t]
    convert_fn = lambda f: MySQLField(name=f.name, t=type_fn(f.name, f.t), pkey=f.pkey)
    show_fn = lambda f: f"`{f.name}` {f.t}" + (" PRIMARY KEY" if f.pkey else "")
    arg_fn = lambda fs: "(" + ", ".join(map(show_fn, fs)) + ")"
    tfs = map(lambda t: (t, get_sqlite_fields(col, t)), TABLES)
    msqts = starmap(lambda n, fs: (n, list(map(convert_fn, fs))), tfs)
    args = starmap(lambda n, fs: (n, len(fs), arg_fn(fs)), msqts)

    cursor.execute(f"DROP DATABASE root")
    cursor.execute(f"CREATE DATABASE root")
    cursor.execute(f"USE root")
    _ = set(starmap(partial(create_table, cursor, tablemap), args))
    db.commit()
    print(f"Elapsed: {time.time() - t}s")


if __name__ == "__main__":
    pp.install_extras(exclude=["ipython", "django", "ipython_repr_pretty"])
    main()
