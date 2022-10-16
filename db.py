import shutil
from functools import partial

import mysql.connector
import prettyprinter as pp
from beartype import beartype
from beartype.typing import List, Union
from anki.collection import Collection

import ki.functional as F


FILE = "tests/data/collections/original.anki2"
TABLES = ["cards", "col", "graves", "notes", "revlog"]


Row = List[Union[str, int]]
Table = List[Row]


@beartype
def get_table(col: Collection, table: str) -> Table:
    return col.db.all(f"select * from {table}")



def main() -> None:
    """Convert collection to MySQL database."""
    # Connect to the MySQL DB.
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="",
    )
    cursor = db.cursor()

    # Open the collection and select all tables.
    file = shutil.copyfile(FILE, F.mkdtemp() / "col.anki2")
    col = Collection(file)
    tables: Iterable[Table] = map(partial(get_table, col), TABLES)
    pp.pprint(list(tables))


if __name__ == "__main__":
    main()
