import mysql.connector
from anki.collection import Collection


def main() -> None:
    """Convert collection to MySQL database."""
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="",
    )


if __name__ == "__main__":
    main()
