"""A function for removing custom collations from a sqlite file."""
import sqlite3
import tempfile
import subprocess
from pathlib import Path
from beartype import beartype

import anki.collection
from anki.notes import Note
from anki.collection import Collection, NotetypeDict

import ki.functional as F
from ki import checkpoint_anki2
from ki.types import File
from tests.test_ki import get_test_collection


def test_checkpoint_anki2():
    empty = get_test_collection("empty")

    tempd = Path(tempfile.mkdtemp())
    path = F.chk(tempd / "collection.anki2")
    path = F.copyfile(empty.col_file, path)
    col = Collection(path)
    initial = checkpoint_anki2(path, "initial")

    nt: NotetypeDict = col.models.all()[0]
    note: Note = col.new_note(nt)
    note.fields = ["AAA", "BBB"]
    did = 1
    col.add_note(note, did)

    col.save()
    col.db.commit()
    col.close(save=True)
    final: File = checkpoint_anki2(path, "final")
    assert str(final) != str(initial)
    assert F.md5(final) != F.md5(initial)
    p = subprocess.run(
        ["sqldiff", str(initial), str(final)],
        capture_output=True,
        check=True,
    )
    block = p.stdout.decode() + p.stderr.decode()
    print(block)
