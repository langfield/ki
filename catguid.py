"""Cat GUIDs."""
import sys
from anki.collection import Collection


def main() -> None:
    """Cat GUIDs."""
    col = Collection(sys.argv[1])
    for nid in col.find_notes(""):
        note = col.get_note(nid)
        print("===============")
        print(note.keys())
        print(note.fields)
        print(note.guid)


if __name__ == "__main__":
    main()
