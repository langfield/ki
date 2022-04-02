"""Architecture for a cleaner version of _clone()."""
import re
import tempfile
import subprocess
from typing import Dict, List
from pathlib import Path

from tqdm import tqdm
from apy.anki import Anki

from ki import KiNote, get_sort_fieldname, get_notepath, get_batches, slugify

BATCH_SIZE = 500
HTML_REGEX = r"</?\s*[a-z-][^>]*\s*>|(\&(?:[\w\d]+|#\d+|#x[a-f\d]+);)"
TQDM_NUM_COLS = 70


def _clone(colpath: Path, targetdir: Path, silent: bool):

    # Create temp directory for htmlfield text files.
    root = Path(tempfile.mkdtemp()) / "ki" / "fieldhtml"
    root.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {}
    decks: Dict[str, List[KiNote]] = {}

    # Open deck with `apy`, and dump notes and markdown files.
    with Anki(path=colpath) as a:
        all_nids = list(a.col.find_notes(query=""))
        for nid in tqdm(all_nids, ncols=TQDM_NUM_COLS, disable=silent):
            kinote = KiNote(a, a.col.getNote(nid))
            decks[kinote.deck] = decks.get(kinote.deck, []) + [kinote]
            for fieldname, fieldtext in kinote.fields.items():
                if re.search(HTML_REGEX, fieldtext):
                    fid = get_field_note_id(nid, fieldname)
                    paths[fid] = dump(fid, fieldtext, root)
        tidy(paths, silent)
        for deckname in sorted(set(decks.keys()), key=len, reverse=True):
            deckpath = get_and_create_deckpath(deckname, targetdir)
            for kinote in decks[deckname]:
                sort_fieldname = get_sort_fieldname(a, kinote.n.note_type())
                notepath = get_notepath(kinote, sort_fieldname, deckpath)
                payload = get_tidy_payload(kinote, paths)
                notepath.write_text(payload, encoding="UTF-8")


def dump(fid: str, fieldtext: str, root: Path) -> Path:
    """Dump field with given fid to a unique path."""
    htmlpath = root / fid
    htmlpath.write_text(fieldtext)
    return htmlpath


def tidy(paths: Dict[str, Path], silent: bool) -> None:
    """Spin up subprocesses for tidying field HTML in-place."""
    path_strings = [str(path) for path in paths.values()]
    batches = list(get_batches(path_strings, BATCH_SIZE))
    for batch in tqdm(batches, ncols=TQDM_NUM_COLS, disable=silent):
        pathsline = " ".join(batch)
        command = f"tidy -q -m -i -omit -utf8 --tidy-mark no {pathsline}"
        subprocess.run(command, shell=True, check=False, capture_output=True)


def get_and_create_deckpath(deckname: str, targetdir: Path) -> Path:
    """Construct path to deck directory and create it."""
    # Strip leading periods so we don't get hidden folders.
    components = deckname.split("::")
    components = [re.sub(r"^\.", r"", comp) for comp in components]
    deckpath = Path(targetdir, *components)
    deckpath.mkdir(parents=True, exist_ok=True)
    return deckpath


def get_field_note_id(nid: int, fieldname: str) -> str:
    """A str ID that uniquely identifies field-note pairs."""
    return f"{nid}{slugify(fieldname, allow_unicode=True)}"


def get_tidy_payload(kinote: KiNote, paths: Dict[str, Path]) -> str:
    """Get the payload for the note (HTML-tidied if necessary)."""
    # Get tidied html if it exists.
    tidyfields = {}
    for fieldname, fieldtext in kinote.fields.items():
        fid = get_field_note_id(kinote.n.id, fieldname)
        if fid in paths:
            tidyfields[fieldname] = paths[fid].read_text()
        tidyfields[fieldname] = fieldtext

    # Construct note repr from tidyfields map.
    lines = kinote.get_header_lines()
    for fieldname, fieldtext in tidyfields.items():
        lines.append("### " + fieldname)
        lines.append(fieldtext)
        lines.append("")

    # Dump payload to filesystem.
    return "\n".join(lines)
