"""Architecture for a cleaner version of _clone()."""
from typing import Dict, List
from pathlib import Path


def _clone():
    paths: Dict[int, Path] = {}
    decks: Dict[str, List[KiNote]] = {}
    for nid in collection:
        kinote = KiNote(nid)
        decks[kinote.deck] = decks.get(kinote.deck, []) + [kinote]
        for field in kinote:
            if has_html(field):
                paths[nid] = dump(field)
    tidy(paths)
    for deckname in sorted(set(decks.keys()), key=len, reverse=True):
        deckpath = get_and_create_deckpath(deckname)
        for kinote in decks[deckname]:
            notepath = get_notepath(kinote, deckpath)
            payload = get_tidy_payload(kinote, paths)
            notepath.write_text(payload, encoding="UTF-8")


def dump(field: str) -> Path:
    pass

def tidy(paths: Dict[int, Path]) -> None:
    pass

def get_and_create_deckpath(deckname: str) -> Path:
    # Strip leading periods so we don't get hidden folders.
    components = deckname.split("::")
    components = [re.sub(r"^\.", r"", comp) for comp in components]
    deckpath = Path(targetdir, *components)
    deckpath.mkdir(parents=True, exist_ok=True)
    return deckpath
