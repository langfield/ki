"""Architecture for a cleaner version of _clone()."""
def clone():
    paths: Dict[int, Path] = {}
    decks: Dict[str, KiNote] = {}
    for nid in collection:
        kinote = KiNote(nid)
        decks[kinote.deck] = decks.get(kinote.deck, []) + [kinote]
        for field in kinote:
            if has_html(field):
                paths[nid] = dump(field)
    tidy(paths)
    for deck in sorted(set(decks.keys())):
        deckpath = get_and_create_deckpath(deck)
        for kinote in decks[deck]:
            notepath = get_notepath(kinote, deckpath)
            payload = get_tidy_payload(kinote, paths)
            notepath.write_text(payload, encoding="UTF-8")


def dump(field: str) -> Path:
    pass
