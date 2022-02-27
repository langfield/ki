#!/usr/bin/env python3
"""A module containing a class for Anki notes."""
from beartype import beartype
from apy.anki import Note
from apy.convert import html_to_screen, is_generated_html


class KiNote(Note):
    """
    A subclass of ``apy.Note`` for parsing syntax-highlighted code in note fields.

    This is distinct from the anki ``Note`` class, which is accessible using
    ``self.n``.
    """

    def __init__(self, anki, note):
        super().__init__(anki, note)
        self.deck = self.a.col.decks.name(self.n.cards()[0].did)

    @beartype
    def __repr__(self) -> str:
        """Convert note to Markdown format"""
        lines = [
            "## Note",
            f"nid: {self.n.id}",
            f"model: {self.model_name}",
        ]

        lines += [f"deck: {self.get_deck()}"]
        lines += [f"tags: {self.get_tag_string()}"]

        if not any(is_generated_html(x) for x in self.n.values()):
            lines += ["markdown: false"]

        lines += [""]

        for key, val in self.n.items():
            lines.append("### " + key)
            lines.append(html_to_screen(val, parseable=True))
            lines.append("")

        return "\n".join(lines)

    def get_deck(self):
        """Return which deck the note belongs to"""
        return self.deck

    def set_deck(self, deck):
        """Move note to deck"""
        if not isinstance(deck, str):
            raise Exception('Argument "deck" should be string!')

        newdid = self.a.col.decks.id(deck)
        cids = [c.id for c in self.n.cards()]

        if cids:
            self.a.col.decks.setDeck(cids, newdid)
            self.a.modified = True
        self.deck = self.a.col.decks.name(self.n.cards()[0].did)
