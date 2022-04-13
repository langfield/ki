#!/usr/bin/env python3
"""A module containing a class for Anki notes."""
import re

import anki
import markdownify

from beartype import beartype
from beartype.typing import Dict, List

from apy.anki import Note, Anki
from apy.convert import markdown_to_html, html_to_markdown

GENERATED_HTML_SENTINEL = "data-original-markdown"


class KiNote(Note):
    """
    A subclass of ``apy.Note`` for applying transformations to HTML in note
    fields. This is distinct from the anki ``Note`` class, which is accessible
    using ``self.n``.

    Parameters
    ----------
    a : apy.anki.Anki
        Wrapper around Anki collection.
    note : anki.notes.Note
        Anki Note instance.
    """

    @beartype
    def __init__(self, a: Anki, note: anki.notes.Note):

        super().__init__(a, note)

        # TODO: Remove implicit assumption that all cards are in the same deck.
        self.deck = self.a.col.decks.name(self.n.cards()[0].did)

        # Populate parsed fields.
        self.fields: Dict[str, str] = {}
        for key, field in self.n.items():
            self.fields[key] = html_to_screen(field)

    @beartype
    def __repr__(self) -> str:
        """Convert note to Markdown format"""
        lines = self.get_header_lines()

        for key, field in self.n.items():
            lines.append("### " + key)
            lines.append(html_to_screen(field))
            lines.append("")

        return "\n".join(lines)

    @beartype
    def get_header_lines(self) -> List[str]:
        """Get header of markdown representation of note."""
        lines = [
            "## Note",
            f"nid: {self.n.id}",
            f"model: {self.model_name}",
        ]

        lines += [f"deck: {self.get_deck()}"]
        lines += [f"tags: {self.get_tag_string()}"]

        if not any(GENERATED_HTML_SENTINEL in field for field in self.n.values()):
            lines += ["markdown: false"]

        lines += [""]
        return lines

    @beartype
    def get_deck(self) -> str:
        """Return which deck the note belongs to."""
        # TODO: Remove implicit assumption that all cards are in the same deck.
        return self.deck

    @beartype
    def set_deck(self, deck: str) -> None:
        """Move note to deck."""
        # Get the id for deck with name ``deck``, and create it if it doesn't
        # already exist. The type signature for this API method says that it
        # may return ``None``, but this is only in the case where we pass
        # ``create=False``. So in reality we do not expect an exception to ever
        # be raised here.
        newdid: int = self.a.col.decks.id(deck, create=True)
        cids = [c.id for c in self.n.cards()]

        if cids:
            self.a.col.set_deck(cids, newdid)
            self.a.modified = True

        # TODO: Remove implicit assumption that all cards are in the same deck.
        self.deck = self.a.col.decks.name(self.n.cards()[0].did)


@beartype
def html_to_screen(html: str) -> str:
    """Convert html for printing to screen."""
    html = re.sub(r"\<style\>.*\<\/style\>", "", html, flags=re.S)

    generated = GENERATED_HTML_SENTINEL in html
    if generated:
        plain = html_to_markdown(html)
        if html != markdown_to_html(plain):
            html_clean = re.sub(r' data-original-markdown="[^"]*"', "", html)
            plain += (
                "\n\n### Current HTML → Markdown\n"
                f"{markdownify.markdownify(html_clean)}"
            )
            plain += f"\n### Current HTML\n{html_clean}"
    else:
        plain = html

    # For convenience: Un-escape some common LaTeX constructs
    plain = plain.replace(r"\\\\", r"\\")
    plain = plain.replace(r"\\{", r"\{")
    plain = plain.replace(r"\\}", r"\}")
    plain = plain.replace(r"\*}", r"*}")

    plain = plain.replace(r"&lt;", "<")
    plain = plain.replace(r"&gt;", ">")
    plain = plain.replace(r"&amp;", "&")
    plain = plain.replace(r"&nbsp;", " ")

    plain = plain.replace("<br>", "\n")
    plain = plain.replace("<br/>", "\n")
    plain = plain.replace("<br />", "\n")
    plain = plain.replace("<div>", "\n")
    plain = plain.replace("</div>", "")

    # For convenience: Fix mathjax escaping (but only if the html is generated)
    if generated:
        plain = plain.replace(r"\[", r"[")
        plain = plain.replace(r"\]", r"]")
        plain = plain.replace(r"\(", r"(")
        plain = plain.replace(r"\)", r")")

    plain = re.sub(r"\<b\>\s*\<\/b\>", "", plain)
    return plain.strip()
