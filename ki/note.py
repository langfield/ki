#!/usr/bin/env python3
"""A module containing a class for Anki notes."""
import re
import subprocess

import bs4
import anki
import click
import markdownify

from beartype import beartype
from beartype.typing import Dict, List

from apy.anki import Note, Anki
from apy.convert import markdown_to_html, html_to_markdown, _italize

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
        self.deck = self.a.col.decks.name(self.n.cards()[0].did)

        # Populate parsed fields.
        self.fields: Dict[str, str] = {}
        for key, val in self.n.items():
            self.fields[key] = html_to_screen(val, parseable=True)

    @beartype
    def __repr__(self) -> str:
        """Convert note to Markdown format"""
        lines = self.get_header_lines()

        for key, val in self.n.items():
            lines.append("### " + key)
            lines.append(html_to_screen(val, parseable=True))
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

        if not any(is_generated_html(x) for x in self.n.values()):
            lines += ["markdown: false"]

        lines += [""]
        return lines

    @beartype
    def get_deck(self) -> str:
        """Return which deck the note belongs to"""
        return self.deck

    @beartype
    def set_deck(self, deck: str) -> None:
        """Move note to deck."""
        newdid = self.a.col.decks.id(deck)
        cids = [c.id for c in self.n.cards()]

        if cids:
            self.a.col.decks.setDeck(cids, newdid)
            self.a.modified = True
        self.deck = self.a.col.decks.name(self.n.cards()[0].did)


@beartype
def is_generated_html(html: str) -> bool:
    """Check if text is a generated HTML"""
    if html is None:
        return False
    return GENERATED_HTML_SENTINEL in html


def html_to_screen(html, pprint=True, parseable=False):
    """Convert html for printing to screen"""
    if not pprint:
        soup = bs4.BeautifulSoup(
            html.replace("\n", ""), features="html5lib"
        ).next.next.next
        return "".join(
            [el.prettify() if isinstance(el, bs4.Tag) else el for el in soup.contents]
        )

    html = re.sub(r"\<style\>.*\<\/style\>", "", html, flags=re.S)

    generated = is_generated_html(html)
    if generated:
        plain = html_to_markdown(html)
        if html != markdown_to_html(plain):
            html_clean = re.sub(r' data-original-markdown="[^"]*"', "", html)
            if parseable:
                plain += (
                    "\n\n### Current HTML → Markdown\n"
                    f"{markdownify.markdownify(html_clean)}"
                )
                plain += f"\n### Current HTML\n{html_clean}"
            else:
                plain += "\n"
                plain += click.style(
                    "The current HTML value is inconsistent with Markdown!",
                    fg="red",
                    bold=True,
                )
                plain += "\n" + click.style(html_clean, fg="white")
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

    if not parseable:
        plain = re.sub(r"\*\*(.*?)\*\*", click.style(r"\1", bold=True), plain, re.S)

        plain = re.sub(r"\<b\>(.*?)\<\/b\>", click.style(r"\1", bold=True), plain, re.S)

        plain = re.sub(r"_(.*?)_", _italize(r"\1"), plain, re.S)

        plain = re.sub(r"\<i\>(.*?)\<\/i\>", _italize(r"\1"), plain, re.S)

        plain = re.sub(
            r"\<u\>(.*?)\<\/u\>", click.style(r"\1", underline=True), plain, re.S
        )

    return plain.strip()


def tidy(plain: str) -> str:
    """Run some html through tidy. SLOW."""
    p = subprocess.run(
        ["tidy", "-i", "-ashtml", "-utf8", "-q"],
        input=plain.strip(),
        text=True,
        check=False,
        capture_output=True,
    )
    plain = p.stdout
