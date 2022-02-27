#!/usr/bin/env python3
"""A module containing a class for Anki notes."""
import re
import bs4
import click
import markdownify
from beartype import beartype
from apy.anki import Note
from apy.convert import markdown_to_html, html_to_markdown, _italize

GENERATED_HTML_SENTINEL = "data-original-markdown"


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
