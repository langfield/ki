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

    @beartype
    def __repr__(self) -> str:
        """Convert note to Markdown format"""
        lines = [
            '## Note',
            f'nid: {self.n.id}',
            f'model: {self.model_name}',
        ]

        if self.a.n_decks > 1:
            lines += [f'deck: {self.get_deck()}']

        lines += [f'tags: {self.get_tag_string()}']

        if not any(is_generated_html(x) for x in self.n.values()):
            lines += ['markdown: false']

        lines += ['']

        for key, val in self.n.items():
            lines.append('### ' + key)
            lines.append(html_to_screen(val, parseable=True))
            lines.append('')

        return '\n'.join(lines)
