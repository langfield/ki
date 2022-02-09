#!/usr/bin/env python3
"""A module containing a class for Anki notes."""
import apy


class KiNote(apy.Note):
    """
    A subclass of ``apy.Note`` for parsing syntax-highlighted code in note fields.

    This is distinct from the anki ``Note`` class, which is accessible using
    ``self.n``.
    """
    def __repr__(self):
        apy_note_repr: str = super().__repr__()
        apy_note_lines: List[str] = apy_note_repr.split("\n")
        raise NotImplementedError