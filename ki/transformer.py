#!/usr/bin/env python3
"""A Lark transformer for the ki note grammar."""
import re
from dataclasses import dataclass

from lark import Transformer
from lark.lexer import Token

from beartype import beartype
from beartype.typing import (
    List,
    Dict,
    Optional,
    Union,
)

# pylint: disable=invalid-name, too-few-public-methods


@beartype
@dataclass(frozen=True)
class Field:
    """Field content pair."""

    title: str
    content: str


@beartype
@dataclass(frozen=True)
class FlatNote:
    """Flat (as possible) representation of a note."""

    title: str
    nid: int
    model: str
    deck: str
    tags: List[str]
    markdown: bool
    fields: Dict[str, str]


@beartype
@dataclass(frozen=True)
class Header:
    """Note metadata."""

    title: str
    nid: int
    model: str
    deck: str
    tags: List[str]
    markdown: bool


class NoteTransformer(Transformer):
    r"""
    file
      note
        header
          title     Note
          nid: 123412341234

          model: Basic

          deck      a
          tags      None
          markdown: false


        field
          fieldheader
            ###
            Front
          r


        field
          fieldheader
            ###
            Back
          s
    """
    # pylint: disable=no-self-use, missing-function-docstring

    @beartype
    def file(self, flatnotes: List[FlatNote]) -> List[FlatNote]:
        return flatnotes

    @beartype
    def note(self, n: List[Union[Header, Field]]) -> FlatNote:
        assert len(n) >= 2

        header = n[0]
        fields = n[1:]
        assert isinstance(header, Header)
        assert isinstance(fields[0], Field)

        fieldmap: Dict[str, str] = {}
        for field in fields:
            fieldmap[field.title] = field.content

        return FlatNote(
            header.title,
            header.nid,
            header.model,
            header.deck,
            header.tags,
            header.markdown,
            fieldmap,
        )

    @beartype
    def header(self, h: List[Union[str, int, bool, List[str]]]) -> Header:
        return Header(*h)

    @beartype
    def title(self, t: List[str]) -> str:
        """``title: "##" TITLENAME "\n"+``"""
        assert len(t) == 1
        return t[0]

    @beartype
    def tags(self, tags: List[Optional[str]]) -> List[str]:
        return [tag for tag in tags if tag is not None]

    @beartype
    def field(self, f: List[str]) -> Field:
        assert len(f) >= 1
        fheader = f[0]
        lines = f[1:]
        content = "".join(lines)
        return Field(fheader, content)

    @beartype
    def fieldheader(self, f: List[str]) -> str:
        """``fieldheader: FIELDSENTINEL " "* ANKINAME "\n"+``"""
        assert len(f) == 2
        return f[1]

    @beartype
    def deck(self, d: List[str]) -> str:
        """
        Lark syntax:
        ``deck: "deck:" DECKNAME "\n"``

        This is guaranteed to be nonempty.
        """
        assert len(d) == 1
        assert len(d[0]) > 0
        return d[0]

    @beartype
    def NID(self, t: Token) -> int:
        """Return ``-1`` if empty."""
        nid = re.sub(r"^nid:", "", str(t)).strip()
        try:
            return int(nid)
        except ValueError:
            return -1

    @beartype
    def MODEL(self, t: Token) -> str:
        model = re.sub(r"^model:", "", str(t)).strip()
        return model

    @beartype
    def MARKDOWN(self, t: Token) -> bool:
        md = re.sub(r"^markdown:", "", str(t)).strip()
        assert md in ("true", "false")
        return md == "true"

    @beartype
    def DECKNAME(self, t: Token) -> str:
        return str(t).strip()

    @beartype
    def FIELDLINE(self, t: Token) -> str:
        return str(t)

    @beartype
    def TITLENAME(self, t: Token) -> str:
        return str(t)

    @beartype
    def ANKINAME(self, t: Token) -> str:
        return str(t)

    @beartype
    def TAGNAME(self, t: Token) -> str:
        return str(t)
