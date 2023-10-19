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

BACKTICKS = "```\n"


@beartype
@dataclass(frozen=True)
class Field:
    """Field content pair."""

    title: str
    content: str


@beartype
@dataclass(frozen=True)
class Header:
    """Note metadata."""

    title: str
    guid: str
    model: str


@beartype
@dataclass(frozen=True)
class FlatNote:
    """Flat (as possible) representation of a note."""

    title: str
    guid: str
    model: str
    tags: List[str]
    fields: Dict[str, str]


class NoteTransformer(Transformer):
    r"""
    note
      header
        title     Note
        guid: 123412341234

        notetype: Basic

        tags      None


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
    # pylint: disable=missing-function-docstring

    @beartype
    def note(self, n: List[Union[Header, List[str], Field]]) -> FlatNote:
        assert len(n) >= 3

        header = n[0]
        tags = n[1]
        fields = n[2:]
        assert isinstance(header, Header)
        assert isinstance(fields[0], Field)

        # We drop the first character because it is a newline.
        fieldmap: Dict[str, str] = {}
        for field in fields:
            fieldmap[field.title] = field.content[1:]

        return FlatNote(
            title=header.title,
            guid=header.guid,
            model=header.model,
            tags=tags,
            fields=fieldmap,
        )

    @beartype
    def header(self, h: List[str]) -> Header:
        h = filter(lambda s: s != BACKTICKS, h)
        return Header(*h)

    @beartype
    def title(self, t: List[str]) -> str:
        """``title: "##" TITLENAME "\n"+``"""
        assert len(t) == 1
        return t[0]

    @beartype
    def tags(self, tags: List[Optional[str]]) -> List[str]:
        tags = filter(lambda t: t != BACKTICKS, tags)
        return [tag for tag in tags if tag is not None]

    @beartype
    def field(self, f: List[str]) -> Field:
        assert len(f) >= 1
        fheader = f[0]
        lines = f[1:]
        content = "".join(lines)
        if content[-2:] != "\n\n":
            raise RuntimeError(
                f"Nonterminating fields must have >= 1 trailing empty line:\n{content}"
            )
        return Field(fheader, content[:-1])

    @beartype
    def lastfield(self, f: List[str]) -> Field:
        assert len(f) >= 1
        fheader = f[0]
        lines = f[1:]
        content = "".join(lines)
        if len(content) > 0 and content[-1] == "\n":
            content = content[:-1]
        return Field(fheader, content)

    @beartype
    def fieldheader(self, f: List[str]) -> str:
        """``fieldheader: "##" " "* ANKINAME "\n"+``"""
        assert len(f) == 1
        return f[0]

    @beartype
    def GUID(self, t: Token) -> str:
        """Possibly empty for new markdown notes."""
        return re.sub(r"^guid:", "", str(t)).strip()

    @beartype
    def NOTETYPE(self, t: Token) -> str:
        model = re.sub(r"^notetype:", "", str(t)).strip()
        return model

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

    @beartype
    def TRIPLEBACKTICKS(self, t: Token) -> str:
        return str(t)
