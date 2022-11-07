"""Tests for SQLite Lark grammar."""
from itertools import starmap
from functools import partial

import prettyprinter as pp
from loguru import logger

from beartype import beartype
from beartype.typing import List

import hypothesis.strategies as st
from hypothesis import settings, given

# pylint: disable=unused-import
import anki.collection

# pylint: enable=unused-import
from anki.errors import CardTypeError
from anki.models import ModelManager, NotetypeDict, TemplateDict
from anki.collection import Collection

import ki.functional as F
from tests.test_ki import get_test_collection
from tests.test_sql_stateful import fnames, fmts

# pylint: disable=too-many-lines, missing-function-docstring, too-many-locals


EMPTY = get_test_collection("empty")
Collection(EMPTY.col_file).close(downgrade=True)
logger.debug(F.md5(EMPTY.col_file))
COL = Collection(EMPTY.col_file)


@given(st.data())
@settings(max_examples=100)
@beartype
def test_add_notetype(data: st.DataObject) -> None:
    """Add a new notetype."""
    mm: ModelManager = COL.models
    nchars = st.characters(blacklist_characters=['"'], blacklist_categories=["Cs"])
    name: str = data.draw(st.text(min_size=1, alphabet=nchars), "add nt: name")
    nt: NotetypeDict = mm.new(name)
    field_name_lists = st.lists(fnames(), min_size=1, unique_by=lambda s: s.lower())
    fieldnames = data.draw(field_name_lists, "add nt: fieldnames")
    nt["flds"] = list(map(mm.new_field, fieldnames))

    tmplnames = st.lists(st.text(alphabet=nchars, min_size=1), min_size=1)
    tnames: List[str] = data.draw(tmplnames, "add nt: tnames")
    n = len(tnames)
    textlists = st.lists(st.text(), min_size=n, max_size=n, unique=True)
    qtxts = data.draw(textlists, "add nt: qtxts")
    atxts = data.draw(textlists, "add nt: atxts")
    qfmts = list(map(partial(fmts, data, fieldnames), qtxts))
    afmts = list(map(partial(fmts, data, fieldnames), atxts))
    tmpls: List[TemplateDict] = list(map(mm.new_template, tnames))
    triples = zip(tmpls, qfmts, afmts)
    tmpls = list(starmap(lambda t, q, a: t | {"qfmt": q, "afmt": a}, triples))
    nt["tmpls"] = tmpls
    try:
        mm.add_dict(nt)
    except CardTypeError as err:
        logger.debug(err)
        t = tmpls[0]
        logger.debug(pp.pformat(t))
        raise err
