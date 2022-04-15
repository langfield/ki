@beartype
@dataclass(frozen=True)
class ColNote:
    """A note that exists in the Anki DB."""
    n: Note
    title: str
    nid: int
    model: str
    deck: str
    tags: List[str]
    markdown: bool
    fields: Dict[str, str]


@safe
@beartype
def get_note(col: Collection, nid: int) -> Result[Note, Exception]:
    try:
        return Ok(col.get_note(nid))
    except anki.errors.NotFoundError:
        return add_note_from_flatnote(col, flatnote)


def get_fullnote(col: Collection, flatnote: FlatNote) -> ColNote:
    note: Res[Note] = get_note(col, flatnote.nid)
