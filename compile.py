"""Compile a deck."""
#!/usr/bin/env python3
import re
import os
import json
import shutil
import zipfile
import argparse
import unicodedata

from beartype import beartype

# pylint: disable=unused-import
import anki.collection

# pylint: enable=unused-import

from anki import hooks
from anki.exporting import AnkiExporter
from anki.collection import Collection


class AnkiPackageExporter(AnkiExporter):
    """Modified ``.apkg`` writer."""

    ext = ".apkg"

    @beartype
    def __init__(self, col: Collection) -> None:
        AnkiExporter.__init__(self, col)

    @beartype
    def exportInto(self, path: str) -> None:
        """Export a deck."""
        with zipfile.ZipFile(
            path, "w", zipfile.ZIP_DEFLATED, allowZip64=True, strict_timestamps=False
        ) as z:
            media = self.doExport(z, path)
            z.writestr("media", json.dumps(media))

    # pylint: disable=arguments-differ
    @beartype
    def doExport(self, z: zipfile.ZipFile, path: str) -> dict[str, str]:
        """Actually do the exporting."""
        # Export into an anki2 file.
        colfile = path.replace(".apkg", ".anki2")
        AnkiExporter.exportInto(self, colfile)
        z.write(colfile, "collection.anki2")

        media = export_media(z, self.mediaFiles, self.mediaDir)

        # Tidy up intermediate files.
        os.unlink(colfile)
        p = path.replace(".apkg", ".media.db2")
        if os.path.exists(p):
            os.unlink(p)
        shutil.rmtree(path.replace(".apkg", ".media"))
        return media


@beartype
def export_media(z: zipfile.ZipFile, files: list[str], fdir: str) -> dict[str, str]:
    media = {}
    for i, file in enumerate(files):
        file = hooks.media_file_filter(file)
        mpath = os.path.join(fdir, file)
        if os.path.isdir(mpath):
            continue
        if os.path.exists(mpath):
            if re.search(r"\.svg$", file, re.IGNORECASE):
                z.write(mpath, str(i), zipfile.ZIP_DEFLATED)
            else:
                z.write(mpath, str(i), zipfile.ZIP_STORED)
            media[str(i)] = unicodedata.normalize("NFC", file)
            hooks.media_files_did_export(i)

    return media


def main() -> None:
    """Compile a top-level deck into a ``.apkg`` binary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection")
    parser.add_argument("--deck")
    args: argparse.Namespace = parser.parse_args()
    col = Collection(args.collection)

    did = col.decks.id(args.deck)

    exporter = AnkiPackageExporter(col)

    # pylint: disable=invalid-name
    exporter.includeSched = False
    exporter.includeMedia = True
    exporter.includeTags = True
    exporter.includeHTML = True
    # pylint: enable=invalid-name
    exporter.cids = None
    exporter.did = did

    deck_name = re.sub('[\\\\/?<>:*|"^]', "_", args.deck)
    filename = f"{deck_name}{exporter.ext}"
    file = os.path.normpath(filename)
    exporter.exportInto(file)


if __name__ == "__main__":
    main()
