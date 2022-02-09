#!/usr/bin/env python3
"""
Python package `ki` is a utility for editing ``.anki2`` deck collection files.
Rather than providing an interactive UI like the Anki desktop client or
``apy``, `ki` aims to allow natural editing *in the filesystem*. It uses
existing tooling implmented in ``apy`` to parse the Anki collection SQLite file
and convert its contents to human-readable markdown files. These files (one per
Anki note) are then dumped to a configurable location in the filesystem as a
git repository, whose structure mirrors that of the decks in the collection. In
effect, `ki` treats the git repo it generates as a local copy of the
collection, and the ``.anki2`` collection file as a remote. All git operations
like pulling updates to the collection into `ki` and pushing updates from `ki`
into Anki are handled by the user, without any wrappers. This means merge
conflicts can be handled in the usual, familiar way, and additional remotes
(e.g. a human-readable backup of a collection on github) can be added easily.
Users are free to pick the editor of their choice, perform batch editing with
command line tools like `awk` or `sed`, and even add CI actions. In general,
the purpose of `ki` is to allow users to work on large, complex Anki decks in
exactly the same way they work on large, complex software projects.

.. include:: ./DOCUMENTATION.md
"""
