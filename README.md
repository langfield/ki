<h1 align="center">
  <img src="./docs/u1F367-shavedice.svg" width="25%" height="25%">
</h1>

**Ki** is a
[command-line](https://developer.mozilla.org/en-US/docs/Learn/Tools_and_testing/Understanding_client-side_tools/Command_line)
tool for converting Anki collections and decks into [git
repositories](https://git-scm.com/). Check out the
[**documentation**](https://langfield.github.io/ki/)!

* [Getting started](https://langfield.github.io/ki/)
* [Demo video](https://asciinema.org/a/500300)

<p align="center">
  <a href="https://asciinema.org/a/502129">
    <img src="https://cdn.jsdelivr.net/gh/langfield/ki@main/docs/push2.svg" width="90%" height="90%">
  </a>
</p>

### Installation

Install the following dependencies:

* [Python 3.9+](https://www.python.org/downloads/)
* [Tidy](https://www.html-tidy.org/)
* [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

Then open a terminal or command-line window, and install with the following
command:
```console
python3 -m pip install git+https://github.com/langfield/ki.git@main
```

### Questions

Want to try this but can't figure out how? **Please, please** [open an
issue](https://github.com/langfield/ki/issues/new) and explain what you need
help with. Any and all complaints, questions, and rants are welcome!


### Features

Ki provides three high-level features:

* **Cloning** your collection of Anki cards into a folder (called a **repository**)
  of human-readable markdown files.
* **Pulling** changes you make in Anki into your repository.
* **Pushing** changes you make to the markdown files in your repository back to
  Anki.

It also supports:

* **Collaborative decks on GitHub** via git submodules, which means you can
  download shared decks, upload your own, contribute changes to others' decks,
  and review/merge others' contributions to your own decks.
* **Full version history of your decks**, with the ability to **roll-back** to
  any previous [commit](https://github.com/git-guides/git-commit) of your
  collection (these are made any time you run a ki command).
* **Automatic backups** of your collection before every write to the
  [Anki database file](https://github.com/ankidroid/Anki-Android/wiki/Database-Structure)
  (you can find these in your repository in the `.ki/backups` directory).
* **Automatic APKG releases** on every tagged commit via GitHub actions. Using
  the included `compile.py` script, you can easily set up a workflow to build
  and publish your deck `.apkg` binaries automatically!
* Support for `Basic`, `Basic (and reversed)`, `Cloze`, `Cloze-Overlapper`, and
  any arbitrary notetypes you make.
* Full version history of edits to all your notetypes.
* Full version history of all your media files.
* Private fields for collaborative decks (so you can make your own notes and
  edits, and still get updates from deck maintainers without losing any data).
* Auto-indentation and tidying of HTML in note fields and notetype templates.
* Human-readable filenames (the filename is generated from a note's sort field
  content).
* Meaningful card file locations. If you move a card file to a different deck
  folder, it will move that card to that deck.
* Meaningful card file deletions. If you delete a card file in your repository,
  the card is deleted from your collection (but only if you *commit* the
  deletion, and because everything is version-controlled, you can always get it
  back).
* Manually adding new notes. You can write a markdown file in the [ki note format](#note-grammar)
  and push the new note to Anki, or you can copy an existing note to make a
  variant of it.
* **Granular merge conflict resolution.** When the usual sync process can't
  figure out how to automatically sync your decks, it forces you to overwrite
  *everything* with either (1) the server-side version of your collection, or
  (2) the local version of your collection. With ki, you can pick and choose
  which things you want to keep from the server, and which things you want to
  keep from your device.
* Collision-free reads and writes to the `collection.anki2` database file via
  SQLite3 lock acquisition (no errors from editing the collection at the same
  time as the Anki desktop client).
* Atomic database writes (we edit a copy of the collection in a temporary
  directory, and then replace the existing collection in a single
  `shutil.copyfile()` call).
* Changes are only pulled when the hash of the database has changed.
* Warnings for duplicate notes, unhealthy notes, etc.
* Cards for a single note split across multiple decks (these are represented
  with symlinks).
* Potential interoperation with mobile apps. The note grammar is programming
  language-agnostic, and so a parser could be written in e.g. Kotlin to import
  decks from GitHub right on your phone, but this would take a nontrivial
  amount of development effort.
* Unicode (`UTF-8`), foreign language characters, and symbols in all fields.
* Properly-escaped `LaTeX` and `MathJaX`.


### Note grammar

The following is an example of a markdown file automatically generated in the
ki note format.

````markdown
# Note
```
guid: dc6H$t-~MK
notetype: iKnow! Sentences
```

### Tags
```
languages
japanese
jp-sentences
jp-transportation
```

## Expression
駅からはタクシーに<b>乗って</b>ください。

## Meaning
Please take a taxi from the station.
乗る -- ride, take

## Reading
えき からは たくしー に <b>のって</b> ください

## Audio
[sound:e3c984736d8b1c2bdc467f2a1c98659a.mp3]

## Image_URI
<img src="e2d8a60b59f2be8ebcbffafa165c7a0d.jpg">

## iKnowID
sentence:247153

## iKnowType
sentence
````

### Explicitly unsupported

* Sibling cards in different decks


### Credits

Thanks to @husarcik, @pwintz for productive discussions about the architecture
of this project!
