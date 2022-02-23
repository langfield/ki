
`ki` provides command-line functions to:

1. **clone** a `.anki2` collection into a directory as a git repository.
2. **pull** changes from the Anki desktop client (and AnkiWeb) into an existing
   repository.
3. **push** changes (safely!) back to Anki.

### Source code
This is documentation for the `ki`
[repository](https://github.com/langfield/ki). If you have `git`, you can clone
a local copy of the source code by running the following command in a terminal:
```bash
git clone git@github.com:langfield/ki.git
```

# Installation
`ki` is tested on Python 3.9 and Anki 2.1.49.

1. Install the `ki` package from PyPI:
```bash
pip install anki-ki
```

# Usage

## Cloning an Anki collection into a new `ki` repository

The `ki clone` command takes one required argument (the path to a `.anki2`
file) and one optional argument (a path to a target directory). The usage is
meant to mirror that of `git clone`.

An example of the `clone` subcommand usage and its output is given below.

```bash
$ ki clone ~/.local/share/Anki2/lyra/collection.anki2 decks
```
```bash
Found .anki2 file at '/home/lyra/.local/share/Anki2/lyra/collection.anki2'
Computed md5sum: ad7ea6d486a327042cf0b09b54626b66
Wrote md5sum to '/home/lyra/decks/.ki/hashes'
Cloning into '/home/lyra/decks/'...
100%|█████████████████████████| 28886/28886 [00:10<00:00, 2883.78it/s]
```

## Pulling changes from an Anki collection into an existing `ki` repository

Once an Anki collection has been cloned, we can `pull` changes made by the Anki
desktop client into our repository.

An example of the `pull` subcommand usage and its output is given below.

```bash
$ ki pull
```
```bash
Pulling from '/home/lyra/.local/share/Anki2/lyra/collection.anki2'
Computed md5sum: 199216c39eeabe23a1da016a99ffd3e2
Updating 5a9ef09..9c30b73
Fast-forward
 note1645010162168.md |  4 ++--
 note1645222430007.md | 11 +++++++++++
 2 files changed, 13 insertions(+), 2 deletions(-)
 create mode 100644 note1645222430007.md

From /tmp/tmpt5a3yd9a/ki/local/199216c39eeabe23a1da016a99ffd3e2/
 * branch            main       -> FETCH_HEAD
 * [new branch]      main       -> anki/main

Wrote md5sum to '/home/lyra/decks/.ki/hashes'
```

`ki` first deletes any residual ephemeral repositories in `/tmp/ki/remote/`.
These would only remain here if a previous pull command failed.

It then verifies that the path to the `.anki2` file specified in the `.ki/`
directory (analogous to the `.git/` directory) still exists.

It computes and records the hash of the collection file. In this way, `ki`
keeps track of whether the collection database has changed since the last
`clone`/`pull`.

Finally, the collection is then cloned into an ephemeral repository in a temp
directory, which is then `git pull`-ed into the current repository.

At this point, if the git operation fails, the user can take over and manage
the merge themselves.

## Pushing changes in a `ki` repository to an Anki collection

When we want to push our changes back to the Anki desktop client, we can use
`ki push` to do that.

An example of the `push` subcommand usage and its output is given below.

```bash
$ ki push
```
```bash
Pushing to '/home/lyra/.local/share/Anki2/lyra/collection.anki2'
Computed md5sum: 199216c39eeabe23a1da016a99ffd3e2
Verified md5sum matches latest hash in '/home/lyra/decks/.ki/hashes'
Generating local .anki2 file from latest commit: 2aa009729b6dd337dd1ce795df611f5a49
Writing changes to '/tmp/tmpyiids2qm/original.anki2'...
100%|█████████████████████████████████| 2/2 [00:00<00:00, 1081.56it/s]
Database was modified.
Writing backup of .anki2 file to '/home/lyra/decks/.ki/backups'
Overwrote '/home/lyra/.local/share/Anki2/lyra/collection.anki2'
```

We store 5 backups of the collection prior to a push.

> **INTERNAL**. It is not necessary to have a persistent "remote" copy of the
repo to pull from. The remote can be ephemeral. It only exists when we `ki pull`,
and then `ki` deletes it. This is safe because we're checking the
`md5sum` of `collection.anki2`. Notably, it is not created when we `ki clone`
or `ki push`.

# Editing notes

An example of a generated markdown note is given below:
```markdown
# Note
nid: 1636122987400
model: Basic
deck: Decks::Mathematics::Differentiable Manifolds
tags:
markdown: false

## Front
What sort of object is `\(C_0(X)\)`?

## Back
A Banach algebra, and more specifically a `\(C^*\)`-algebra
```

# Getting started

This section will walk through the following example workflow:

1. **Cloning** an existing collection into a `ki` repository.
2. **Downloading** a collaborative deck from [GitHub](https://github.com/).
3. **Pulling** other users' changes to the deck from [GitHub](https://github.com/).
4. **Editing** the collaborative deck.
5. **Pushing** those edits back to [GitHub](https://github.com/).

## Cloning a collection

Before cloning, you'll need to find your `.anki2` collection file.
This is where Anki stores the data for all your notes.

> **Note.** If you're new to Anki, or are unfamiliar with the terms *collection*,
*profile*, *note*, or *card*, you may wish to take a look at the Anki
[documentation](https://docs.ankiweb.net/intro.html).

If you already know the path to the `.anki2` collection file you want to clone,
skip to the [section on running the clone command][Running the clone command].


### Finding the `.anki2` collection file

To find your collection file, you must first find your Anki data directory. The
location of this varies by operating system.

In most cases, you should be able to find your data directory at the path given
below for your respective OS:

#### MacOS

```
~/Library/Application Support/Anki2
```

#### Windows

```
%APPDATA%\Anki2 
```

#### GNU/Linux
```
~/.local/share/Anki2
```


> **Note.** You can read more about the default Anki data directory locations
[here](https://docs.ankiweb.net/files.html#file-locations).

---


If you are running Anki 2.1 (which you should be, because `ki` is not tested
with lower versions), opening this directory will reveal several files and
subdirectories. The following example output is from a machine running Debian
GNU/Linux:

```
user@host:~/.local/share/Anki2$ ls
 addons21   crash.log   prefs21.db   README.txt  'User 1'
```
In particular, there is a subdirectory for each **profile**. In the above
example, there is only one profile, `User 1`. But, in general, there may be
many profiles associated with a given Anki installation.

Below you can see a visual representation of the directory structure of an
Anki data directory with two profiles, `User 1`, and `User 2`:

```bash
Anki2/
├── addons21
│   ├── 1046608507
│   ├── 109531687
│   ├── 1097423555
│   └── 1972239816
├── crash.log
├── prefs21.db
├── README.txt
├── User 1
│   ├── backups
│   ├── collection2.log
│   ├── collection.anki2
│   ├── collection.log
│   ├── collection.media
│   ├── collection.media.db2
│   └── deleted.txt
└── User 2
    ├── collection.anki2
    ├── collection.anki2-wal
    └── collection.media
```

As you can see, there is a `collection.anki2` file in each profile
subdirectory.

If you're not sure of the name of your user profile, it can be seen in the
title bar of the Anki desktop client:

<p align="center">
  <img width="460" src="anki.png">
</p>

And this is what we see when we enter the profile directory for `User 2` and
list its contents:
```
user@host:~/.local/share/Anki2$ cd User\ 2/
user@host:~/.local/share/Anki2/User 2$ ls
collection.anki2  collection.anki2-wal  collection.media
```

So the path that we want is:
```
~/.local/share/Anki2/User\ 2/collection.anki2
```

### Running the `ki clone` command
[Running the clone command]: #Running-the-ki-clone-command


# How it works

`ki` is built on top of existing tooling implemented in the python package
[`apy`](https://github.com/lervag/apy), which is used to parse the Anki
collection SQLite file and convert its contents to human-readable markdown
files.

These files (one per Anki note) are then dumped to a configurable location in
the filesystem as a git repository, whose structure mirrors that of the decks
in the collection. In effect, `ki` treats the git repo it generates as a local
copy of the collection, and the `.anki2` collection file as a remote.

All operations like pulling updates to the collection into `ki` and pushing
updates from `ki` into Anki are handled by git under the hood.

This appproach has several advantages:

1. Merge conflicts can be handled in the usual, familiar way.
2. Additional remotes (e.g. a human-readable backup of a collection on github)
   can be added easily.
3. Users are free to pick the editor of their choice, perform batch editing
   with command line tools like `awk` or `sed`, and even add CI actions.




# Model

The following diagram shows the dataflow of a typical Anki/`ki` stack.

```
                 +-------------+          +--------------+
                 |             |          |              |
                 |   AnkiWeb  -------------  AnkiMobile  |
                 |             |   sync   |              |
                 +------|------+          +--------------+
                        |
                        | sync
                        |
                 +------|------+
                 |             |
                 |    Anki     |
                 |             |
                 +------|------+
                        |
                        | deck edits
                        |
               +--------|--------+               +------------------+
               |                 |    ki clone   |                  |
               |                 ---------------->                  |
               | Collection file |               |     ~/decks/     |
               |    (.anki2)     |    ki push    | (git repository) |
               |                 <----------------                  |
               |                 |               |                  |
               +--------|--------+               +---------^--------+
                        |                                  |
                        | ki pull                          |
                        |                                  |
                        |                                  |
             +----------v----------+                       |
             |                     |                       |
             | /tmp/ki/remote/AAA  |           ki pull     |
             |  (git repository)   -------------------------
             |    [ephemeral]      |
             |                     |
             +---------------------+
```
The node labeled Anki is the Anki desktop client on the localhost. It
communicates with the AnkiWeb servers via Anki's sync feature. Other clients
(e.g. AnkiDroid and AnkiMobile) are able to (1) pull changes made by the
desktop client into their local collections via AnkiWeb, and (2) push changes
made locally back to AnkiWeb.

When the Anki desktop client is started on the localhost, it opens and places a
lock on the `.anki2` SQLite file. During the session, changes are possibly made
to the deck, and the SQLite file is unlocked when the program is closed.

Since `ki` must read from this database file, that means that `ki` commands
will not work while Anki is running. This is **by design**: the database is
locked for a reason, and enforcing this constraint lowers the likelihood that
users' decks become corrupted.

An ephemeral repository is used as an auxiliary step during the `ki pull`
operation so that we can merge the Anki desktop client's changes into our
repository via git.

# Generating html

By default, `ki` parses the html of each field and dumps the content only,
insofar as that is possible. It also supports parsing arbitrary html elements
autogenerated by addons and regenerated the updated content. In the following
subsection, we walk through an example.

## Example: generating syntax-highlighted code blocks

The anki addon developer Glutanimate has an addon called `syntax-highlighting`,
which adds UI elements to the Anki note editor that automatically generates a
syntax highlighted version of a code block from the clipboard. In effect, it
generates a formatted HTML table for the code listing that gets dumped into the
source of relevant note field.

A fork of this addon for the latest version of Anki (2.1.49 at the time of
writing), is available here:
https://ankiweb.net/shared/info/1972239816

And the source tree for the original addon is on github:
https://github.com/glutanimate/syntax-highlighting


For example, consider the following python code block:
```python
n = 1
n >> 1
print(n)
```

Given the above code, the addon generates the following HTML:
```html
<table class="highlighttable">
    <tbody>
        <tr>
            <td class="linenos">
                <div class="linenodiv">
                    <pre>
                        <span class="normal">1</span>
                        <span class="normal">2</span>
                        <span class="normal">3</span>
                    </pre>
                </div>
            </td>
            <td class="code">
                <div class="highlight">
                    <pre>
                        <code>
                            <span class="n">n</span>
                            <span class="o">=</span>
                            <span class="mi">1</span>
                            <br>
                                <span class="n">n</span>
                                <span class="o">&gt;&gt;</span>
                                <span class="mi">1</span>
                                <br>
                                    <span class="nb">print</span>
                                    <span class="p">(</span>
                                    <span class="n">n</span>
                                    <span class="p">)</span>
                                    <br>
                                    </code>
                                </pre>
                </div>
            </td>
        </tr>
    </tbody>
</table>
```
Editing fields like this could become annoying very quickly. It would be better
if `ki` just gave us the markdown version above (only 3 lines), and then
regenerated the note field HTML when converting the repository back into a
`.anki2` deck.

### Adding `ki` HTML attributes

And in fact, this is possible. We first fork the addon so we can add some extra
data to our generated HTML. In particular, we'd like to add an attribute
`ki-src` whose value is the UTF-8 encoded source code. In general, this will be
the encoded version of the source of whatever you'd like to autoformat.

We also add a `ki-formatter` attribute, whose value is an identifier that
specifies a custom python module (we must implement this) that transforms the
(possibly edited) `ki-src` text back into a HTML element of the form seen
above.

So let's call our `ki-formatter` identifier `syntax-hl-python`. Then our addon
has to change the opening tag of the snippet above to look like:
```html
<table class="highlighttable"; ki-src="n = 1\nn >> 1\nprint(n)\n"; ki-formatter="syntax-hl-python">
```

All `ki` needs is the original text of the code block prior to html formatting,
and a function that can reapply the formatting to the modified text. Since the
html table was generated by an addon, we already have a python function for
this, and in general you can provide a `~/.config/ki/ki.json` file that maps
implementation IDs to paths of python modules. The module must have a top-level
function defined of the form `format(text: str) -> bs4.Tag`. If you have an
addon implementation, you can import it here and use it in your `format()`
implementation. you can add a `ki` attribute whose value is the base64 encoding
of the code block, and a `implementation` attribute whose value is the name of
a function. At import-time, `ki` will decode this and write the human-readable
source to the relevant markdown file instead.
