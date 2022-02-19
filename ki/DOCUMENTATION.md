
`ki` provides command-line functions to:

1. **clone** a `.anki2` collection into a directory as a git repository,
2. **pull** changes from the Anki desktop client (and AnkiWeb) into an existing
   repository,
3. **push** changes (safely!) back to Anki.

> **INTERNAL.** Perhaps we should support making each deck a separate git
> submodule, so that users can have collaborative decks and work on them within
> their own collections.

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
Cloning into '/home/lyra/decks/' ...
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
Found .anki2 file at '/home/lyra/.local/share/Anki2/lyra/collection.anki2'
Computed md5sum: ad7ea6d486a327042cf0b09b54626b66
Wrote md5sum to '/home/lyra/.ki/hashes'
Cloning .anki2 database into ephemeral repository at '/tmp/ki/remote/'
Running 'git remote add origin /tmp/ki/remote/ad7ea6d4.git'
Running 'git pull'
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
Found .anki2 file at /home/lyra/.local/share/Anki2/lyra/collection.anki2
Computed md5sum: ad7ea6d486a327042cf0b09b54626b66
Verified md5sum matches latest hash in '/home/lyra/.ki/hashes'
Checked out latest commit at '/tmp/ki/local/'
Generating local .anki2 file from latest commit
Backing up original .anki2 file
Overwriting collection
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
