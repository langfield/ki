
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

# Usage reference

## Clone

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

## Pull

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

## Push

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

# Getting started
[getting started]: #getting-started

This section will walk through the following example workflow:

1. [**Cloning**][cloning a collection] an existing collection into a `ki` repository.
2. [**Editing**][editing notes] the note files in the repository.
3. [**Pushing**][pushing committed changes] those edits back to Anki.
4. [**Pulling**][pulling changes from anki] changes made in Anki into the repository.

[cloning a collection]: #cloning-a-collection

Before cloning, we'll need to find our `.anki2` collection file.
This is where Anki stores the data for all our notes.

> **Note.** If you're new to Anki, or are unfamiliar with the terms *collection*,
*profile*, *note*, or *card*, you may wish to take a look at the Anki
[documentation](https://docs.ankiweb.net/intro.html).

If you already know the path to the `.anki2` collection file you want to clone,
skip to the section on [running the clone command][running the clone command].


### Finding the `.anki2` collection file

To find our collection file, we must first find our Anki data directory. The
location of this varies by operating system.

In most cases, you should be able to find your data directory at the path given
below for your respective OS:

#### MacOS

```bash
~/Library/Application Support/Anki2
```

#### Windows

```bash
%APPDATA%\Anki2
```

#### GNU/Linux
```bash
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

#### Multiple profiles

Below we can see a visual representation of the directory structure of an
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

Note that there is a `collection.anki2` file in each profile subdirectory.

If you're not sure of the name of your user profile, it can be seen in the
title bar of the Anki desktop client:

<p align="center">
  <img width="460" src="anki.png">
</p>

Most Anki installations will only have one profile, and if you haven't changed
the default profile name, it will probably be called `User 1`. Let's enter the
profile directory for `User 1` and list its contents:
```
user@host:~/.local/share/Anki2$ cd User\ 1/
user@host:~/.local/share/Anki2/User 1$ ls
backups  collection2.log  collection.anki2  collection.log  collection.media  collection.media.db2  deleted.txt
```

So if we want to clone `User 1`'s collection, the path that we want is:
```
~/.local/share/Anki2/User\ 1/collection.anki2
```
We'll pass this as a command-line argument to the `ki` executable in the next
section.

### Running the clone command
[running the clone command]: #running-the-ki-clone-command

Now we're ready to actually clone the collection into a repository. The `ki clone`
command works similarly to `git clone`, in that it will create a new directory
for the repository *within* the current working directory. So if we want to
clone our collection into a new subdirectory in `~/` (the home directory on
macOS and GNU/Linux), we would first make sure we're in the home directory.
Second, we need to check that **Anki is closed** before cloning. Nothing bad
will happen if we clone while Anki is open, but the command will fail because
the database is locked. Once we've done that, we can run the command:
```bash
ki clone ~/.local/share/Anki2/User 1/collection.anki2
```
And we should see output that looks similar to this:
```bash
lyra@oxford$ ki clone ~/.local/share/Anki2/User 1/collection.anki2
Found .anki2 file at '/home/lyra/.local/share/Anki2/User 1/collection.anki2'
Computed md5sum: ad7ea6d486a327042cf0b09b54626b66
Wrote md5sum to '/home/lyra/collection/.ki/hashes'
Cloning into '/home/lyra/collection/'...
100%|█████████████████████████| 28886/28886 [00:10<00:00, 2883.78it/s]
```
If we list the contents of the home directory, we can see that `ki` did
indeed create a new directory called `collection`:
```bash
lyra@oxford:~$ ls
collection  pkgs
```

## Editing notes
[editing notes]: #editing-notes

Now that we've successfully cloned our Anki collection into a `ki` repository,
we can start editing notes! Our home directory looks like this:
```bash
lyra@oxford:~$ ls
collection  pkgs
```
And we see the repo we cloned, which is called `collection`.

Let's change directories to the newly cloned `ki` repo and take a look at
what's inside:
```bash
lyra@oxford:~$ cd collection/
lyra@oxford:~/collection$ ls --classify
algebras/ manifolds/ rings/
```
We see that we have three directories, which represent three Anki decks. This
is just an example; you'll see directories corresponding to the top-level decks
in your Anki collection.

> **Note.** The `ls --classify` command adds a trailing `/` to the end of
> directories to distinguish them from ordinary files.

Lets enter the `manifolds` directory and see what's inside.

```bash
lyra@oxford:~/collection$ cd manifolds/
lyra@oxford:~/collection/manifolds$ ls
MANIFOLDS.md
```

So we see a single markdown file called `MANIFOLDS.md`, which contains the
notes for the manifolds deck. If we had subdecks of the manifolds deck, we
would see more subdirectories here, and each one would have a markdown file in
it as well. Lets open this file and see what's inside.

We'll use vim to open the markdown file in this example, but any text editor
will work.

```bash
lyra@oxford:~/collection/manifolds$ vi MANIFOLDS.md
```
```markdown
# Note
nid: 1622849751948
model: Basic
deck: manifolds
tags:
markdown: false

## Front
Diffeomorphism

## Back
A smooth surjective map between manifolds which has a smooth inverse.

# Note
nid: 1566621764508
model: Basic
deck: manifolds
tags:
markdown: false

## Front
distribution (on a smooth manifold)

## Back
A distribution on \(M\) of rank \(k\) is a rank-\(k\) subbundle of \(TM\)
```

So we see the structure of two notes inside this file. For each note, there is
a section for note metadata, and a section for each field.

There is a typo in the first note. It says `smooth surjective map`, but it
should say `smooth bijective map`. Lets fix it, save our changes, and go back
to the terminal. When we go back up to the root of the repository and run `git
status`, we can see which files we've changed.

> **INTERNAL.** Add the output of git status here.

And running `git diff` shows us the content of the unstaged changes:

> **INTERNAL.** Add the output of git diff here.

Then we can commit our changes as usual.

```bash
lyra@oxford:~/collection$ git add manifolds/MANIFOLDS.md
lyra@oxford:~/collection$ git commit -m "Fix typo in diffeomorphism definition: 'surjective' -> 'bijective'"
```

At this point we would usually `git push`, but if we try that in a `ki`
repository, we'll see this:
```bash
lyra@oxford:~/collection$ git push
fatal: No configured push destination.
Either specify the URL from the command-line or configure a remote repository using

    git remote add <name> <url>

and then push using the remote name

    git push <name>

```
Since we're not pushing to an ordinary `git` remote, but to the Anki SQLite3
database, we must use `ki push` instead, which is covered briefly in the next
section.

## Pushing committed changes back to Anki
[pushing committed changes]: #pushing-committed-changes

This part is super easy! Similar to when we cloned, we must remember to **close
Anki** before pushing, or the command will fail (gracefully). All right, now we
just run the command:
```bash
lyra@oxford:~/collection$ ki push
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

As the output suggests, `ki` saves a backup of our collection each time we
`push`, just in case we wish to hard-revert a change you've made.

Now we can open Anki and view the changes we've made in the note browser!

## Pulling changes from Anki into the repository
[pulling changes from anki]: #pulling-changes-from-anki

So now we know how to make changes from the filesystem and push them back to
Anki, but suppose that after we cloned our repository, we made some edits
*within* Anki, and we'd like those to show up in our repository? For this,
we'll need to **close Anki**, and then run the following command:

```bash
lyra@oxford:~/collection$ ki pull
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

And we're done! Our repository is up to date, as `ki` will tell us if we try to pull again:
```bash
lyra@oxford:~/collection$ ki pull
ki pull: up to date.
```

### Merge conflicts

Occasionally, when we edit the same lines in the same note fields in both Anki
and our local repository, we may encounter a merge conflict:

```bash
lyra@oxford:~/collection$ ki pull
Pulling from '/home/lyra/.local/share/Anki2/User 1/collection.anki2'
Computed md5sum: debeb6689f0b83d520ff913067c598e9
Auto-merging note1645788806304.md
CONFLICT (add/add): Merge conflict in note1645788806304.md
Automatic merge failed; fix conflicts and then commit the result.

From /tmp/tmpgkq4ilfy/ki/local/debeb6689f0b83d520ff913067c598e9/
 * branch            main       -> FETCH_HEAD
 * [new branch]      main       -> anki/main

Wrote md5sum to '/home/mal/collection/.ki/hashes'
```

This is expected behavior, and since the process of resolving merge conflicts
is the same for `ki` repositories as `git` repositories (since `ki`
repositories *are* git repositories), we refer to
[StackOverflow](https://stackoverflow.com/questions/161813/how-to-resolve-merge-conflicts-in-a-git-repository)
for how to proceed.



# Collaborative decks

This section assumes knowledge of the basic `ki` operations and familiarity
with `git`. If you haven't yet cloned your Anki collection into a `ki`
repository, read the [getting started][getting started] section.

1. [**Cloning**][cloning a collaborative deck from github] a collaborative deck from [GitHub](https://github.com/).
2. [**Editing**][editing a collaborative deck] the collaborative deck.
3. [**Pulling**][pulling other users' changes from github] other users' changes to the deck from [GitHub](https://github.com/).
4. [**Pushing**][pushing edits back to github] edits back to [GitHub](https://github.com/).


## Cloning a collaborative deck from GitHub
[cloning a collaborative deck from github]: #cloning-a-collaborative-deck-from-github

Now that we've created our first `ki` repository, we might want to try our hand
at collaborating on a deck with other Anki users. We won't actually need to
make use of the `ki` program to do this, because **`ki` repositories are also
git repositories**, and so we can clone collaborative decks from GitHub as
`git-submodules` of our collection repo.

> **Note.** If you're completely unfamiliar with `git`, consider reading this
> short
> [introduction](https://blog.teamtreehouse.com/git-for-designers-part-1).

Suppose we've cloned an Anki collection into a `ki` repository in our home
directory, just like we did in the [getting started][getting started] section,
and we want to add a collaborative deck from GitHub to our collection. Let's
walk through an example. Our home directory looks like this:
```bash
lyra@oxford:~$ ls
collection  pkgs
```
And we see the repo we cloned, which is called `collection`.

To add a collaborative deck repo as a submodule, we'll first need to change
directories to the newly cloned `ki` repo:
```bash
lyra@oxford:~$ cd collection/
lyra@oxford:~/collection$ ls --classify
algebras/ groups/ rings/
```
We see that we have three directories, which represent three Anki decks. This
is just an example; you'll see directories corresponding to the top-level decks
in your Anki collection.

> **Note.** The `ls --classify` command adds a trailing `/` to the end of
> directories to distinguish them from ordinary files.

### Adding the repository as a git submodule

Suppose we want to add the collaborative deck
[https://github.com/langfield/manifolds.git](https://github.com/langfield/manifolds.git)
to our collection. We can do that by running the command:

```bash
git-submodule add https://github.com/langfield/manifolds.git
```
which yields the output:
```bash
lyra@oxford~/collection$ git-submodule add https://github.com/langfield/manifolds.git
Cloning into 'manifolds'...
remote: Counting objects: 11, done.
remote: Compressing objects: 100% (10/10), done.
remote: Total 11 (delta 0), reused 11 (delta 0)
Unpacking objects: 100% (11/11), done.
Checking connectivity... done.
```

And we can see that the command was successful because we have a new
directory/deck called `manifolds` in our repo:
```bash
lyra@oxford:~/collection$ ls --classify
algebras/ groups/ manifolds/ rings/
```

Nice!

## Editing a collaborative deck
[editing a collaborative deck]: #editing-a-collaborative-deck

There are two ways to edit a collaborative deck locally:

1. Edit the markdown files in the `ki` repository.
2. Edit the deck inside the Anki desktop client.


---

After we've cloned the `manifolds` deck repository into a submodule of our `ki`
repository, we may want to make some edits to the deck.




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
the encoded version of the source of whatever we'd like to autoformat.

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
this, and in general we can provide a `~/.config/ki/ki.json` file that maps
implementation IDs to paths of python modules. The module must have a top-level
function defined of the form `format(text: str) -> bs4.Tag`.

If we have an addon implementation, we can import it here and use it in our
`format()` implementation. We can add a `ki` attribute whose value is the
base64 encoding of the code block, and a `implementation` attribute whose value
is the name of a function. At import-time, `ki` will decode this and write the
human-readable source to the relevant markdown file instead.
