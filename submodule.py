"""Convert a deck into a remote submodule."""
#!/usr/bin/env python3
import argparse
from pathlib import Path

import git
import git_filter_repo
from beartype import beartype
from beartype.typing import List

from ki import cp_repo, BRANCH_NAME
from ki.types import Dir, Rev
import ki.maybes as M
import ki.functional as F


REMOTE_NAME = "origin"


HELP_KIREPO = """
Path (relative or absolute) to a ki repo containing the subdeck which is to be
converted into a submodule. This is the directory that is created when you run
`ki clone`, and is usually called `collection`. It should contain a `.ki`
subdirectory.\n
"""

HELP_DECK = """
Relative path from the root of the ki repo to the subdeck which is to be
converted into a submodule.\n
"""

HELP_REMOTE = """
A git remote URL. This could be one of the following.

SSH remote of the form:     `git@github.com:user/submodule.git`
HTTPS remote of the form:   `https://github.com/user/submodule.git`

In the above examples, you should replace `user` with the relevant username,
and `submodule` with the relelvant repo name. If you are using a hosted git
server other than `github.com`, then you should replace `github.com` with the
relevant URL.\n
"""


@beartype
def main() -> None:
    """Convert a arbitrary subdeck into a git submodule with remote."""
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--kirepo", required=True, help=HELP_KIREPO.lstrip("\n"))
    parser.add_argument("--deck", required=True, help=HELP_DECK.lstrip("\n"))
    parser.add_argument("--remote", required=True, help=HELP_REMOTE.lstrip("\n"))
    args: argparse.Namespace = parser.parse_args()

    # Make a copy of the given repository.
    kirepo = M.kirepo(F.chk(Path(args.kirepo)))
    repo: git.Repo = kirepo.repo
    head: Rev = M.head(repo)
    copy: git.Repo = cp_repo(head, f"submodule-{head.sha}")
    copyroot: Dir = F.root(copy)

    # Harden all symlinks.
    _ = map(M.hardlink, F.walk(copyroot))

    # Filter ``copy`` to include only the given deck.
    print(f"Operating on '{copyroot}'")
    F.chdir(copyroot)
    arglist: List[str] = ["--subdirectory-filter", args.deck, "--force"]
    gfrargs = git_filter_repo.FilteringOptions.parse_args(arglist, error_on_empty=False)
    git_filter_repo.RepoFilter(gfrargs).run()

    # Clean up.
    copy.git.reset(hard=True)
    copy.git.gc(aggressive=True)
    copy.git.prune()

    # Delete all remotes.
    for remote in copy.remotes:
        copy.delete_remote(remote)
        assert not remote.exists()
    assert len(copy.remotes) == 0

    # Add the new remote.
    remote = copy.create_remote(REMOTE_NAME, args.remote)
    assert remote.exists()

    # Commit in the submodule.
    copy.git.add(all=True)
    copy.index.commit("Initial submodule commit.")
    out = copy.git.push("--set-upstream", REMOTE_NAME, BRANCH_NAME, force=True)
    print(out)

    # Remove the deck and commit the deletion.
    repo.git.rm(args.deck, r=True)
    repo.index.commit(f"Remove '{args.deck}'")

    # Remove folder in case git does not remove it.
    # This happens if there are empty folders inside the removed desks folder, such as _media
    deskdir: Dir = F.root(repo).joinpath(args.deck)
    if deskdir.exists():
        F.rmtree(deskdir)

    # Add, initialize, and update the submodule.
    repo.git.submodule("add", args.remote, args.deck)
    repo.git.submodule("init")
    repo.git.submodule("update")

    # Commit the addition.
    repo.git.add(".gitmodules", args.deck)
    sha = repo.index.commit(f"Add submodule '{args.deck}' with remote '{args.remote}'")
    print(f"Committed submodule at rev '{sha}'")


if __name__ == "__main__":
    main()
