#!/usr/bin/env bash
set -e

pip install -e .

# Create root repository.
rm -rf /tmp/subtree
rm -rf /tmp/collections
mkdir -p /tmp/subtree
mkdir -p /tmp/collections
cp tests/data/collections/multideck.anki2 /tmp/collections/multideck.anki2
cd /tmp/subtree
ki clone /tmp/collections/multideck.anki2
cd /tmp/subtree/multideck

# Create subtree (local) remote.
mkdir /tmp/subtree/github
cd /tmp/subtree/github
git init --initial-branch main
git checkout -b alt
echo 'a' > a
git add a
git commit -m "Initial commit for branch 'alt'"

# Convert `aa` to a subdeck with remote `/tmp/subtree/github`
# and clone subdeck repo to `tmp/ki/aa`.
echo ""
cd /tmp/subtree/multideck
subdeck aa /tmp/subtree/github
git clone -b main /tmp/subtree/github /tmp/subtree/aa

# Check that cloned subdeck repo has a remote pointing to `/tmp/subtree/github` as well.
echo ""
cd /tmp/subtree/aa
git remote -v

# Commit in the root and push to subdeck.
echo ""
cd /tmp/subtree/multideck/aa
echo 'b' > b
git add b
git commit -m "Add b"
cd /tmp/subtree/multideck
git subtree push --prefix aa /tmp/subtree/github main

# Commit in the subdeck repo, push to remote, and pull into root.
cd /tmp/subtree/aa
echo 'c' > c
git add c
git commit -m "Add c"
git config pull.rebase false
git pull --no-edit
git push origin main
cd /tmp/subtree/multideck
git subtree pull -m "Merge branch 'main' of /tmp/subtree/github" --prefix aa /tmp/subtree/github main

echo ""
git log --oneline -n 20
git log --oneline -n 20 | grep "Add b"
git log --oneline -n 20 | grep "Add c"

echo ""
cd /tmp/subtree/github
git checkout main
git log --oneline -n 20
git log --oneline -n 20 | grep "Add b"
git log --oneline -n 20 | grep "Add c"
git checkout -
