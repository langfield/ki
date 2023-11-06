#!/usr/bin/env bash
set -e

# Create root repository.
rm -rf /tmp/subtree
mkdir -p /tmp/subtree
cp -r tests/data/repos/multideck /tmp/subtree/multideck
cd /tmp/subtree/multideck
git init --initial-branch main
git add .
git commit -m "Initial commit"

# Create subtree (local) remote.
mkdir /tmp/subtree/github
cd /tmp/subtree/github
git init --initial-branch main
git checkout -b alt
echo 'a' > a
git add a
git commit -m "Initial commit for branch 'alt'"

# Convert `aa` to a subdeck with remote `/tmp/subtree/github`
# and clone subdeck repo to `/tmp/subtree/aa`.
echo ""
cd /tmp/subtree/multideck
subdeck aa /tmp/subtree/github
git clone -b main /tmp/subtree/github /tmp/subtree/aa

# Check that cloned subdeck repo has a remote pointing to `/tmp/subtree/github` as well.
echo ""
cd /tmp/subtree/aa
git remote -v

echo ""
git log --oneline -n 20

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

# Convention here is that we squash when pulling into the root repository.
git subtree pull -m "Merge branch 'main' of /tmp/subtree/github" --prefix aa /tmp/subtree/github main --squash

echo ""
git log --oneline -n 20

echo ""
git log --oneline -n 20 | grep "Add b"
cat aa/c | grep 'c'


echo ""
cd /tmp/subtree/github
git log --oneline -n 20

echo ""
git checkout main
git log --oneline -n 20 | grep "Add b"
git log --oneline -n 20 | grep "Add c"
git checkout -
