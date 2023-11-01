#!/usr/bin/env bash
mkdir big-repo
cd big-repo
git init --initial-branch=main

echo ''
echo '0' > 0
git add .
git commit -m "Initial commit"

echo ''
mkdir little-repo
cd little-repo
echo 'a' > a
git add .
git commit -m "little-repo"
cd ../..

# Here we want to add stuff that will push this to github.
# We will take the SSH remote as a command-line argument.
echo ''
git clone ./big-repo little-repo
cd little-repo
git filter-repo --force --subdirectory-filter little-repo --path-rename little-repo/:
cd ..

echo ''
cd big-repo
git rm -r little-repo
git commit -m "Remove little-repo"
git subtree add --prefix little-repo git@github.com:langfield/subtree.git main --squash
