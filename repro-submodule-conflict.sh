#!/usr/bin/env bash
# Populate repo.
mkdir repo
mkdir repo/a/
mkdir repo/b/
echo 'a' > repo/a/a.txt
echo 'b' > repo/b/b.txt
cd repo
git init
git add .
git commit -m "Initial commit"

# Create remote branch with new commit.
git checkout -b remote
echo 'c' > b/c.txt
echo 'bb' > b/b.txt
git add .
git commit -m "Changes"

# Convert `b` into a submodule on `master`.
git checkout master
cd ..
git clone repo copy
cd copy
git filter-repo --subdirectory-filter b
cd ..
cd repo
git rm -rf b
git add .
git commit -m "Remove b"
git submodule add ../copy b
git commit -m "Add b as submodule"

# Make changes within the submodule.
echo 'd' > b/d.txt
cd b
git add .
git commit -m "Add new file in submodule b"

# Commit submodule changes on `master`.
cd ..
git add .
git commit -m "Add submodule commit"

# Start the merge procedure.
git status
git submodule deinit --all
git checkout remote
rm -rf /tmp/b
mv b /tmp/b
git add .
git commit -m "Remove b"
git checkout master
git merge --no-edit remote
git submodule sync
git submodule update --init
git status
cd b
git status

# Initialize a new repository with the copied files from remote
cd /tmp/b
git init
git add .
git commit -m "Initial commit"

# Merge the new repository into the submodule
cd -
git remote add newstuff /tmp/b
git fetch newstuff
git merge --no-edit --allow-unrelated-histories newstuff/master
git status
