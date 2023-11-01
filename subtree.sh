#!/usr/bin/env bash
set -e
subdeck=$(echo "$1" | sed 's,/*$,,')
remote=$2
root=$(git rev-parse --show-toplevel)

echo $root

rm -rf /tmp/ki
mkdir -p /tmp/ki
cd /tmp/ki
git clone $root $subdeck
cd $subdeck
git filter-repo --force --subdirectory-filter $subdeck --path-rename $subdeck/:
cd $(git rev-parse --show-toplevel)
git remote add origin $remote
git push -u origin main

cd $root
git rm -r $subdeck
git commit -m "Remove `$subdeck`"
git subtree add --prefix $subdeck $remote main --squash
