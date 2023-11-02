#!/usr/bin/env bash
set -e
subdeck=$(echo "$1" | sed 's,/*$,,')
remote=$2
target=$3
root=$(git rev-parse --show-toplevel)

[ -d $target ] && echo "target directory '$target' already exists" && exit 1
[ -f $target ] && echo "a file already exists at '$target'" && exit 1

mkdir -p $target
cd $target
git clone $root $subdeck
cd $subdeck
git filter-repo --force --subdirectory-filter $subdeck --path-rename $subdeck/:
cd $(git rev-parse --show-toplevel)
git remote add origin $remote
git push -u origin main

cd $root
git rm -r $subdeck
git commit -m "Remove \`$subdeck\`"
git subtree add --prefix $subdeck $remote main --squash
echo "repo for deck '$subdeck' created at '$target'"
