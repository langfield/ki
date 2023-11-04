#!/usr/bin/env bash

rm -rf /tmp/ki
mkdir -p /tmp/ki
cp -r tests/data/repos/multideck /tmp/ki/multideck
cd /tmp/ki/multideck
git init --initial-branch main
git add .
git commit -m "Initial commit"
cd -
root=$(pwd)
cd  /tmp/ki
mkdir github
cd github
git init --initial-branch main
git checkout -b alt
cd /tmp/ki/multideck

echo ""
"$root/subtree.sh" aa /tmp/ki/github /tmp/ki/target
