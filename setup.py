#!/usr/bin/env python
import setuptools


def read_file(filename):
    """Fetch the contents of a file"""
    with open(filename) as file:
        return file.read()


setuptools.setup(
    name="ki",
    version="0.0.18a",
    description="",
    url="",
    author="",
    author_email="",
    license="",
    packages=["ki"],
    package_data={"ki": ["grammar.lark", "jekyll-gh-pages.yml"]},
    include_package_data=True,
    install_requires=read_file("requirements.txt"),
    scripts=["subdeck"],
    entry_points={
        "console_scripts": [
            "ki = ki:ki",
        ],
    },
)
