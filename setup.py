#!/usr/bin/env python
import setuptools


def read_file(filename):
    """Fetch the contents of a file"""
    with open(filename) as file:
        return file.read()

setuptools.setup(
    name=package.__name__.replace('_', '-'),
    version=package.__version__,
    description=package.__doc__.strip().split('\n')[0],
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    url=package.__url__,
    author=package.__author__,
    author_email=package.__email__,
    license=package.__license__,
    packages=["ki"],
    install_requires=read_file('requirements.txt'),
    entry_points={
        'console_scripts': [
            'ki = ki:ki',
        ],
    },
)
