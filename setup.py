#!/usr/bin/env python
import setuptools


def read_file(filename):
    """Fetch the contents of a file"""
    with open(filename) as file:
        return file.read()

setuptools.setup(
    name='ki',
    version='0.0.1a',
    description='',
    url='',
    author='',
    author_email='',
    license='',
    packages=["ki"],
    install_requires=read_file('requirements.txt'),
    entry_points={
        'console_scripts': [
            'ki = ki:ki',
        ],
    },
)
