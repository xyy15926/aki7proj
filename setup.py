#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: setup.py
#   Author: xyy15926
#   Created: 2023-07-10 14:37:11
#   Updated: 2023-07-10 15:38:33
#   Description:
# ---------------------------------------------------------
from setuptools import setup, find_packages
from setuptools.command.install_scripts import install_scripts

setup(
    name='ringbear',
    version='0.1',
    author='UBeaRLy',
    author_email='ubearly@outlook.com',
    description='A project for collecting same tricks.',
    url='',
    packages=find_packages(),
    # Classifier tags
    classifier=[
        'Development Status :: 3 - Alpha',
        'Intended :: Data Analysis(Mainly)',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
    python_requires='>=3.6, <=3.11',
    # Requirements for installment.
    install_requires=[
        'rich >= 13, != 13.1, <= 15',
    ],
    # Requirements for tests will be installed only when `python setup.py test`
    # is called.
    tests_requires=[
        'pytest >= 7, <= 10',
    ],
    # Extra requirments won't be installed automatically, just to indicate
    # the dependencies for specific usage.
    extra_require={
        'PDF': ['ReportLab>=1.2', 'RXP'],
    },
    entry_points={
        'concole_scripts': [],
    },
)

