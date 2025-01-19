#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: setup.py
#   Author: xyy15926
#   Created: 2023-07-10 14:37:11
#   Updated: 2025-01-19 19:43:14
#   Description:
#     Classifer Ref: <https://pypi.org/pypi?%3Aaction=list_classifiers>
# ---------------------------------------------------------
from setuptools import (
    setup,
    find_packages,
    find_namespace_packages,
)
from setuptools.command.install_scripts import install_scripts

setup(
    name="aki7",
    version="0.2.1",
    author="UBeaRLy",
    author_email="ubearly@outlook.com",
    description="A project for collecting some tricks.",
    url="https://github.com/xyy15926",
    packages=find_namespace_packages(include=[
        "flagbear*",
        "ringbear*",
        "modsbear*",
        "suitbear*",
    ]),
    include_package_data=True,
    # Classifier tags
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console"
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="Data Science",
    license="Apache License 2.0",
    python_requires=">=3.6, <=3.14",
    # Requirements for installment.
    install_requires=[
        "numpy >= 1.2",
        "scipy >= 1.10",
        "pandas >= 1.4",
        "scikit-learn >= 1.1",
        "tqdm >= 4.0",
    ],
    # Plugins packages for setuptools.
    setup_requires=[],
    # Requirements for tests will be installed only when `python setup.py test`
    # is called.
    tests_require=[
        "pytest >= 7, <= 10",
        "ipython >= 8.4.0",
    ],
    # Extra requirments won't be installed automatically, just to indicate
    # the dependencies for specific usage.
    extras_require={
        "ml": [
            "torch",
            "tensorboard",
            "networkx",
            "xgboost",
        ],
        "locale": [
            "chinese_calendar",
            "jieba",
            "pdfplumber",
        ],
        "display": [
            "pyecharts",
        ],
        "db": [
            "SQLAlchemy",
            "sqlite",
            "PyMySQL",
            "cx-Oracle",
        ],
        "quant": [
            "TA-Lib",
            "numpy_financial",
        ],
    },
    entry_points={
        "concole_scripts": [],
    },
    scripts=[],
)
