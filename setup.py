#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: setup.py
#   Author: xyy15926
#   Created: 2023-07-10 14:37:11
#   Updated: 2025-01-20 21:29:43
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
    name="ubears",
    version="0.2.3",
    author="UBeaRLy",
    author_email="ubearly@outlook.com",
    description="A project for collecting some tricks.",
    url="https://github.com/xyy15926",
    packages=find_namespace_packages(
        where="src",
        include=[
            "ubears*"
        ],
    ),
    package_dir={"": "src"},
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
    python_requires=">=3.8, <=3.14",
    # Requirements for installment.
    install_requires=[
        "numpy >= 1.2, <= 2.2.2",
        "scipy >= 1.10, <= 1.15.1",
        "pandas >= 1.4, <= 2.2.3",
        "scikit-learn >= 1.1, <= 1.6.1",
        "tqdm >= 4.0",
        "pyecharts >= 2.0",
        "SQLAlchemy >= 1.4",
        "chinese_calendar",
        "jieba",
        "pdfplumber",
        "numpy_financial",
    ],
    # Plugins packages for setuptools.
    setup_requires=[],
    # Requirements for tests will be installed only when `python setup.py test`
    # is called.
    tests_require=[
        "pytest >= 7, <= 10",
        "ipython >= 8.4.0",
        "flake8",
    ],
    # Extra requirments won't be installed automatically, just to indicate
    # the dependencies for specific usage.
    extras_require={
        "all": [
            "torch",
            "tensorboard",
            "networkx",
            "xgboost",
            "PyMySQL",
            "cx-Oracle",
            "TA-Lib",
        ],
    },
    entry_points={
        "concole_scripts": [],
    },
    scripts=[],
)
