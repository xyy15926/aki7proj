#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: deco.py
#   Author: xyy15926
#   Created: 2023-12-22 17:43:34
#   Updated: 2023-12-22 17:57:07
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from typing import TypeVar, Any
try:
    from typing import NamedTuple, Self
except ImportError:
    from typing_extensions import NamedTuple, Self
from collections.abc import Iterator, Callable
import logging

from functools import wraps

logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
def deprecated(to_: str | None = None):
    logger.info("Deprecated function, maybe try {to_}.")

    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorate
