#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: autom.py
#   Author: xyy15926
#   Created: 2024-08-12 10:06:56
#   Updated: 2024-08-12 11:47:27
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from typing import TypeVar, Any, List
try:
    from typing import NamedTuple, Self
except ImportError:
    from typing_extensions import NamedTuple, Self
from collections.abc import Iterator, Callable, Mapping

from IPython.core.debugger import set_trace

import copy
import logging

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")

LEX_ENDFLAG = "$END"


# %%
class DFA:
    """Deterministic Finite Automation.

    1. Represent automation's states transformation with Mapping, along
      with `LEX_ENDFLAG` as the final match.
    2. Search the inner states Mapping for input tokens sequently like
      the states transformation.

    Attrs:
    -----------------------
    word_chain: Mapping storing the states.
    case_sensetive: If to match in case-sensitive schema.
    """
    def __init__(self, patterns: List[str] = None):
        self.states = {}
        if patterns is not None:
            for ptn in patterns:
                self.add(ptn, ptn)

    def add(self, tokens: Iterator,
            dest: Any = True):
        """Add pattern.

        Params:
        -----------------------
        tokens: Iterator of tokens representing the pattern.
        dest: The destination to be returned for the final match.
        """
        wc = self.states
        for ele in tokens:
            wc = wc.setdefault(ele, {})
        else:
            wc[LEX_ENDFLAG] = dest

    def check(self, tokens: Iterator) -> Any | None:
        """Check the input fit with inner patterns.

        Params:
        -----------------------
        tokens: Input tokens.

        Return:
        -----------------------
        None: No match.
        Others: Destination for the final match.
        """
        wc = self.states
        for ele in tokens:
            wc = wc.get(ele, None)
            if wc is None:
                break
        else:
            if LEX_ENDFLAG in wc:
                return wc[LEX_ENDFLAG]
