#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: tree.py
#   Author: xyy15926
#   Created: 2023-12-15 14:01:08
#   Updated: 2023-12-16 15:34:31
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


logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")


TreeVal = TypeVar("TreeVal")


# %%
# TODO: No-recursive version.
class BiTNode:
    """
    Binary tree node.

    Attrs:
    ---------------------
    val: TreeVal
      Value of the node.
    lc: Self
      Left chlid.
    rc: Self
      Right child.
    """
    def __init__(
        self, val: TreeVal,
        left: Self | None = None,
        right: Self | None = None,
    ):
        self.val = val
        self.lc = left
        self.rc = right

    def __repr__(self):
        return f"{self.__class__}: {self.val}"

    def pre_order(self, visit: Callable[[Self], Any] | None = None) -> Any:
        pass

    def in_order(self, visit: Callable[[Self], Any] | None = None) -> Any:
        pass

    def post_order(self, visit: Callable[[Self], Any] | None = None) -> Any:
        visit_none = False
        if visit is None:
            visit_none = True
            st = []
            visit = st.append

        lc, rc = self.lc, self.rc
        if lc is not None:
            lc.post_order(lc, visit)
        if rc is not None:
            rc.post_order(rc, visit)
        ret = visit(self)

        return st if visit_none else ret


# %%
# TODO: No-recursive version.
class GeTNode:
    """
    Generic tree node.

    Attrs:
    ---------------------
    val: TreeVal
      Value of the node.
    chs: list[Self]
      Children tree nodes.
    """
    def __init__(
        self, val: TreeVal,
        children: list[Self] | None = None,
    ):
        self.val = val
        self.chs = children
        self.chn = 0 if self.chs is None else len(children)

    def __repr__(self):
        return f"{self.__class__}: {self.val} with {self.chn} children."

    def post_order(self, visit: Callable[[Self], Any] | None = None) -> Any:
        visit_none = False
        if visit is None:
            visit_none = True
            st = []
            visit = st.append

        chs = self.chs
        if chs is not None:
            for ch in chs:
                ch.post_order(visit)
        ret = visit(self)

        return st if visit_none else ret
