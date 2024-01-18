#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: random.py
#   Author: xyy15926
#   Created: 2023-12-29 14:01:54
#   Updated: 2024-01-12 17:06:07
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

from random import uniform, gauss
from math import sin, cos, pi, sqrt, log

logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
def gen_uniform() -> Iterator:
    while True:
        yield uniform(0, 1)


def box_muller() -> Iterator:
    while True:
        a = uniform(0, 1)
        b = uniform(0, 1)
        yield cos(2 * pi * a) * sqrt(-2 * log(b))
        yield sin(2 * pi * a) * sqrt(-2 * log(b))


uni_gen = gen_uniform()
norm_gen = box_muller()


# %%
def mcmc(
    prob: Callable,
    N: int = 10000,
    start: float = 0
) -> list:
    rands = [0] * N
    last = start
    for i in range(N):
        new = next(norm_gen) + last
        rj = next(uni_gen)
        if rj < min(1, prob(new) / prob(last)):
            last = new
        rands[i] = last
    return rands
