#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: random.py
#   Author: xyy15926
#   Created: 2023-12-29 14:01:54
#   Updated: 2024-11-14 22:05:35
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

from random import uniform, normalvariate
from math import sin, cos, pi, sqrt, log, exp

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
def box_muller() -> Iterator:
    while True:
        a = uniform(0, 1)
        b = uniform(0, 1)
        yield cos(2 * pi * a) * sqrt(-2 * log(b))
        yield sin(2 * pi * a) * sqrt(-2 * log(b))


# %%
def metropolis(
    pdf: Callable,
    N: int = 10000,
    start: float = 0
) -> list:
    """Metropolis Sampling.

    Use the gauss distribution as the transition probability for the
    followings:
    1.1 The gauss distribution ranges from -inf to inf.
    1.2 It's easy to draw samples sujecting to gauss distribution and the
      conditional gausss distribution.
    Attention: This orginal metropolis algorithm is sensetive to the
      initialization for the precesion of the `alpha`, which may lead to
      slow convergence to balance status or even stop during the loops.

    Params:
    ---------------------------
    pdf: Target probability density function.
    N: The number of samples.
    start: The first element of the sequence.

    Return:
    ---------------------------
    List of samples.
    """
    def guass_pdf(x):
        return 1 / sqrt(2 * pi) * exp(-0.5 * x**2)
    rands = [0] * N
    # Init the first element of the sequence.
    last = start
    for i in range(N):
        # Draw sample subject to conditional gauss distribution.
        new = normalvariate(0, 1) + last
        # Draw accept-rejection prob.
        rj = uniform(0, 1)
        # Calcuate the acception rate.
        alpha = guass_pdf(new - last) * pdf(new)
        # Determine to accept or reject sample.
        if rj <= alpha:
            last = new
        rands[i] = last
    return rands


def metropolis_hastings(
    pdf: Callable,
    N: int = 10000,
    start: float = 0
) -> list:
    """Metropolis-Hastings Sampling.

    Use the gauss distribution as the transition probability for the
    followings:
    1.1 The gauss distribution ranges from -inf to inf.
    1.2 It's easy to draw samples sujecting to gauss distribution and the
      conditional gausss distribution.

    Params:
    ---------------------------
    pdf: Target probability density function.
    N: The number of samples.
    start: The first element of the sequence.

    Return:
    ---------------------------
    List of samples.
    """
    rands = [0] * N
    # Init the first element of the sequence.
    last = start
    for i in range(N):
        # Draw sample subject to conditional gauss distribution.
        new = normalvariate(0, 1) + last
        # Draw accept-rejection prob.
        rj = uniform(0, 1)
        # Determine to accept or reject sample.
        if rj < min(1, pdf(new) / pdf(last)):
            last = new
        rands[i] = last
    return rands
