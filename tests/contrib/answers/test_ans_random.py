#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_ans_random.py
#   Author: xyy15926
#   Created: 2024-01-12 16:51:26
#   Updated: 2024-01-12 17:01:36
#   Description:
# ---------------------------------------------------------

# %%
import pytest
import numpy as np

if __name__ == "__main__":
    from importlib import reload
    from answers import ans_random
    reload(ans_random)

from answers.ans_random import mcmc


# %%

def test_mcmc():
    def target_dist(x):
        mu = 8
        sigma = 3
        return 1 / sigma / np.sqrt(2 * np.pi) * np.exp(-0.5 * (x - mu)**2 / sigma ** 2)

    rands = mcmc(target_dist, 20000)
    assert np.isclose(np.mean(rands), 8, atol=1)
    assert np.isclose(np.std(rands), 3, atol=1)
