#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_nested.py
#   Author: xyy15926
#   Created: 2025-11-17 15:24:23
#   Updated: 2025-11-17 18:04:55
#   Description:
# ---------------------------------------------------------

# %%
import pytest
from pytest import mark
from packaging.version import Version
import torch
from torch import nested
from torch.nn import functional as F
from torch.nested._internal import sdpa
from torch.backends.cuda import SDPAParams


# %%
@pytest.mark.skipif(Version(torch.__version__) <= Version("2.7"),
                    reason="Nested-Tensor behavior isn't decided yet.")
@mark.filterwarnings("ignore: .*PyTorch API of nested tensors is in prototype.*")
def test_scaled_dot_product_attention_njt():
    # 1. The size of last dimension must be 8n.
    query = torch.randn(3, 4, 8, dtype=torch.float64)
    key = torch.randn(3, 6, 8, dtype=torch.float64)
    value = torch.randn(3, 6, 8, dtype=torch.float64)

    def build_nested(query, key, value, qlens, kvlens):
        # 2. The ragged-dim can't be the dimension for `heads_n`, which will be
        #   regarded as the dim-1 for 3D Tensor.
        # (bsz, slen, dim) => (bsz, slen, heads_n, qsz)
        # (3, 4, 8) => (3, j2, 8) => (3, j2, 1, 8)
        query = nested.narrow(
            query, dim=1, start=0, length=qlens, layout=torch.jagged
        ).contiguous().unflatten(-1, (-1, 8)).transpose(2, 1)
        key = nested.narrow(
            key, dim=1, start=0, length=kvlens, layout=torch.jagged
        ).contiguous().unflatten(-1, (-1, 8)).transpose(2, 1)
        value = nested.narrow(
            value, dim=1, start=0, length=kvlens, layout=torch.jagged
        ).contiguous().unflatten(-1, (-1, 8)).transpose(2, 1)

        return query, key, value

    qlens = torch.IntTensor([2, 4, 3])
    kvlens = qlens
    q, k, v = build_nested(query, key, value, qlens, kvlens)
    params = SDPAParams(q, k, v, None, 0.0, False, False)
    assert sdpa._can_use_flash_sdpa_jagged(params, True)
    assert sdpa._can_use_efficient_sdpa_jagged(params, True)
    assert sdpa._can_use_math_sdpa_jagged(params, True)
    # `F.scaled_dot_product_attention` does a lot of preprocess for
    # NestedTensor.
    # https://github.com/pytorch/pytorch/blob/main/torch/nested/_internal/sdpa.py
    foutp = F.scaled_dot_product_attention(q, k, v)
    assert foutp is not None

    kvlens = torch.IntTensor([3, 5, 4])
    q, k, v = build_nested(query, key, value, qlens, kvlens)
    params = SDPAParams(q, k, v, None, 0.0, False, False)
    assert sdpa._can_use_flash_sdpa_jagged(params, True)
    assert sdpa._can_use_efficient_sdpa_jagged(params, True)
    assert sdpa._can_use_math_sdpa_jagged(params, True)
    foutp = F.scaled_dot_product_attention(q, k, v)
    assert foutp is not None
