#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: fixture.py
#   Author: xyy15926
#   Created: 2025-11-20 16:26:50
#   Updated: 2025-11-22 20:48:36
#   Description:
# ---------------------------------------------------------

# %%
import torch
import torch.nn.functional as F
try:
    import torch_directml
    DML = torch_directml.device()
    fkwargs_32_dml = {
        "dtype": torch.float32,
        "device": DML,
    }
    fkwargs_64_dml = {
        "dtype": torch.float64,
        "device": DML,
    }
except ImportError as e:
    DML = None
    fkwargs_32_dml = {}
    fkwargs_64_dml = {}

torch.autograd.set_detect_anomaly(False)
fkwargs_32_cpu = {
    "dtype": torch.float32,
    "device": torch.device("cpu"),
}
fkwargs_64_cpu = {
    "dtype": torch.float64,
    "device": torch.device("cpu"),
}


# %%
def all_close(
    lt = torch.Tensor,
    rt = torch.Tensor,
    lnan_to_zero = False,
    rnan_to_zero = False,
    equal_nan = True,
    rtol = 1e-5,
    atol = 1e-3,
) -> bool:
    """Check if `lt` and `rt` are all close element-wisely.

    Params:
    ------------------------
    lt: Left tensor.
    rt: Right tensor or float scalar.
    lnan_to_zero: Replace NaN to 0 in `lt`.
    rnan_to_zero: Replace NaN to 0 in `rt`.
    equal_nan: Treat NaN in `lt` and `rt` are equal.
    rtol: Relative tolerance in comparison.
    atol: Absolute tolerance in comparison.

    Return:
    ------------------------
    Bool scalar.
    """
    if lnan_to_zero and torch.is_tensor(lt):
        lt = torch.nan_to_num(lt, 0.0)
    if (torch.is_tensor(lt)
        and equal_nan
        and lt.device == DML
        and lt.dtype == torch.float64
    ):
        # `float64` doesn't support `torch.allclose(equal_nan=True)`.
        # So to cast the dtype to `float32` implicitly.
        lt = lt.to(dtype=torch.float32)
    if rnan_to_zero and torch.is_tensor(rt):
        rt = torch.nan_to_num(rt, 0.0)
    if torch.is_tensor(rt):
        # In case `lt`, `rt` has different dtypes and devices, as `.to` can't
        # move tensor from GPU to CPU and cast dtype simutaneously.
        rt = (
            rt.to(dtype=lt.dtype, device=lt.device)
            .to(dtype=lt.dtype, device=lt.device)
        )
        return torch.allclose(
            lt,
            rt,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan
        )
    else:
        return torch.allclose(
            lt,
            torch.tensor([rt], dtype=lt.dtype, device=lt.device),
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan
        )


# %%
def ssoftmax(
    inp: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """All-NInf-safe Softmax.

    1. 0.0 will be filled for all-NInf axis instead of NaN.
    2. This maybe a litter slower for maximum check and transpose.

    Params:
    -------------------------------
    inp: Input tensor.
    dim: Dimension along which softmax will be computed.

    Return:
    -------------------------------
    Softmax result with 0.0 filled for shoud-be NaN.
    """
    # Transpose the dimension for softmax so to fit up with the bool index.
    if dim != -1 or dim != inp.dim():
        inp = inp.transpose(-1, dim)
    valid_pos = inp.max(dim=-1).values > -torch.inf
    ret = torch.zeros_like(inp, dtype=inp.dtype, device=inp.device)
    ret[valid_pos] = F.softmax(inp[valid_pos], dim=-1)
    if dim != -1 or dim != inp.dim():
        ret = ret.transpose(-1, dim)
    return ret
