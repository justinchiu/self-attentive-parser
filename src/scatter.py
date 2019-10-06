import torch
from torch_scatter import scatter_add, scatter_max

def scatter_lse(src, index, dim=-1, out=None, dim_size=None, fill_value=float("-inf")):
    dim_size = out.shape[dim] if dim_size is None and out is not None else dim_size
    max_value = scatter_max(src, index, dim=dim, dim_size=dim_size)[0]
    M = max_value.gather(dim, index)
    tmp = scatter_add(
        (src - M).exp(),
        index,
        dim=dim,
        out=out,
        dim_size=dim_size,
        fill_value=fill_value,
    )
    return torch.log(tmp + 1e-16) + max_value
