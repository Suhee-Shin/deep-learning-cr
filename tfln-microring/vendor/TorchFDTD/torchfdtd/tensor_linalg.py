"""Bounded CUDA linear-algebra batches for dense 3x3 material grids."""
import torch


def inverse_3x3(value, batch_size=4096):
    """Keep inversion differentiable without a single grid-sized solver call."""
    if value.shape[-2:] != (3, 3) or batch_size < 1:
        raise ValueError('Require (...,3,3) tensors and positive batch size.')
    flat = value.reshape(-1, 3, 3)
    if not len(flat):
        return value.clone()
    return torch.cat([torch.linalg.inv(batch) for batch in flat.split(batch_size)], dim=0).reshape_as(value)


def symmetric_eigenvalue_bounds(value, batch_size=4096):
    """Global minimum/maximum using bounded vendor eigensolver workspace."""
    if value.shape[-2:] != (3, 3) or batch_size < 1 or value.numel() == 0:
        raise ValueError('Require nonempty (...,3,3) tensors and positive batch size.')
    lowest = value.new_full((), float('inf'))
    highest = value.new_full((), -float('inf'))
    for batch in value.reshape(-1, 3, 3).split(batch_size):
        eigenvalues = torch.linalg.eigvalsh(batch)
        lowest = torch.minimum(lowest, eigenvalues.min())
        highest = torch.maximum(highest, eigenvalues.max())
    return lowest, highest
