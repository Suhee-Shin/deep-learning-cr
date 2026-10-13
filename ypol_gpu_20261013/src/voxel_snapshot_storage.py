"""Lossless sparse full-density snapshots with every evaluated hard mask retained."""
import numpy as np


def save_step_arrays(out, step, arrays, *, compact, full_due, has_binary):
    if full_due or (has_binary and not compact):
        folder = out / 'snapshots'
        folder.mkdir(exist_ok=True)
        np.savez_compressed(folder / f'step_{step:04d}.npz', **arrays)
    elif has_binary:
        folder = out / 'binary_candidates'
        folder.mkdir(exist_ok=True)
        keys = ('binary', 'threshold', 'tabs', 'step', 'beta', 'period_um',
                'layer_height_nm', 'axis_order')
        np.savez_compressed(folder / f'step_{step:04d}.npz',
                            **{key: arrays[key] for key in keys})
