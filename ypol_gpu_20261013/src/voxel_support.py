"""Binary support and vent projections applied before electromagnetic evaluation."""
import numpy as np
from scipy.ndimage import generate_binary_structure, label


def add_posts(mask):
    mask = np.asarray(mask, dtype=bool).copy()
    while True:
        labs, _ = label(mask, generate_binary_structure(3, 1))
        anchors = np.unique(labs[-1])
        floating = mask & ~np.isin(labs, anchors[anchors > 0])
        if not floating.any():
            return mask.astype(np.uint8)
        ids, sizes = np.unique(labs[floating], return_counts=True)
        coords = np.argwhere(labs == ids[np.argmax(sizes)])
        bottom = coords[coords[:, 0] == coords[:, 0].max()]
        z, x, y = min(bottom, key=lambda q: (~mask[q[0]:, q[1], q[2]]).sum())
        mask[z:, x, y] = True


def closed_air(mask):
    air = np.asarray(mask) == 0
    labs, _ = label(air, generate_binary_structure(3, 1))
    exterior = np.unique(np.concatenate([labs[0].ravel(), labs[:, 0, :].ravel(), labs[:, -1, :].ravel(),
                                         labs[:, :, 0].ravel(), labs[:, :, -1].ravel()]))
    return air & ~np.isin(labs, exterior[exterior > 0])


def supported_and_vented(mask):
    """Discrete posts/vents; filled-column fallback prevents cyclic repairs."""
    mask = np.asarray(mask, dtype=np.uint8).copy()
    for _ in range(8):
        mask = add_posts(mask)
        enclosed = closed_air(mask)
        if not enclosed.any():
            return mask
        labs, n = label(enclosed, generate_binary_structure(3, 1))
        for component in range(1, n+1):
            coords = np.argwhere(labs == component)
            top = coords[coords[:, 0] == coords[:, 0].min()]
            z, x, y = min(top, key=lambda q: mask[:q[0]+1, q[1], q[2]].sum())
            mask[:z+1, x, y] = 0
    return np.maximum.accumulate(mask, axis=0)
