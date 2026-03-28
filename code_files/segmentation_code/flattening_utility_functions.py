# from code_files.segmentation_code.custom_dataclasses import np
import numpy as np


def warp_line_by_shift(y_line, warper_shift_y_full, direction="to_flat"):
    """
    Adjust ANY other 1D line (len=W) by the hypersmoother shift.

    y_line: array-like, shape (W,), y coords in ORIGINAL frame by default.
    direction:
        - "to_flat": map original->flattened coordinates  (subtract shift)
        - "to_orig": map flattened->original coordinates  (add shift)

    Example (ILM line):
        ilm_flat = warp_line_by_hypersmooth(ilm_orig, hs, "to_flat")
        ilm_back = warp_line_by_hypersmooth(ilm_flat, hs, "to_orig")
    """
    y_line = np.asarray(y_line, dtype=np.float32)
    shift = warper_shift_y_full.astype(np.float32, copy=False)

    assert y_line.shape[0] == shift.shape[0], "y_line must match width W"

    if direction == "to_flat":
        return y_line + shift
    elif direction == "to_orig":
        return y_line - shift
    else:
        raise ValueError("direction must be 'to_flat' or 'to_orig'")

def flatten_to_path(img, y_path_full, *, fill=0.0, target_y=None):
    """
    Column-warp img so y_path_full becomes horizontal at target_y (median by default).

    img: (H,W)
    y_path_full: (W,) float, y location per column in *img coordinates*
    Returns: flat_img (float32), shift_y_full (float32), target_y (float)
    """
    H, W = img.shape
    y_path_full = np.asarray(y_path_full, dtype=np.float32)
    if target_y is None:
        target_y = float(np.median(y_path_full))

    shift_y_full = (target_y - y_path_full).astype(np.float32)  # + => shift DOWN

    y = np.arange(H, dtype=np.float32)
    imgf = img.astype(np.float32, copy=False)
    flat = np.empty((H, W), dtype=np.float32)
    for j in range(W):
        src = y - float(shift_y_full[j])
        flat[:, j] = np.interp(src, y, imgf[:, j], left=fill, right=fill)

    return flat, shift_y_full, float(target_y)