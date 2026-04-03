from __future__ import annotations

import numpy as np

from code_files.texture_package_prod.texture_extraction_utilities import (
    compute_dense_texture_maps,
)

from pathlib import Path

def _reduce_between_bounds(
    volume,
    upper,
    lower,
    stat="mean",
    interp_x=True,
    extra_pad=2,
):
    """
    Reduce a flattened (Z,Y,X) volume into a (Z,X) en-face map
    between upper and lower Y-bounds.
    """
    if stat == "mean":
        projector = TextureSlabProjector(
            texture_vol=volume,
            reference_line=(np.asarray(upper) + np.asarray(lower)) / 2,
            max_top_offset=0,
            max_bottom_offset=0,
            extra_pad=extra_pad,
        )
        return projector.project_between(upper, lower, interp_x=interp_x)

    if stat not in {"median", "std"}:
        raise ValueError("stat must be 'mean', 'median', or 'std'")

    vol = np.asarray(volume, dtype=np.float32)
    upper = np.asarray(upper, dtype=np.float32)
    lower = np.asarray(lower, dtype=np.float32)

    top = np.floor(np.minimum(upper, lower)).astype(np.int32)
    bottom = np.ceil(np.maximum(upper, lower)).astype(np.int32)

    y0 = max(0, int(np.nanmin(top)) - extra_pad)
    y1 = min(vol.shape[1] - 1, int(np.nanmax(bottom)) + extra_pad)

    crop = vol[:, y0:y1 + 1, :]
    top = np.clip(top - y0, 0, crop.shape[1] - 1)
    bottom = np.clip(bottom - y0, 0, crop.shape[1] - 1)

    out = np.full((crop.shape[0], crop.shape[2]), np.nan, dtype=np.float32)

    for z in range(crop.shape[0]):
        for x in range(crop.shape[2]):
            lo = min(top[z, x], bottom[z, x])
            hi = max(top[z, x], bottom[z, x]) + 1
            vals = crop[z, lo:hi, x]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            out[z, x] = np.nanmedian(vals) if stat == "median" else np.nanstd(vals)

    if interp_x:
        for z in range(out.shape[0]):
            good = np.isfinite(out[z])
            if good.sum() >= 2:
                out[z] = np.interp(np.arange(out.shape[1]), np.where(good)[0], out[z, good])
            elif good.sum() == 1:
                out[z, :] = out[z, good][0]

    return out


def slab_stat_map(
    flat_volume,
    reference_line,
    bottom_offset,
    top_offset,
    stat="mean",
    interp_x=True,
):
    """
    Positive offsets are upward in the image.
    """
    ref = np.asarray(reference_line, dtype=np.float32)
    upper = ref - top_offset
    lower = ref - bottom_offset

    return _reduce_between_bounds(
        volume=flat_volume,
        upper=upper,
        lower=lower,
        stat=stat,
        interp_x=interp_x,
    )


def full_retina_stat_map(
    flat_volume,
    ilm_line,
    rpe_line,
    stat="mean",
    interp_x=True,
):
    return _reduce_between_bounds(
        volume=flat_volume,
        upper=ilm_line,
        lower=rpe_line,
        stat=stat,
        interp_x=interp_x,
    )


def retinal_thickness_map(
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    abs_value: bool = True,
) -> np.ndarray:
    """
    Return a (Z, X) thickness map from two surfaces.
    For current use this is usually ilm_smooth minus the algorithm-selected RPE-family line.
    """
    upper = np.asarray(upper_bound, dtype=np.float32)
    lower = np.asarray(lower_bound, dtype=np.float32)

    if upper.shape != lower.shape:
        raise ValueError('upper_bound and lower_bound must have the same shape')

    out = upper - lower
    if abs_value:
        out = np.abs(out)
    return out.astype(np.float32, copy=False)


def compute_extra_enface_maps(
    flat_volume,
    flat_layers,
    reference_key="rpe_smooth",
    slab_offsets=((5, 15),),
    interp_x=True,
):
    """
    Returns direct en-face maps from a flattened image volume.
    """
    out = {}

    rpe = np.asarray(flat_layers[reference_key], dtype=np.float32)
    ilm = np.asarray(flat_layers["ilm_smooth"], dtype=np.float32)

    out["retinal_thickness"] = retinal_thickness_map(ilm, rpe, abs_value=True)

    for bottom, top in slab_offsets:
        # for stat in ("mean", "median", "std"):
        for stat in (["mean"]):
            out[f"slab_{stat}|{bottom}->{top}"] = slab_stat_map(
                flat_volume=flat_volume,
                reference_line=rpe,
                bottom_offset=bottom,
                top_offset=top,
                stat=stat,
                interp_x=interp_x,
            )

    # for stat in ("mean", "median", "std"):
    for stat in (["mean"]):
        out[f"full_retina_{stat}"] = full_retina_stat_map(
            flat_volume=flat_volume,
            ilm_line=ilm,
            rpe_line=rpe,
            stat=stat,
            interp_x=interp_x,
        )

    return out


def compute_textures_on_enface_maps(
    enface_maps,
    window=31,
    step=4,
    families=("firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm", "lbp", "gradient"),
    include_wavelet=False,
    levels=16,
    min_valid_frac=0.5,
    n_jobs=8,
):
    """
    Run the usual texture family on each 2D en-face map.
    """
    out = {}

    for map_name, arr in enface_maps.items():
        arr = np.asarray(arr, dtype=np.float32)
        mask = np.isfinite(arr)

        feat_maps, _ = compute_dense_texture_maps(
            image=arr,
            window=window,
            step=step,
            mask=mask,
            families=families,
            include_wavelet=include_wavelet,
            levels=levels,
            min_valid_frac=min_valid_frac,
            n_jobs=n_jobs,
        )

        for feat_name, feat_map in feat_maps.items():
            out[f"{map_name}__{feat_name}"] = feat_map

    return out



class CompactTextureSlabProjector:
    """
    Project a compact texture volume shaped (Z, R, C) into en-face maps.

    Important:
    - does NOT materialize the full compact volume in RAM
    - reads one Z-slice at a time from the zarr-backed dataset
    - projects first on the sampled compact X grid
    - upsample_x(...) can then expand to full X afterward
    """
    def __init__(
        self,
        texture_vol,                  # zarr dataset or ndarray, shape (Z, R, C)
        row_centers_full: np.ndarray, # sampled row centers in full-image Y coords
        col_centers: np.ndarray,      # sampled col centers in full-image X coords
    ):
        if getattr(texture_vol, "ndim", None) != 3:
            raise ValueError("texture_vol must be (Z, R, C)")

        self.texture_vol = texture_vol
        self.row_centers_full = np.asarray(row_centers_full, dtype=np.int32)
        self.col_centers = np.asarray(col_centers, dtype=np.int32)

        self.Z, self.R, self.C = texture_vol.shape

        if len(self.row_centers_full) != self.R:
            raise ValueError("row_centers_full length does not match texture_vol.shape[1]")
        if len(self.col_centers) != self.C:
            raise ValueError("col_centers length does not match texture_vol.shape[2]")

    def _sample_bounds(self, upper_bound: np.ndarray, lower_bound: np.ndarray):
        """
        Convert full-image bounds (Z, X_full) into compact sampled-row indices (Z, C).
        """
        upper_s = np.asarray(upper_bound, dtype=np.float32)[:, self.col_centers]
        lower_s = np.asarray(lower_bound, dtype=np.float32)[:, self.col_centers]

        top = np.minimum(upper_s, lower_s)
        bottom = np.maximum(upper_s, lower_s)

        start = np.searchsorted(self.row_centers_full, top, side="left")
        stop = np.searchsorted(self.row_centers_full, bottom, side="right")

        start = np.clip(start, 0, self.R)
        stop = np.clip(stop, 0, self.R)
        return start, stop

    @staticmethod
    def _reduce_cols(arr_z: np.ndarray, start_z: np.ndarray, stop_z: np.ndarray, stat: str):
        """
        arr_z: (R, C) for one z-slice
        start_z, stop_z: (C,) sampled row bounds for this z-slice
        """
        out = np.full(arr_z.shape[1], np.nan, dtype=np.float32)

        for c in range(arr_z.shape[1]):
            lo = int(start_z[c])
            hi = int(stop_z[c])
            if hi <= lo:
                continue

            vals = arr_z[lo:hi, c]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue

            if stat == "mean":
                out[c] = float(np.nanmean(vals))
            elif stat == "median":
                out[c] = float(np.nanmedian(vals))
            elif stat == "std":
                out[c] = float(np.nanstd(vals))
            else:
                raise ValueError("stat must be 'mean', 'median', or 'std'")

        return out

    def project_between(
        self,
        upper_bound: np.ndarray,   # (Z, X_full)
        lower_bound: np.ndarray,   # (Z, X_full)
        stat: str = "mean",
    ) -> np.ndarray:
        """
        Return en-face map on the compact sampled X grid: shape (Z, C).
        """
        start, stop = self._sample_bounds(upper_bound, lower_bound)

        out = np.full((self.Z, self.C), np.nan, dtype=np.float32)

        for z in range(self.Z):
            arr_z = np.asarray(self.texture_vol[z], dtype=np.float32)  # (R, C)
            out[z] = self._reduce_cols(arr_z, start[z], stop[z], stat=stat)

        return out

    def upsample_x(
        self,
        sampled_map: np.ndarray,   # (Z, C)
        full_x: int,
    ) -> np.ndarray:
        """
        Upsample sampled compact-X en-face map back to full X.
        """
        sampled_map = np.asarray(sampled_map, dtype=np.float32)
        if sampled_map.shape != (self.Z, self.C):
            raise ValueError(f"sampled_map must have shape {(self.Z, self.C)}")

        out = np.full((self.Z, full_x), np.nan, dtype=np.float32)

        for z in range(self.Z):
            good = np.isfinite(sampled_map[z])
            if good.sum() >= 2:
                out[z] = np.interp(
                    np.arange(full_x),
                    self.col_centers[good],
                    sampled_map[z, good],
                )
            elif good.sum() == 1:
                out[z, :] = sampled_map[z, good][0]

        return out


def project_texture_compact_zarr_to_enface_for_volume(
    vol_path,
    compact_zarr_path,
    flat_layers_npz,
    line_key=None,
    candidate_slabs=((5, 15),),
    features=None,
    stat="mean",
    upsample=True,
):
    """
    Project compact texture B-scan maps into en-face maps for one volume.

    Returns
    -------
    dict[str, np.ndarray]
        keys like:
            raw__mean|5->15|mean
        values:
            (Z, X_full) if upsample=True
            (Z, C)      if upsample=False
    """
    import zarr
    from code_files import file_utils

    vol_path = Path(vol_path)

    root = zarr.open_group(str(compact_zarr_path), mode="r")
    manifest_json = root.attrs.get("manifest_json", None)
    if manifest_json is None:
        raise ValueError(f"No manifest_json found in {compact_zarr_path}")

    import json
    manifest = json.loads(manifest_json)

    flat_layers = np.load(flat_layers_npz)

    if line_key is None:
        line_key = file_utils.get_algorithm_key_from_filepath(vol_path)

    center = np.asarray(flat_layers[line_key], dtype=np.float32)
    row_centers_full = np.asarray(manifest["row_centers_full"], dtype=np.int32)
    col_centers = np.asarray(manifest["col_centers"], dtype=np.int32)

    if features is None:
        features = sorted(root.array_keys())

    full_x = center.shape[1]
    out = {}

    for feat in features:
        proj = CompactTextureSlabProjector(
            texture_vol=root[feat],
            row_centers_full=row_centers_full,
            col_centers=col_centers,
        )

        for bottom_offset, top_offset in candidate_slabs:
            sampled = proj.project_between(
                upper_bound=center - top_offset,
                lower_bound=center - bottom_offset,
                stat=stat,
            )

            arr = proj.upsample_x(sampled, full_x=full_x) if upsample else sampled
            out[f"{feat}|{bottom_offset}->{top_offset}|{stat}"] = arr

    return out



# class CompactTextureSlabProjector:
#     """
#     Direct slab projection from the compact sampled-band representation.
#     Avoids materializing full-size dense volumes just to make en-face maps.
#     """
#     def __init__(
#         self,
#         texture_vol: np.ndarray,          # (Z, R, C)
#         row_centers_full: np.ndarray,     # full-image Y coords of sampled rows
#         col_centers: np.ndarray,          # full-image X coords of sampled cols
#     ):
#         arr = np.asarray(texture_vol, dtype=np.float32)
#         if arr.ndim != 3:
#             raise ValueError('texture_vol must be (Z, R, C)')

#         self.row_centers_full = np.asarray(row_centers_full, dtype=np.int32)
#         self.col_centers = np.asarray(col_centers, dtype=np.int32)

#         if arr.shape[1] != len(self.row_centers_full):
#             raise ValueError('row_centers_full length does not match texture_vol.shape[1]')
#         if arr.shape[2] != len(self.col_centers):
#             raise ValueError('col_centers length does not match texture_vol.shape[2]')

#         valid = np.isfinite(arr)
#         vals = np.where(valid, arr, 0.0)

#         self.Z, self.R, self.C = arr.shape
#         self.csum = np.empty((self.Z, self.R + 1, self.C), dtype=np.float32)
#         self.ccnt = np.empty((self.Z, self.R + 1, self.C), dtype=np.int32)

#         self.csum[:, 0, :] = 0
#         self.ccnt[:, 0, :] = 0
#         np.cumsum(vals, axis=1, out=self.csum[:, 1:, :])
#         np.cumsum(valid.astype(np.int32), axis=1, out=self.ccnt[:, 1:, :])

#     def project_between(
#         self,
#         upper_bound: np.ndarray,   # (Z, X) full-image coords
#         lower_bound: np.ndarray,   # (Z, X) full-image coords
#         interp_x: bool = True,
#         full_x: int | None = None,
#     ) -> np.ndarray:
#         upper_s = np.asarray(upper_bound, dtype=np.float32)[:, self.col_centers]
#         lower_s = np.asarray(lower_bound, dtype=np.float32)[:, self.col_centers]

#         top = np.minimum(upper_s, lower_s)
#         bottom = np.maximum(upper_s, lower_s)

#         start = np.searchsorted(self.row_centers_full, top, side='left')
#         stop = np.searchsorted(self.row_centers_full, bottom, side='right')

#         start = np.clip(start, 0, self.R)
#         stop = np.clip(stop, 0, self.R)

#         z_idx = np.arange(self.Z)[:, None]
#         c_idx = np.arange(self.C)[None, :]

#         sums = self.csum[z_idx, stop, c_idx] - self.csum[z_idx, start, c_idx]
#         cnts = self.ccnt[z_idx, stop, c_idx] - self.ccnt[z_idx, start, c_idx]

#         out_sampled = np.full((self.Z, self.C), np.nan, dtype=np.float32)
#         good = cnts > 0
#         out_sampled[good] = sums[good] / cnts[good]

#         if not interp_x:
#             return out_sampled

#         if full_x is None:
#             full_x = upper_bound.shape[1]

#         out_full = np.full((self.Z, full_x), np.nan, dtype=np.float32)
#         for z in range(self.Z):
#             g = np.isfinite(out_sampled[z])
#             if g.sum() >= 2:
#                 out_full[z] = np.interp(
#                     np.arange(full_x),
#                     self.col_centers[g],
#                     out_sampled[z, g],
#                 )
#             elif g.sum() == 1:
#                 out_full[z, :] = out_sampled[z, g][0]

#         return out_full

class TextureSlabProjector:
    def __init__(
        self,
        texture_vol,
        reference_line,
        max_top_offset,
        max_bottom_offset,
        extra_pad=0,
    ):
        import numpy as np

        ref = np.asarray(reference_line, dtype=np.float32)   # (Z, X)

        top = np.floor(ref - max_top_offset).astype(np.int32)
        bottom = np.ceil(ref - max_bottom_offset).astype(np.int32)

        y_min = int(np.nanmin(np.minimum(top, bottom))) - extra_pad
        y_max = int(np.nanmax(np.maximum(top, bottom))) + extra_pad

        full_Y = texture_vol.shape[1]
        y_min = max(0, y_min)
        y_max = min(full_Y - 1, y_max)

        arr = np.asarray(texture_vol[:, y_min:y_max + 1, :], dtype=np.float32)
        self.y0 = y_min
        self.Z, self.Y, self.X = arr.shape

        valid = np.isfinite(arr)
        vals = np.where(valid, arr, 0.0)

        vals_yzx = np.transpose(vals, (1, 0, 2))
        cnts_yzx = np.transpose(valid.astype(np.int32), (1, 0, 2))

        self.csum = np.empty((self.Y + 1, self.Z, self.X), dtype=np.float32)
        self.ccnt = np.empty((self.Y + 1, self.Z, self.X), dtype=np.int32)

        self.csum[0] = 0
        self.ccnt[0] = 0
        np.cumsum(vals_yzx, axis=0, out=self.csum[1:])
        np.cumsum(cnts_yzx, axis=0, out=self.ccnt[1:])

    def project_between(self, upper_bound, lower_bound, interp_x=True):
        import numpy as np

        top = np.floor(np.minimum(upper_bound, lower_bound)).astype(np.int32)
        bottom = np.ceil(np.maximum(upper_bound, lower_bound)).astype(np.int32)

        # move from full-image Y coords into cropped coords
        top = top - self.y0
        bottom = bottom - self.y0

        # example here: exclude the boundary rows themselves
        start = top + 1
        stop = bottom

        start = np.clip(start, 0, self.Y)
        stop = np.clip(stop, 0, self.Y)

        z_idx = np.arange(self.Z)[:, None]
        x_idx = np.arange(self.X)[None, :]

        sums = self.csum[stop, z_idx, x_idx] - self.csum[start, z_idx, x_idx]
        cnts = self.ccnt[stop, z_idx, x_idx] - self.ccnt[start, z_idx, x_idx]

        out = np.full((self.Z, self.X), np.nan, dtype=np.float32)
        good = cnts > 0
        out[good] = sums[good] / cnts[good]

        if interp_x:
            for z in range(self.Z):
                g = np.isfinite(out[z])
                if g.sum() >= 2:
                    out[z] = np.interp(np.arange(self.X), np.where(g)[0], out[z, g])

        return out
