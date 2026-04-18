#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import napari
import numpy as np
from dask import array as da
from dask.cache import Cache

sys.path.append(str(Path(__file__).resolve().parents[1]))  # adds repo root to path

import code_files.file_utils as fu
from code_files.zarr_file_utils import ensure_image_zarr


# Multithreading hack for older python / napari launches
main = sys.modules["__main__"]
if not hasattr(main, "__spec__"):
    main.__spec__ = importlib.util.spec_from_loader("__main__", loader=None)


def parse_args():
    p = argparse.ArgumentParser(
        description="Minimal raw OCT volume viewer using cached Zarr images only."
    )
    p.add_argument(
        "--vol_dir",
        type=Path,
        default=None,
        help="Directory containing OCT volumes.",
    )
    p.add_argument(
        "--glob",
        type=str,
        default=None,
        help="Glob pattern inside --vol_dir to find volumes.",
    )
    p.add_argument(
        "--cube_numbers",
        type=str,
        default=None,
        help="Comma-separated cube numbers to include, e.g. 1,3,10",
    )
    p.add_argument(
        "--z_stride",
        type=int,
        default=1,
        help="Display every nth B-scan while reusing full-Z cached raw Zarr.",
    )
    p.add_argument(
        "--overwrite_files",
        action="store_true",
        help="Rebuild cached raw image Zarr files.",
    )
    p.add_argument(
        "--use_skip_yaml",
        action="store_true",
        help="Apply your existing skip-yaml filtering from file_utils.get_all_vol_paths.",
    )
    p.add_argument(
        "--scale_y",
        type=float,
        default=1 / 3,
        help="Napari display scaling for the Y axis.",
    )
    return p.parse_args()


def load_one_volume(vp: Path, *, z_stride: int, overwrite: bool):
    """Return (img, name) for one raw OCT volume, using/reusing a cached Zarr."""
    image_zarr = ensure_image_zarr(
        vp,
        z_stride=1,   # cache full-Z raw volume once; thin only at display time
        overwrite=overwrite,
    )

    img = da.from_zarr(str(image_zarr))
    img = img[::z_stride]
    img = img.rechunk((1, img.shape[-2], img.shape[-1]))
    return img, vp.stem


def main():
    args = parse_args()

    all_vol_paths = fu.get_all_vol_paths(
        args.vol_dir,
        glob=args.glob,
        cube_numbers=args.cube_numbers,
        use_skip_yaml=args.use_skip_yaml,
    )
    print(f"[pager] Found {len(all_vol_paths)} volumes")

    import dask

    dask.config.set(scheduler="threads", num_workers=min(6, (os.cpu_count() or 6)))
    dask_cache = Cache(1 * 1024**3)
    dask_cache.register()

    viewer = napari.Viewer(ndisplay=2)

    state = {
        "idx": 0,
        "img_layer": None,
        "name": None,
    }

    def _add_current_volume():
        vp = all_vol_paths[state["idx"]]
        img, name = load_one_volume(
            vp,
            z_stride=args.z_stride,
            overwrite=args.overwrite_files,
        )

        state["name"] = name

        viewer.dims.axis_labels = ("z", "y", "x")
        viewer.dims.order = (0, 1, 2)
        scale = (1, args.scale_y, 1)

        cur = list(viewer.dims.current_step)
        cur[0] = img.shape[0] // 2
        viewer.dims.current_step = tuple(cur)

        clims = (
            (0, int(np.iinfo(img.dtype).max))
            if np.issubdtype(img.dtype, np.integer)
            else (0.0, 1.0)
        )

        if state["img_layer"] is None:
            state["img_layer"] = viewer.add_image(
                img,
                name="OCT_volume",
                colormap="gray",
                blending="additive",
                rendering="translucent",
                contrast_limits=clims,
                scale=scale,
                metadata={"src_path": str(vp)},
            )
        else:
            state["img_layer"].data = img
            state["img_layer"].scale = scale
            state["img_layer"].metadata["src_path"] = str(vp)
            state["img_layer"].contrast_limits = clims

        viewer.status = f"[{state['idx'] + 1}/{len(all_vol_paths)}] {state['name']}"
        print("order:", viewer.dims.order, "displayed:", viewer.dims.displayed, "labels:", viewer.dims.axis_labels)

    def _next_volume(v):
        if state["idx"] + 1 >= len(all_vol_paths):
            v.status = "Already at last volume."
            return
        state["idx"] += 1
        _add_current_volume()

    def _prev_volume(v):
        if state["idx"] == 0:
            v.status = "Already at first volume."
            return
        state["idx"] -= 1
        _add_current_volume()

    _add_current_volume()

    viewer.bind_key("Ctrl-]", _next_volume, overwrite=True)
    viewer.bind_key("Ctrl-[", _prev_volume, overwrite=True)

    print("now running raw viewer (pagination: Ctrl-[ prev, Ctrl-] next)")
    napari.run()


if __name__ == "__main__":
    main()
