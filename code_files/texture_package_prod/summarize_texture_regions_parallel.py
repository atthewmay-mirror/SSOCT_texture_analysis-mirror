from __future__ import annotations

import argparse
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from code_files.texture_package_prod.texture_enface_utils import make_enface_isotropic_x
from code_files.texture_package_prod.texture_regions import (
    canonical_case_id,
    summarize_by_regions,
    build_region_value_map,
    prepare_case_for_etdrs,
    save_overlay_mosaic_pdf,
)
import file_utils as fu


# ---------------------------------------------------------------------
# Repo imports
# ---------------------------------------------------------------------

def _find_repo_root(start: Path) -> Path:
    for cand in [start, *start.parents]:
        if (cand / "code_files").exists():
            return cand
    return start


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

try:
    from code_files.texture_package_prod.texture_regions import (
        make_etdrs_grid_plus_rings,
        summarize_by_regions,
    )
except Exception as e:
    raise ImportError(
        "Could not import texture_regions.py from the repo. "
        "Run this script from inside the repo or edit REPO_ROOT/sys.path."
    ) from e


# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------

def discover_cases(enface_root: Path, glob_pat: str) -> list[tuple[str, Path]]:
    cases: list[tuple[str, Path]] = []

    for p in sorted(enface_root.glob(glob_pat)):
        if p.is_file() and p.suffix == ".npz":
            cases.append((canonical_case_id(p.stem), p))
    if cases:
        return cases

    for d in sorted(enface_root.iterdir()):
        if not d.is_dir():
            continue
        npzs = sorted(d.glob(glob_pat))
        npzs = [p for p in npzs if p.is_file() and p.suffix == ".npz"]
        if npzs:
            if len(npzs) != 1:
                print(f"[{d.name}] expected 1 npz, found {len(npzs)}; using first")
            cases.append((canonical_case_id(d.name), npzs[0]))

    if not cases:
        raise FileNotFoundError(f"No cases found under {enface_root}")
    return cases


# ---------------------------------------------------------------------
# Annotation / mask helpers
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Rigid geometry helpers
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Region helpers
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def save_summary_plots(
    out_dir: Path,
    case_id: str,
    feature_name: str,
    feature_image: np.ndarray,
    masks: dict[str, np.ndarray],
    region_means: dict[str, float],
    exclusion_mask: np.ndarray,
):
    ring_order = ["center", "inner_ring", "outer_ring"]
    extra_ring_names = sorted(k for k in masks if k.startswith("extra_ring_") and k.count("_") == 2)
    ring_order.extend(extra_ring_names)
    ring_order.extend(["outer_region", "whole"])

    ring_vals = [region_means.get(k, np.nan) for k in ring_order]
    finite = np.array([v for v in region_means.values() if np.isfinite(v)], dtype=np.float32)

    if finite.size:
        vmin = float(np.nanpercentile(finite, 2))
        vmax = float(np.nanpercentile(finite, 98))
        if vmax <= vmin:
            vmax = vmin + 1e-6
    else:
        vmin, vmax = 0.0, 1.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=180)

    bg = _normalize_for_display(feature_image)
    axes[0].imshow(bg, cmap="gray", alpha=0.28)
    region_map = build_region_value_map(masks, region_means)
    show = np.ma.masked_invalid(region_map)
    im = axes[0].imshow(show, cmap="viridis", vmin=vmin, vmax=vmax, alpha=0.95)

    if exclusion_mask is not None and exclusion_mask.any():
        rgba = np.zeros((*exclusion_mask.shape, 4), dtype=np.float32)
        rgba[..., 0] = 1.0
        rgba[..., 3] = 0.16 * exclusion_mask.astype(np.float32)
        axes[0].imshow(rgba)

    axes[0].axis("off")
    axes[0].set_title(f"{feature_name}: ETDRS region heatmap", fontsize=9)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].bar(np.arange(len(ring_order)), ring_vals)
    axes[1].set_xticks(np.arange(len(ring_order)))
    axes[1].set_xticklabels(ring_order, rotation=45, ha="right")
    axes[1].set_title(f"{feature_name}: whole-ring means")
    axes[1].set_ylabel("mean")

    fig.suptitle(case_id)
    fig.tight_layout()
    fig.savefig(out_dir / f"{case_id}__{feature_name}__summary.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------
def split_int_id_and_eye(case_id: str) -> tuple[str, str]:
    int_id = fu.get_integer_id(case_id)
    eye = fu.get_eye(case_id)
    return int_id, eye


# def _burden_region_lists(regions: list[str]) -> tuple[list[str], list[str]]:
#     coarse = []
#     fine = []

#     for region in regions:
#         if region == "whole":
#             continue

#         if region == "center":
#             coarse.append(region)
#             fine.append(region)
#             continue

#         if region in {"inner_ring", "outer_ring", "outer_region"}:
#             coarse.append(region)
#             continue

#         if re.fullmatch(r"extra_ring_\d+", region):
#             coarse.append(region)
#             continue

#         if re.fullmatch(r"(inner|outer|outer_region)_(temporal|superior|nasal|inferior)", region):
#             fine.append(region)
#             continue

#         if re.fullmatch(r"extra_ring_\d+_(temporal|superior|nasal|inferior)", region):
#             fine.append(region)
#             continue

#     return coarse, fine


# def _d_mean_and_rms(left_vals, right_vals) -> tuple[float, float]:
#     left = np.asarray(left_vals, dtype=np.float32)
#     right = np.asarray(right_vals, dtype=np.float32)

#     good = np.isfinite(left) & np.isfinite(right)
#     if not np.any(good):
#         return np.nan, np.nan

#     d = np.abs(left[good] - right[good])
#     return float(np.mean(d)), float(np.sqrt(np.mean(d ** 2)))


# def add_intereye_burden_rows(summary_long: pd.DataFrame) -> pd.DataFrame:
#     need = {"case_id", "feature", "region", "stat", "value"}
#     missing = need - set(summary_long.columns)
#     if missing:
#         raise ValueError(f"summary_long missing columns: {sorted(missing)}")

#     base = summary_long.loc[summary_long["stat"] == "mean"].copy()

#     case_meta = base[["case_id"]].drop_duplicates().copy()
#     case_meta[["int_id", "eye"]] = case_meta["case_id"].apply(
#         lambda s: pd.Series(split_int_id_and_eye(s))
#     )

#     base = base.merge(case_meta, on="case_id", how="left")

#     new_rows = []

#     for int_id, pair_df in base.groupby("int_id", sort=False):
#         print(f"adding the intereye for {int_id}")
#         eyes = pair_df[["case_id", "eye"]].drop_duplicates()

#         if set(eyes["eye"]) != {"OD", "OS"}:
#             continue
#         if (eyes["eye"] == "OD").sum() != 1 or (eyes["eye"] == "OS").sum() != 1:
#             continue

#         od_case = eyes.loc[eyes["eye"] == "OD", "case_id"].iloc[0]
#         os_case = eyes.loc[eyes["eye"] == "OS", "case_id"].iloc[0]

#         for feature, feat_df in pair_df.groupby("feature", sort=False):
#             mat = feat_df.pivot_table(
#                 index="eye",
#                 columns="region",
#                 values="value",
#                 aggfunc="first",
#             )

#             if "OD" not in mat.index or "OS" not in mat.index:
#                 continue

#             coarse_regions, fine_regions = _burden_region_lists(list(mat.columns))

#             for region_group_name, region_list in (
#                 ("intereye_coarse", coarse_regions),
#                 ("intereye_fine", fine_regions),
#             ):
#                 if not region_list:
#                     continue

#                 d_mean, d_rms = _d_mean_and_rms(
#                     mat.loc["OD", region_list].to_numpy(),
#                     mat.loc["OS", region_list].to_numpy(),
#                 )

#                 for case_id in (od_case, os_case):
#                     new_rows.append(
#                         {
#                             "case_id": case_id,
#                             "feature": feature,
#                             "region": region_group_name,
#                             "stat": "D_mean",
#                             "value": d_mean,
#                         }
#                     )
#                     new_rows.append(
#                         {
#                             "case_id": case_id,
#                             "feature": feature,
#                             "region": region_group_name,
#                             "stat": "D_rms",
#                             "value": d_rms,
#                         }
#                     )

#     if not new_rows:
#         return summary_long

#     return pd.concat([summary_long, pd.DataFrame(new_rows)], ignore_index=True)

def process_case(
    case_id: str,
    enface_path: Path,
    annotation_root: Path,
    out_dir: Path,
    onh_label: int,
    fovea_label: int,
    radii: tuple[float, float, float],
    extra_radii: tuple[float, ...],
    show_features: list[str] | None,
    max_panels_per_page: int = 60,
    save_mosaic: bool = False,
    save_summaries: bool = False,
) -> pd.DataFrame:
    prepared = prepare_case_for_etdrs(
        case_id=case_id,
        enface_path=enface_path,
        annotation_root=annotation_root,
        onh_label=onh_label,
        fovea_label=fovea_label,
        radii=radii,
        extra_radii=extra_radii,
    )

    maps = prepared["maps"]
    masks = prepared["masks"]
    exclusion_std = prepared["exclusion_mask"]
    fovea_std_xy = prepared["fovea_xy"]

    if show_features is None:
        show_features = list(maps)[:6]
    else:
        show_features = [f for f in show_features if f in maps]

    overlay_maps = {}
    summary_payloads = {}
    rows = []

    for feature_name, arr_std in maps.items():
        region_means_full = summarize_by_regions(arr_std, masks, stats=("mean",))
        region_means = {k.replace("__mean", ""): v for k, v in region_means_full.items()}

        for region_name, value in region_means.items():
            rows.append(
                {
                    "case_id": prepared["case_id"],
                    "feature": feature_name,
                    "region": region_name,
                    "stat": "mean",
                    "value": value,
                }
            )

        if feature_name in show_features:
            if save_mosaic:
                overlay_maps[feature_name] = arr_std
            if save_summaries:
                summary_payloads[feature_name] = (arr_std, region_means)

    if save_mosaic and overlay_maps:
        overlay_dir = out_dir / "overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        save_overlay_mosaic_pdf(
            overlay_dir / f"{prepared['case_id']}__overlay.pdf",
            case_id=prepared["case_id"],
            transformed_maps=overlay_maps,
            show_features=show_features,
            fovea_xy=fovea_std_xy,
            radii=radii,
            extra_radii=extra_radii,
            exclusion_mask=exclusion_std,
            max_panels_per_page=max_panels_per_page,
        )

    if save_summaries and summary_payloads:
        summary_plot_dir = out_dir / "summary_plots"
        summary_plot_dir.mkdir(parents=True, exist_ok=True)
        for feature_name, (feature_image, region_means) in summary_payloads.items():
            save_summary_plots(
                summary_plot_dir,
                case_id=prepared["case_id"],
                feature_name=feature_name,
                feature_image=feature_image,
                masks=masks,
                region_means=region_means,
                exclusion_mask=exclusion_std,
            )

    return pd.DataFrame(rows)




def _run_one_case(task: dict) -> pd.DataFrame:
    return process_case(**task)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--enface-root", type=Path, required=True)
    ap.add_argument("--annotation-root", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--glob", default="*.npz", help="Case discovery glob under --enface-root")
    ap.add_argument("--onh-label", type=int, default=1)
    ap.add_argument("--fovea-label", type=int, default=2)
    ap.add_argument("--max-panels-per-page", type=int, default=60)

    ap.add_argument(
        "--radii",
        type=float,
        nargs=3,
        default=(45, 135, 270),
        help="center inner outer radii in enface pixels",
    )
    ap.add_argument(
        "--extra-radii",
        type=float,
        nargs="*",
        default=(470,),
        help="extra outer ring radii in enface pixels",
    )
    ap.add_argument(
        "--show-features",
        nargs="*",
        default=None,
        help="Feature names to overlay / plot. Default: first six found in each case.",
    )
    ap.add_argument("--save_mosaic", action="store_true")
    ap.add_argument("--save_summaries", action="store_true")
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--fail_fast", action="store_true")

    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    cases = discover_cases(args.enface_root, args.glob)
    print(f"found {len(cases)} cases")

    tasks = [
        dict(
            case_id=case_id,
            enface_path=enface_path,
            annotation_root=args.annotation_root,
            out_dir=args.outdir,
            onh_label=args.onh_label,
            fovea_label=args.fovea_label,
            radii=tuple(args.radii),
            extra_radii=tuple(args.extra_radii),
            show_features=args.show_features,
            max_panels_per_page=args.max_panels_per_page,
            save_mosaic=args.save_mosaic,
            save_summaries=args.save_summaries,
        )
        for case_id, enface_path in cases
    ]

    all_df: list[pd.DataFrame] = []
    failures: list[tuple[str, str]] = []

    if args.n_jobs <= 1:
        for task in tasks:
            case_id = task["case_id"]
            print(f"starting {case_id}")
            try:
                df = _run_one_case(task)
                all_df.append(df)
                print(f"finished {case_id}")
            except Exception as e:
                if args.fail_fast:
                    raise
                failures.append((case_id, repr(e)))
                print(f"FAILED {case_id}: {e}")
    else:
        max_workers = min(args.n_jobs, len(tasks))
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            future_to_case = {ex.submit(_run_one_case, task): task["case_id"] for task in tasks}
            for fut in as_completed(future_to_case):
                case_id = future_to_case[fut]
                try:
                    df = fut.result()
                    all_df.append(df)
                    print(f"finished {case_id}")
                except Exception as e:
                    if args.fail_fast:
                        raise
                    failures.append((case_id, repr(e)))
                    print(f"FAILED {case_id}: {e}")

    if not all_df:
        raise RuntimeError("No cases completed successfully")

    summary_long = pd.concat(all_df, ignore_index=True)
    summary_long.to_csv(args.outdir / "texture_region_summary_long.csv", index=False)

    summary_wide = summary_long.pivot_table(
        index="case_id",
        columns=["feature", "region", "stat"],
        values="value",
    )
    summary_wide.to_csv(args.outdir / "texture_region_summary_wide.csv")

    if failures:
        fail_df = pd.DataFrame(failures, columns=["case_id", "error"])
        fail_df.to_csv(args.outdir / "failed_cases.csv", index=False)
        print(f"wrote failures for {len(failures)} cases")


if __name__ == "__main__":
    main()
