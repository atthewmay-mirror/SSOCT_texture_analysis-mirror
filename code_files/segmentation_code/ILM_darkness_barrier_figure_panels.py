from __future__ import annotations

"""
Small, figure-specific ILM pipeline for illustrating the darkness-barrier DP cost.

This is intentionally additive and does not change the production ILM pipeline. It
reuses the existing ILM steps through the thinned DP input, then runs two otherwise
matched DP passes:

    1. no darkness barrier
    2. with the same darkness-barrier settings used by step_ilm_DP

It then exports title-free PNG panels using the same README-panel machinery as the
existing ILM panel exporter.
"""

from pathlib import Path
from typing import Optional

import numpy as np

import code_files.segmentation_code.flattening_utility_functions as fuf
import code_files.segmentation_code.segmentation_plot_utils as spu
import code_files.segmentation_code.segmentation_step_functions as ssf
import code_files.segmentation_code.segmentation_utility_functions as suf
from code_files.segmentation_code.ILM_segmentation_readme_panels import (
    _ilm_peak_suppression_debug,
    save_ilm_segmentation_readme_panels,
)
from code_files.segmentation_code.segmentation_readme_panels import (
    ReadmePanel,
    _get,
    _line_dict,
)


# -----------------------------
# DP comparison steps
# -----------------------------

def _ilm_dp_cost_from_ctx(ctx: ssf.ILMContext, *, onh_value_factor: float = 0.5) -> np.ndarray:
    """Build the ILM DP cost exactly as in step_ilm_DP, without running DP."""
    if ctx.thinline_inv_cost is None:
        raise ValueError("ctx.thinline_inv_cost is missing; run step_ilm_ax_grad_thinner first.")

    cost = 1 - ctx.thinline_inv_cost
    if ctx.ONH_region is not None:
        cost = suf.modify_cost_with_ONH_info(cost, ctx.ONH_region, ONH_value_factor=onh_value_factor)
    return cost


def _onh_kwargs(ctx: ssf.ILMContext, *, lambda_step_in_ONH_region: float = 0.001) -> dict:
    """Only pass ONH kwargs when ONH information exists."""
    if ctx.ONH_region is None:
        return {}
    return dict(
        ONH_region=ctx.ONH_region,
        lambda_step_in_ONH_region=lambda_step_in_ONH_region,
    )


def step_ilm_DP_without_darkness_barrier(ctx: ssf.ILMContext) -> ssf.ILMContext:
    """
    Run the ILM DP pass without the darkness-barrier transition penalty.

    This mirrors step_ilm_DP but intentionally omits Bcs/dbf. The output is stored
    separately so the standard barrier-enabled line can be run afterward.
    """
    lambda_step = 0.01

    cost = _ilm_dp_cost_from_ctx(ctx)
    DP_path, C = suf.run_DP_on_cost_matrix(
        cost,
        max_step=3,
        lambda_step=lambda_step,
        **_onh_kwargs(ctx),
    )

    ctx.final_DP_cost_no_darkness_barrier = cost
    ctx.final_DP_cumulative_cost_no_darkness_barrier = C
    ctx.ilm_raw_no_darkness_barrier = DP_path
    return ctx


def step_ilm_DP_with_darkness_barrier_for_fig3(ctx: ssf.ILMContext) -> ssf.ILMContext:
    """
    Run the ILM DP pass with the usual darkness-barrier settings, storing C too.

    This is functionally the same as step_ilm_DP, but keeps explicit figure-3
    attributes so the no-barrier/barrier comparison panels are self-contained.
    """
    t = 0.6
    p = 1
    dbf = 5
    lambda_step = 0.01

    cost = _ilm_dp_cost_from_ctx(ctx)
    norm_cost = suf.normalize_image_per_column(cost)
    Bcs, barrier = suf.calculate_darkness_barrier_and_Bcs(norm_cost, t=t, p=p)

    DP_path, C = suf.run_DP_on_cost_matrix(
        cost,
        max_step=3,
        lambda_step=lambda_step,
        dbf=dbf,
        Bcs=Bcs,
        **_onh_kwargs(ctx),
    )

    ctx.final_DP_cost = cost
    ctx.final_DP_darkness_barrier_img = barrier
    ctx.final_DP_cumulative_cost_with_darkness_barrier = C
    ctx.ilm_raw_with_darkness_barrier = DP_path

    # Keep compatibility with the downstream ILM upsample/unsmooth steps.
    ctx.ilm_raw = DP_path
    return ctx


# -----------------------------
# Figure-3 panel helpers
# -----------------------------

def _working_line_to_original(ctx_ilm, line) -> Optional[np.ndarray]:
    """Map a flattened/downsampled working-space ILM line back to original space."""
    if line is None:
        return None

    y = np.asarray(line, dtype=float)
    if y.ndim != 1 or y.size == 0:
        return None

    y_up = suf.upsample_path(
        y,
        vertical_factor=ctx_ilm.cfg.vertical_factor,
        original_length=ctx_ilm.cfg.original_height,
    )

    shift = _get(ctx_ilm, "hypersmoother_params.hypersmoother_shift_y_full")
    if shift is not None:
        y_up = fuf.warp_line_by_shift(y_up, shift, direction="to_orig")
    return y_up


def _peak_overlay_on_flattened_bscan(ctx_ilm):
    """Show peak detections in red on the flattened/downsampled working B-scan."""
    img = _get(ctx_ilm, "img")
    if img is None:
        return None

    peaks, _ = _ilm_peak_suppression_debug(ctx_ilm)
    if peaks is None:
        return img
    return spu.overlay_peaks_on_image(img, peaks)


def default_ilm_darkness_barrier_figure_panels() -> list[ReadmePanel]:
    """Panels for the ILM darkness-barrier comparison figure."""
    return [
        ReadmePanel(
            slug="panel_A_original_bscan",
            title="Panel A. Original B-scan",
            description="Original B-scan before ILM-specific flattening and DP processing.",
            source_functions=("ssf.ILMContext",),
            get_array=lambda ilm, rpe: _get(ilm, "original_image"),
        ),
        ReadmePanel(
            slug="panel_B_flattened_bscan_with_peaks",
            title="Panel B. Flattened B-scan with pre-suppression peaks",
            description=(
                "The hypersmoother-flattened/downsampled working B-scan with the same column-wise "
                "peaks used for ILM peak suppression overlaid in red."
            ),
            source_functions=("ssf.step_ilm_hypersmoother", "ssf.step_ilm_peak_suppression"),
            get_array=lambda ilm, rpe: _peak_overlay_on_flattened_bscan(ilm),
        ),
        ReadmePanel(
            slug="panel_C_flattened_peak_suppressed_input",
            title="Panel C. Flattened post-peak-suppressed image",
            description=(
                "Peak-suppressed ILM gradient image in flattened/downsampled working coordinates. "
                "This illustrates the cost-image source after suppressing deeper distractor peaks."
            ),
            source_functions=("ssf.step_ilm_peak_suppression",),
            get_array=lambda ilm, rpe: _get(ilm, "peak_suppressed"),
        ),
        ReadmePanel(
            slug="panel_D_original_DP_without_darkness_barrier",
            title="Panel D. DP path without darkness barrier",
            description=(
                "Original B-scan with the matched ILM DP pass run without Bcs/dbf. In attenuation or "
                "imperfect-flattening regions this path can jump to the brighter RPE-adjacent line."
            ),
            source_functions=("step_ilm_DP_without_darkness_barrier", "suf.run_DP_on_cost_matrix"),
            get_array=lambda ilm, rpe: _get(ilm, "original_image"),
            get_lines=lambda ilm, rpe: _line_dict(
                ilm_raw=_working_line_to_original(ilm, _get(ilm, "ilm_raw_no_darkness_barrier")),
            ),
        ),
        ReadmePanel(
            slug="panel_E_original_DP_with_darkness_barrier",
            title="Panel E. DP path with darkness barrier",
            description=(
                "Original B-scan with the same DP pass using the darkness-barrier transition penalty. "
                "The barrier discourages jumping through dark space to a deeper bright line."
            ),
            source_functions=("step_ilm_DP_with_darkness_barrier_for_fig3", "suf.calculate_darkness_barrier_and_Bcs"),
            get_array=lambda ilm, rpe: _get(ilm, "original_image"),
            get_lines=lambda ilm, rpe: _line_dict(
                ilm_raw=_working_line_to_original(ilm, _get(ilm, "ilm_raw_with_darkness_barrier")),
            ),
        ),
    ]


def save_ilm_darkness_barrier_figure_panels(
    ctx_ilm,
    *,
    out_dir: str | Path,
    prefix: str = "ilm_fig3",
    panels: Optional[list[ReadmePanel]] = None,
    dpi: int = 300,
    transparent: bool = True,
    write_manifest: bool = True,
) -> list[dict]:
    """Save the ILM darkness-barrier figure panels as individual PNGs."""
    return save_ilm_segmentation_readme_panels(
        ctx_ilm=ctx_ilm,
        out_dir=out_dir,
        prefix=prefix,
        panels=panels or default_ilm_darkness_barrier_figure_panels(),
        dpi=dpi,
        transparent=transparent,
        write_manifest=write_manifest,
    )


def step_ilm_save_darkness_barrier_figure_panels(ctx_ilm: ssf.ILMContext) -> ssf.ILMContext:
    """Optional endpoint step for saving the figure-3 PNG panels from a pipeline."""
    out_dir = ctx_ilm.kwargs.get(
        "darkness_barrier_figure_out_dir",
        "/Volumes/T9/iowa_research/Han_AIR_Dec_2025/results/ILM_darkness_barrier_panels/",
    )
    prefix = ctx_ilm.kwargs.get(
        "darkness_barrier_figure_prefix",
        f"ilm_fig3_{ctx_ilm.ID if ctx_ilm.ID is not None else ctx_ilm.idx}",
    )
    save_ilm_darkness_barrier_figure_panels(
        ctx_ilm,
        out_dir=out_dir,
        prefix=prefix,
    )
    return ctx_ilm


# -----------------------------
# Figure-specific pipelines
# -----------------------------

ILM_DARKNESS_BARRIER_FIGURE_STEPS: list[ssf.ILMStepFn] = [
    ssf.step_ilm_hypersmoother,
    ssf.step_ilm_downsample_and_preprocess,
    ssf.step_ilm_compute_enhancement,
    ssf.step_ilm_peak_suppression,
    ssf.step_ilm_ax_grad_thinner,
    step_ilm_DP_without_darkness_barrier,
    step_ilm_DP_with_darkness_barrier_for_fig3,
    ssf.step_ilm_upsample,
    ssf.step_ilm_unsmooth,
]

ILM_DARKNESS_BARRIER_FIGURE_STEPS_WITH_SAVE: list[ssf.ILMStepFn] = (
    ILM_DARKNESS_BARRIER_FIGURE_STEPS + [step_ilm_save_darkness_barrier_figure_panels]
)


def process_ilm_darkness_barrier_figure_bscan(
    idx_and_img,
    *,
    save_panels: bool = True,
    out_dir: str | Path | None = None,
):
    """
    Convenience runner for one B-scan tuple.

    Accepts tuples shaped like either:
        (idx, bscan, ONH_info)
        (idx, bscan, ONH_info, work_id)
    """
    idx, bscan, ONH_info, *rest = idx_and_img
    work_id = rest[0] if rest else idx

    kwargs = {}
    if out_dir is not None:
        kwargs["darkness_barrier_figure_out_dir"] = str(out_dir)

    ctx = ssf.ILMContext(
        idx=idx,
        ID=work_id,
        original_image=bscan.copy(),
        ONH_region=ONH_info,
        cfg=ssf.ILMConfig(),
        kwargs=kwargs,
    )

    steps = ILM_DARKNESS_BARRIER_FIGURE_STEPS_WITH_SAVE if save_panels else ILM_DARKNESS_BARRIER_FIGURE_STEPS
    return ssf.run_pipeline(ctx, steps=steps)
