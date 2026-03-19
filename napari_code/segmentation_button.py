
from __future__ import annotations
import os
import time
import tempfile
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import napari
from qtpy.QtWidgets import QWidget, QPushButton, QVBoxLayout, QMessageBox

# --- CONFIG ---
# Absolute path to the *external* Python interpreter (other conda env)
EXTERNAL_PY = "/Users/matthewhunt/miniconda3/envs/han_air/bin/python"


# Path to the external runner script (relative to THIS file)
RUNNER_SCRIPT = (Path(__file__).resolve().parents[1] / "code_files" / "segmentation_code" / "seg_runner_new.py")

# Where the final multipage PDF should land by default
DEFAULT_REPORT_DIR = (Path(__file__).resolve().parents[1] / "reports")  # repo_root/reports

# Optional: directory of mini volumes for the external runner (can override in UI call)
DEFAULT_MINI_ROOT = (Path(__file__).resolve().parents[1] / "data_all_volumes_extra_mini")  # <<< EDIT ME


def _active_image_layer(viewer: napari.Viewer):
    layer = viewer.layers.selection.active
    if layer is None:
        print("hey, the layer is none, check this out")
    if layer is None or not isinstance(layer, napari.layers.Image):
        # fall back to first Image layer
        for L in viewer.layers:
            if isinstance(L, napari.layers.Image):
                return L
        return None
    return layer


def _current_z_index(viewer: napari.Viewer, layer) -> Optional[int]:
    """Return the current slice index for a 3D (Z,Y,X) image layer.
    Note: napari Layer objects don't expose `.viewer`; use the passed-in viewer.
    """
    data = layer.data
    if getattr(data, 'ndim', 0) == 4:
        z = int(viewer.dims.current_step[1])
    elif getattr(data, 'ndim', 0) == 3:
        z = int(viewer.dims.current_step[0])
    elif getattr(data, 'ndim', 0) != 3:
        return None
    # Assume first axis is Z for (Z, Y, X) data
    print(f"using z slice at {z}")
    return z


def _save_current_slice(layer, z_index: int) -> Path:
    """Realize the currently shown 2D slice to a temp .npy, returning its path."""
    if layer.data.ndim == 4: # Ignore the leading flattened channel
        data_to_use = layer.data[0,:,:,:]
    else:
        data_to_use = layer.data
    arr2d = np.asarray(data_to_use[z_index])  # triggers dask compute for one slice if dask-backed
    arr2d = np.ascontiguousarray(arr2d)
    tmpdir = Path(tempfile.gettempdir())
    tname = f"napari_slice_{int(time.time())}_{os.getpid()}.npy"
    out = tmpdir / tname
    np.save(out, arr2d)
    return out


def _build_subprocess_cmd(
    current_bscan_npy: Path,
    out_pdf: Path,
    img_src_path,
    z_index,
    mini_root: Optional[Path] = None,
    k_samples: int = 5,
    downsample: float = 1.5,
    max_workers: int = 8,
    debug: bool = False,   # <-- ADD
    run_extra_slices: bool = False,
    z_stride: int = 1,
    RPE_OR_ILM: str = "RPE",
) -> list[str]:
    mini_root = Path(mini_root) if mini_root is not None else DEFAULT_MINI_ROOT
    cmd = [
        str(EXTERNAL_PY), str(RUNNER_SCRIPT),
        "--current-bscan", str(current_bscan_npy),
        "--out", str(out_pdf),
        "--mini-root", str(mini_root),
        "--k", str(k_samples),
        "--downsample", str(downsample),
        "--max-workers", str(max_workers),
        "--img_src_path",str(img_src_path),
        "--z_index",str(z_index),
        "--z_stride",str(z_stride),
        "--RPE_OR_ILM",str(RPE_OR_ILM),
    ]


    if debug:
        # Run under pdb and force debug mode in the script
        cmd = [cmd[0]] +  ["-m", "pdb", "-c", "continue", '--'] + cmd[1:]
        cmd += [ "--debug"]
    if run_extra_slices:
        cmd += [ "--run_extra_slices"]

    return cmd


import traceback

def safe_breakpoint():
    # Prints the stack then drops you into pdb from the console thread
    try:
        raise RuntimeError("debug breakpoint")
    except RuntimeError:
        traceback.print_exc()
    import pdb; pdb.set_trace()

# inside your callback instead of pdb.set_trace():

class SegmentationButton(QWidget):
    """Small QWidget with one button to launch the external runner."""
    def __init__(self, viewer: napari.Viewer, mini_root: Optional[Path] = None,
                 debug_mode = False,
                run_extra_slices = False,
                 title = "Run ILM+RPE PDF on Current Slice",
                 z_stride=1,
                 RPE_OR_ILM="RPE",):  # flip to True when you want pdb
        super().__init__()
        self.viewer = viewer
        self.mini_root = Path(mini_root) if mini_root else DEFAULT_MINI_ROOT
        self.debug_mode = debug_mode
        self.run_extra_slices = run_extra_slices

        self.button = QPushButton(title)
        self.button.clicked.connect(self._on_click)

        layout = QVBoxLayout()
        layout.addWidget(self.button)
        self.setLayout(layout)
        self.z_stride = z_stride
        self.RPE_OR_ILM = RPE_OR_ILM 

    def _on_click(self):
        try:
            layer = _active_image_layer(self.viewer)
            if layer is None:
                QMessageBox.warning(self, "No image layer", "Select an image layer.")
                return
            if getattr(layer.data, 'ndim', 0) not in [3,4]:
                QMessageBox.warning(self, "Not 3D/4D", f"Expected a 3D or 4D ([ch],Z,Y,X) image, and yours is {layer.data.shape}")
                return

            # z = _current_z_index(layer)
            z = _current_z_index(self.viewer, layer)

            if z is None:
                QMessageBox.warning(self, "No Z index", "Could not determine current slice.")
                return


            # from IPython import embed
            # embed(colors="neutral")
            if "src_path" in layer.metadata:
                img_path = Path(layer.metadata["src_path"])

            # Fallback: if layer was loaded via a napari reader plugin, it may have .source.path
            elif getattr(layer, "source", None) and getattr(layer.source, "path", None):
                img_path = Path(layer.source.path)
                print("unable to get the src_path directly from the Path(layer.metadata")
            else:
                QMessageBox.warning(self, "Missing source path",
                                    "This layer has no recorded file path (metadata['src_path']).")
                # return
            # Realize and save the current slice to a temp .npy
            npy_path = _save_current_slice(layer, z)

            # Make output path
            DEFAULT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            out_pdf = DEFAULT_REPORT_DIR / 'seg_reports' / f"seg_report_{timestamp}.pdf"
            print(f"will save output pdf at {out_pdf}")

            # Build and launch external process

            # Build and launch external process
            cmd = _build_subprocess_cmd(
                current_bscan_npy=npy_path,
                out_pdf=out_pdf,
                mini_root=self.mini_root,
                k_samples=5,
                downsample=1.5,
                max_workers=8,
                debug=self.debug_mode,
                run_extra_slices = self.run_extra_slices,
                img_src_path=img_path,
                z_index = z,
                z_stride = self.z_stride,
                RPE_OR_ILM=self.RPE_OR_ILM

            )

            if self.debug_mode:
                # Foreground: napari UI will block; use the terminal you launched napari from
                subprocess.run(cmd, check=False)
            else:
                # Non-blocking normal run
                subprocess.Popen(cmd)
                self.viewer.status = f"Launched external segmentation job → {out_pdf.name}"


            # # Non-blocking launch; we don't wait or capture stdout (no UI output desired)
            # subprocess.Popen(cmd)
            # self.viewer.status = f"Launched external segmentation job → {out_pdf.name}"
        except Exception as e:
            # QMessageBox.critical(self, "Launch error", str(e))
            tb = traceback.format_exc()
            print(tb)  # goes to terminal if you launched napari from Terminal
            QMessageBox.critical(self, "Launch error", tb)
            # (optional) re-raise during development so napari debug tools catch it
            # raise


def add_segmentation_button(viewer: napari.Viewer,z_stride=1) -> SegmentationButton:
    """Factory function you can call from your napari plugin to add the button."""
    mini_root="/Users/matthewhunt/Research/Iowa_Research/Han_AIR/data_all_volumes/"
    widget = SegmentationButton(viewer,mini_root=mini_root,debug_mode=False,title="Run RPE Layers on Current Slice",z_stride=z_stride)
    viewer.window.add_dock_widget(widget, area='right', name='ILM+RPE PDF')

    widget = SegmentationButton(viewer,mini_root=mini_root,debug_mode=False,title="Run ILM Layers on Current Slice",z_stride=z_stride,RPE_OR_ILM="ILM")
    viewer.window.add_dock_widget(widget, area='right', name='ILM PDF')


    widget = SegmentationButton(viewer,mini_root=mini_root,debug_mode=True,title="Run RPE Layers on Current Slice DEBUG",z_stride=z_stride)
    viewer.window.add_dock_widget(widget, area='right', name='RPE PDF DEBUG')

    widget = SegmentationButton(viewer,mini_root=mini_root,debug_mode=True,title="Run ILM Layers on Current Slice DEBUG",z_stride=z_stride,RPE_OR_ILM="ILM")
    viewer.window.add_dock_widget(widget, area='right', name='ILM PDF DEBUG')

    widget = SegmentationButton(viewer,mini_root=mini_root,debug_mode=False,title="Run Layers on Current+Extra Slices",run_extra_slices=True,z_stride=z_stride)
    viewer.window.add_dock_widget(widget, area='right', name='ILM+RPE PDF Extra Slices')

    widget = SegmentationButton(viewer,mini_root=mini_root,debug_mode=True,title="Run Layers on Current+Extra Slices DEBUG",run_extra_slices=True,z_stride=z_stride)
    viewer.window.add_dock_widget(widget, area='right', name='ILM+RPE PDF Extra Slices DEBUG')
    return widget