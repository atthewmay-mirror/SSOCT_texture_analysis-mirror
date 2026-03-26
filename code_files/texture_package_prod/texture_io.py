#reviewed

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import file_utils as _external_file_utils
except Exception:
    _external_file_utils = None


DEFAULT_SS_SHAPE = (1024, 1536, 512)


def load_image(path: str | Path) -> np.ndarray:
    """Load a 2D/3D image with Pillow and return numpy."""
    return np.asarray(Image.open(path))


def load_ss_volume(
    path: str | Path,
    shape: tuple[int, int, int] = DEFAULT_SS_SHAPE,
    mmap: bool = True,
    z_step: int = 1,
    y_step: int = 1,
    x_step: int = 1,
) -> np.ndarray:
    """Load a SmartScan-style volume. Uses local file_utils if available, otherwise a self-contained fallback."""
    path = Path(path)
    if _external_file_utils is not None and hasattr(_external_file_utils, 'load_ss_volume2'):
        try:
            return _external_file_utils.load_ss_volume2(
                path,
                z_step=z_step,
                y_step=y_step,
                x_step=x_step,
                mmap=mmap,
            )
        except Exception:
            pass

    if path.suffix == '.npy':
        vol = np.load(path, mmap_mode='r' if mmap else None)
    else:
        if mmap:
            arr = np.memmap(path, dtype=np.uint16, mode='r', shape=shape)
        else:
            arr = np.fromfile(path, dtype=np.uint16).reshape(shape)
        vol = np.rot90(arr, k=2, axes=(1, 2))
    return vol[::z_step, ::y_step, ::x_step]


def load_layers_npz(vol_path: str | Path,layers_root:str | Path) -> dict[str, np.ndarray]:
    """Load a layers npz into a plain dict.. Shoudl refactor into my main file utils"""
    layer_path = _external_file_utils.new_get_corresponding_layer_path(Path(vol_path),Path(layers_root))
    obj = np.load(layer_path)
    return {k: obj[k] for k in obj.files}


# def load_landmark_dict(path: str | Path) -> dict[str, dict[str, tuple[float, float]]]:
#     """Parse the project dictionary file into {'name': {'onh_xy': ..., 'fovea_xy': ...}}."""
#     text = Path(path).read_text()
#     tree = ast.parse(text, mode='exec')
#     for node in tree.body:
#         if isinstance(node, ast.Assign):
#             for target in node.targets:
#                 if isinstance(target, ast.Name) and target.id == 'location_dict':
#                     raw = ast.literal_eval(node.value)
#                     out = {}
#                     for name, (onh_xy, fovea_xy) in raw.items():
#                         out[name] = {
#                             'onh_xy': tuple(map(float, onh_xy)),
#                             'fovea_xy': tuple(map(float, fovea_xy)),
#                         }
#                     return out
#     raise ValueError(f'Could not find location_dict in {path}')


def infer_eye_from_onh_fovea(onh_xy: tuple[float, float], fovea_xy: tuple[float, float]) -> str:
    """Infer eye laterality from whether ONH lies nasal to the fovea in image space."""
    return 'R' if onh_xy[0] > fovea_xy[0] else 'L'


def to_gray(image: np.ndarray, prefer_green: bool = False) -> np.ndarray:
    """Simple grayscale conversion."""
    image = np.asarray(image)
    if image.ndim == 2:
        return image.astype(np.float32)
    if image.ndim != 3:
        raise ValueError('Expected HxW or HxWxC image')
    if prefer_green and image.shape[2] >= 3:
        return image[..., 1].astype(np.float32)
    return image[..., :3].astype(np.float32).mean(axis=2)
