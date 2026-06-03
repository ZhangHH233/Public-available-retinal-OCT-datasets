# preprocess_chiu_dme_mat_instance_label.py
# -*- coding: utf-8 -*-

"""
Preprocess DUKE-BOE / Chiu DME OCT .mat dataset.

Main functions:
1. Read .mat files under input_root.
2. Extract OCT volume, manualFluid1, manualFluid2, and retinal layer annotations.
3. Save each B-scan as image PNG.
4. Save manualFluid1 and manualFluid2 as per-slice label PNG,
   while preserving original instance IDs.
5. Save overlay visualizations only for slices containing lesions.
6. Generate metadata.csv and volume-level train/val split.

Recommended usage:

python preprocess_chiu_dme_mat_instance_label.py ^
  --input_root "F:\1_项目文档\2026_instaneSeg\BOE_ Chiu_DME_OCT_datasets\Segmentation of OCT images (DME)_datasets" ^
  --output_root "F:\1_项目文档\2026_instaneSeg\BOE_ Chiu_DME_OCT_datasets\preprocessed_instance" ^
  --save_overlay

First inspect keys:
python preprocess_chiu_dme_mat_instance_label.py --input_root "..." --inspect
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import scipy.io as sio
import imageio.v3 as iio

try:
    import h5py
except ImportError:
    h5py = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# -----------------------------
# Candidate variable names
# -----------------------------

IMAGE_KEYS = [
    "images", "Images", "image", "Image", "I", "img", "imgs", "volume", "Volume"
]

FLUID_KEYS_MANUAL1 = [
    "manualFluid1", "ManualFluid1", "fluid1", "Fluid1", "manual_fluid1"
]

FLUID_KEYS_MANUAL2 = [
    "manualFluid2", "ManualFluid2", "fluid2", "Fluid2", "manual_fluid2"
]

LAYER_KEYS_MANUAL1 = [
    "manualLayers1", "ManualLayers1", "layers1", "Layers1", "manual_layers1"
]

LAYER_KEYS_MANUAL2 = [
    "manualLayers2", "ManualLayers2", "layers2", "Layers2", "manual_layers2"
]

LAYER_KEYS_GENERIC = [
    "manualLayers", "ManualLayers", "layers", "Layers",
    "automaticLayers", "AutomaticLayers", "autoLayers",
    "automaticLayersDME", "AutomaticLayersDME",
    "automaticLayersNormal", "AutomaticLayersNormal"
]


# -----------------------------
# IO helpers
# -----------------------------

def load_mat_file(mat_path: Path) -> Dict[str, Any]:
    """
    Load MATLAB .mat file.
    Supports traditional .mat via scipy.io.loadmat.
    Supports v7.3 HDF5 .mat via h5py if scipy fails.
    """
    try:
        data = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
        data = {k: v for k, v in data.items() if not k.startswith("__")}
        return data
    except NotImplementedError:
        if h5py is None:
            raise ImportError("This .mat may be v7.3. Please install h5py.")
        return load_hdf5_mat(mat_path)


def load_hdf5_mat(mat_path: Path) -> Dict[str, Any]:
    """
    Basic loader for MATLAB v7.3 HDF5 files.
    """
    data = {}

    def read_obj(obj):
        arr = np.array(obj)
        return arr

    with h5py.File(str(mat_path), "r") as f:
        for key in f.keys():
            try:
                data[key] = read_obj(f[key])
            except Exception:
                pass
    return data


def find_first_key(data: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    for key in candidates:
        if key in data:
            return key
    return None


def to_numpy_array(x: Any) -> np.ndarray:
    arr = np.asarray(x)
    arr = np.squeeze(arr)
    return arr


# -----------------------------
# Shape normalization
# -----------------------------

def normalize_volume_shape(arr: np.ndarray, name: str = "array") -> np.ndarray:
    """
    Normalize volume to shape (H, W, Z).
    Expected Chiu DME image shape is often (496, 768, 61).
    """
    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        arr = arr[:, :, None]

    if arr.ndim != 3:
        raise ValueError(f"{name} should be 2D or 3D, got shape {arr.shape}")

    s = arr.shape

    # Possible (Z, H, W)
    if s[0] <= 128 and s[1] > 128 and s[2] > 128:
        arr = np.transpose(arr, (1, 2, 0))

    # Re-check in case HDF5 has strange order
    if arr.shape[0] <= 128 and arr.shape[1] > 128 and arr.shape[2] > 128:
        arr = np.transpose(arr, (2, 1, 0))

    return arr


def normalize_layer_shape(layer_arr: np.ndarray, target_z: int) -> np.ndarray:
    """
    Normalize retinal layer array to shape (B, W, Z).
    """
    layer_arr = np.asarray(layer_arr)
    layer_arr = np.squeeze(layer_arr)

    if layer_arr.ndim != 3:
        raise ValueError(f"Layer array should be 3D, got {layer_arr.shape}")

    s = layer_arr.shape

    # (B, W, Z)
    if s[2] == target_z and s[0] <= 20:
        return layer_arr

    # (Z, W, B)
    if s[0] == target_z and s[2] <= 20:
        return np.transpose(layer_arr, (2, 1, 0))

    # (W, B, Z)
    if s[2] == target_z and s[1] <= 20:
        return np.transpose(layer_arr, (1, 0, 2))

    raise ValueError(
        f"Cannot infer layer shape {layer_arr.shape} with target_z={target_z}"
    )


def match_volume_shape(arr: np.ndarray, image_shape: Tuple[int, int, int], name: str = "mask") -> np.ndarray:
    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        arr = arr[:, :, None]

    if arr.shape == image_shape:
        return arr

    candidates = [
        arr,
        np.transpose(arr, (1, 0, 2)) if arr.ndim == 3 else arr,
        np.transpose(arr, (2, 1, 0)) if arr.ndim == 3 else arr,
        np.transpose(arr, (1, 2, 0)) if arr.ndim == 3 else arr,
        np.transpose(arr, (2, 0, 1)) if arr.ndim == 3 else arr,
    ]

    for cand in candidates:
        if cand.shape == image_shape:
            return cand

    raise ValueError(
        f"{name} shape {arr.shape} cannot match image shape {image_shape}"
    )


# -----------------------------
# Extract helpers
# -----------------------------

def extract_image_volume(data: Dict[str, Any]) -> Tuple[np.ndarray, str]:
    key = find_first_key(data, IMAGE_KEYS)
    if key is None:
        raise KeyError(f"Cannot find image volume. Available keys: {list(data.keys())}")
    arr = normalize_volume_shape(to_numpy_array(data[key]), name=key)
    return arr, key


def extract_fluid_manual(
    data: Dict[str, Any],
    key_candidates: List[str],
    image_shape: Tuple[int, int, int],
    name: str,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    key = find_first_key(data, key_candidates)
    if key is None:
        return None, None

    arr = normalize_volume_shape(to_numpy_array(data[key]), name=key)
    arr = match_volume_shape(arr, image_shape, name=name)
    return arr, key


def extract_layer_volume(
    data: Dict[str, Any],
    layer_source: str,
    target_z: int,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    if layer_source == "none":
        return None, None

    if layer_source == "manual1":
        key = find_first_key(data, LAYER_KEYS_MANUAL1)
    elif layer_source == "manual2":
        key = find_first_key(data, LAYER_KEYS_MANUAL2)
    else:
        key = find_first_key(data, LAYER_KEYS_GENERIC)

    if key is None:
        return None, None

    layer = normalize_layer_shape(to_numpy_array(data[key]), target_z=target_z)
    return layer, key


# -----------------------------
# Preprocessing helpers
# -----------------------------

def normalize_image_uint8(img: np.ndarray, p_low: float = 1.0, p_high: float = 99.5) -> np.ndarray:
    img = img.astype(np.float32)
    finite = np.isfinite(img)

    if not np.any(finite):
        return np.zeros_like(img, dtype=np.uint8)

    lo = np.percentile(img[finite], p_low)
    hi = np.percentile(img[finite], p_high)

    if hi <= lo:
        hi = lo + 1e-6

    img = np.clip((img - lo) / (hi - lo), 0, 1)
    return (img * 255).astype(np.uint8)


def make_instance_label_from_raw(mask: np.ndarray) -> np.ndarray:
    """
    Keep original instance IDs in fluid annotation.
    NaN -> 0
    Positive integers are preserved.
    """
    mask = np.asarray(mask)
    mask = np.squeeze(mask)

    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask slice, got shape {mask.shape}")

    mask = np.nan_to_num(mask, nan=0.0)
    mask = np.rint(mask)  # just in case values are float but actually integer IDs
    mask[mask < 0] = 0
    return mask.astype(np.uint16)


def has_lesion(label: np.ndarray) -> bool:
    return bool(np.any(label > 0))


def count_instances_from_label(label: np.ndarray) -> int:
    ids = np.unique(label)
    ids = ids[ids > 0]
    return int(len(ids))


def save_overlay_instance(
    img_uint8: np.ndarray,
    inst_label: np.ndarray,
    out_path: Path,
    title: str = "",
):
    """
    Save overlay for an instance label.
    Only used when lesion exists.
    """
    if plt is None:
        return

    positive_ids = np.unique(inst_label)
    positive_ids = positive_ids[positive_ids > 0]
    if len(positive_ids) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(img_uint8, cmap="gray")
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(img_uint8, cmap="gray")
    masked = np.ma.masked_where(inst_label == 0, inst_label)
    axes[1].imshow(masked, cmap="nipy_spectral", alpha=0.55, interpolation="nearest")
    axes[1].set_title(f"Overlay, instances={len(positive_ids)}")
    axes[1].axis("off")

    if title:
        fig.suptitle(title)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# -----------------------------
# Main preprocessing
# -----------------------------

def preprocess_dataset(
    input_root: Path,
    output_root: Path,
    layer_source: str = "manual1",
    save_overlay_flag: bool = False,
    val_ratio: float = 0.2,
    seed: int = 42,
):
    mat_files = sorted(input_root.rglob("*.mat"))

    if not mat_files:
        raise FileNotFoundError(f"No .mat files found under {input_root}")

    dirs = {
        "images": output_root / "images",
        "labels_fluid1": output_root / "labels_fluid1",
        "labels_fluid2": output_root / "labels_fluid2",
        "layers": output_root / "layers",
        "npz": output_root / "npz",
        "overlays_fluid1": output_root / "overlays_fluid1",
        "overlays_fluid2": output_root / "overlays_fluid2",
        "splits": output_root / "splits",
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    metadata_rows = []
    volume_ids = []

    for mat_path in mat_files:
        subject_id = mat_path.stem
        volume_ids.append(subject_id)

        print(f"\nProcessing: {mat_path}")
        data = load_mat_file(mat_path)

        image_vol, image_key = extract_image_volume(data)
        H, W, Z = image_vol.shape

        fluid1_vol, fluid1_key = extract_fluid_manual(
            data=data,
            key_candidates=FLUID_KEYS_MANUAL1,
            image_shape=image_vol.shape,
            name="manualFluid1",
        )

        fluid2_vol, fluid2_key = extract_fluid_manual(
            data=data,
            key_candidates=FLUID_KEYS_MANUAL2,
            image_shape=image_vol.shape,
            name="manualFluid2",
        )

        layer_vol, layer_key = extract_layer_volume(
            data=data,
            layer_source=layer_source,
            target_z=Z,
        )

        print(f"  Image key:  {image_key}, shape: {image_vol.shape}")
        print(f"  Fluid1 key: {fluid1_key}, shape: {None if fluid1_vol is None else fluid1_vol.shape}")
        print(f"  Fluid2 key: {fluid2_key}, shape: {None if fluid2_vol is None else fluid2_vol.shape}")
        print(f"  Layer key:  {layer_key}, shape: {None if layer_vol is None else layer_vol.shape}")

        for z in range(Z):
            image_2d = image_vol[:, :, z]
            img_uint8 = normalize_image_uint8(image_2d)

            if fluid1_vol is not None:
                label1 = make_instance_label_from_raw(fluid1_vol[:, :, z])
            else:
                label1 = np.zeros((H, W), dtype=np.uint16)

            if fluid2_vol is not None:
                label2 = make_instance_label_from_raw(fluid2_vol[:, :, z])
            else:
                label2 = np.zeros((H, W), dtype=np.uint16)

            sample_id = f"{subject_id}_slice{z:03d}"

            img_path = dirs["images"] / f"{sample_id}.png"
            label1_path = dirs["labels_fluid1"] / f"{sample_id}.png"
            label2_path = dirs["labels_fluid2"] / f"{sample_id}.png"
            npz_path = dirs["npz"] / f"{sample_id}.npz"

            iio.imwrite(img_path, img_uint8)
            iio.imwrite(label1_path, label1.astype(np.uint16))
            iio.imwrite(label2_path, label2.astype(np.uint16))

            layer_slice = None
            if layer_vol is not None:
                layer_slice = layer_vol[:, :, z]
                layer_path = dirs["layers"] / f"{sample_id}.npy"
                np.save(layer_path, layer_slice)

            np.savez_compressed(
                npz_path,
                image=img_uint8,
                manualFluid1=label1.astype(np.uint16),
                manualFluid2=label2.astype(np.uint16),
                layers=layer_slice if layer_slice is not None else np.array([]),
                subject_id=subject_id,
                slice_index=z,
                image_key=image_key,
                fluid1_key=fluid1_key if fluid1_key is not None else "",
                fluid2_key=fluid2_key if fluid2_key is not None else "",
                layer_key=layer_key if layer_key is not None else "",
            )

            # Save overlays only for slices with lesion
            if save_overlay_flag:
                if has_lesion(label1):
                    overlay1_path = dirs["overlays_fluid1"] / f"{sample_id}.png"
                    save_overlay_instance(
                        img_uint8,
                        label1,
                        overlay1_path,
                        title=f"{sample_id} - manualFluid1",
                    )

                if has_lesion(label2):
                    overlay2_path = dirs["overlays_fluid2"] / f"{sample_id}.png"
                    save_overlay_instance(
                        img_uint8,
                        label2,
                        overlay2_path,
                        title=f"{sample_id} - manualFluid2",
                    )

            metadata_rows.append({
                "sample_id": sample_id,
                "subject_id": subject_id,
                "slice_index": z,
                "image_path": str(img_path),
                "label1_path": str(label1_path),
                "label2_path": str(label2_path),
                "npz_path": str(npz_path),
                "height": H,
                "width": W,
                "fluid1_has_lesion": int(has_lesion(label1)),
                "fluid2_has_lesion": int(has_lesion(label2)),
                "fluid1_num_instances": count_instances_from_label(label1),
                "fluid2_num_instances": count_instances_from_label(label2),
                "fluid1_max_instance_id": int(label1.max()) if label1.size > 0 else 0,
                "fluid2_max_instance_id": int(label2.max()) if label2.size > 0 else 0,
                "image_key": image_key,
                "fluid1_key": fluid1_key if fluid1_key is not None else "",
                "fluid2_key": fluid2_key if fluid2_key is not None else "",
                "layer_key": layer_key if layer_key is not None else "",
            })

    save_metadata(output_root / "metadata.csv", metadata_rows)
    make_volume_split(volume_ids, dirs["splits"], val_ratio=val_ratio, seed=seed)

    print("\nDone.")
    print(f"Output root: {output_root}")
    print(f"Metadata: {output_root / 'metadata.csv'}")


# -----------------------------
# Metadata / split
# -----------------------------

def save_metadata(csv_path: Path, rows: List[Dict[str, Any]]):
    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_volume_split(volume_ids: List[str], split_dir: Path, val_ratio: float = 0.2, seed: int = 42):
    volume_ids = sorted(set(volume_ids))
    rng = random.Random(seed)
    rng.shuffle(volume_ids)

    n_val = max(1, int(round(len(volume_ids) * val_ratio)))
    val_ids = sorted(volume_ids[:n_val])
    train_ids = sorted(volume_ids[n_val:])

    (split_dir / "train_volumes.txt").write_text("\n".join(train_ids), encoding="utf-8")
    (split_dir / "val_volumes.txt").write_text("\n".join(val_ids), encoding="utf-8")

    print(f"Train volumes: {len(train_ids)}, Val volumes: {len(val_ids)}")


# -----------------------------
# Inspect mode
# -----------------------------

def inspect_dataset(input_root: Path):
    mat_files = sorted(input_root.rglob("*.mat"))

    if not mat_files:
        print(f"No .mat files found under {input_root}")
        return

    for mat_path in mat_files:
        print("\n" + "=" * 80)
        print(f"File: {mat_path}")
        try:
            data = load_mat_file(mat_path)
            for key, value in data.items():
                try:
                    arr = np.asarray(value)
                    print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
                except Exception:
                    print(f"  {key}: type={type(value)}")
        except Exception as e:
            print(f"  Failed to read: {e}")


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_root",
        type=str,
        default=r"F:\1_项目文档\2026_instaneSeg\BOE_ Chiu_DME_OCT_datasets\Segmentation of OCT images (DME)_datasets",
        help="Root directory containing .mat files.",
    )

    parser.add_argument(
        "--output_root",
        type=str,
        default=r"F:\1_项目文档\2026_instaneSeg\BOE_ Chiu_DME_OCT_datasets\preprocessed_instance",
        help="Output directory.",
    )

    parser.add_argument(
        "--layer_source",
        type=str,
        default="manual1",
        choices=["manual1", "manual2", "auto", "none"],
        help="Which layer annotation to save.",
    )

    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Validation volume ratio.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for volume split.",
    )

    parser.add_argument(
        "--save_overlay",
        action="store_true",
        help="Save overlay visualizations for slices with lesions.",
    )

    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Only inspect .mat keys and shapes.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    if args.inspect:
        inspect_dataset(input_root)
        return

    preprocess_dataset(
        input_root=input_root,
        output_root=output_root,
        layer_source=args.layer_source,
        save_overlay_flag=args.save_overlay,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()