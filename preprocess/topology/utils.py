import os
import tempfile
import numpy as np


def validate_point_cloud_array(points: np.ndarray, sample_name: str) -> None:
    points_array = np.asarray(points)
    if points_array.ndim != 2:
        raise ValueError(
            f"Expected a 2D point cloud array for sample '{sample_name}', "
            f"got shape {points_array.shape}."
        )
    if points_array.shape[0] == 0:
        raise ValueError(f"Point cloud for sample '{sample_name}' is empty.")
    if points_array.shape[1] != 3:
        raise ValueError(
            f"Expected point cloud coordinates with shape (N, 3) for sample "
            f"'{sample_name}', got shape {points_array.shape}."
        )


def save_npz_atomic(output_path: str, **arrays: np.ndarray) -> None:
    output_dir = os.path.dirname(output_path) or "."
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_dir,
            suffix=".npz",
            prefix=".tmp_",
            delete=False,
        ) as temp_file:
            temp_path = temp_file.name

        np.savez_compressed(temp_path, **arrays)
        os.replace(temp_path, output_path)
    except Exception:
        if temp_path is not None and os.path.exists(temp_path):
            os.remove(temp_path)
        raise


def save_persistence_diagrams(
    output_path: str, diagrams: dict[int, np.ndarray], dimensions: tuple[int, ...]
) -> None:
    arrays = {f"pd{dim}": diagrams[dim] for dim in dimensions}
    save_npz_atomic(output_path, **arrays)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise ValueError("Value must be a positive integer.")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise ValueError("Value must be a strictly positive float.")
    return parsed
