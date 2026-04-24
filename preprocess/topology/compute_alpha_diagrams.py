import os
import numpy as np

from tqdm import tqdm
from pathlib import Path
from gudhi import AlphaComplex
from datasets.utils import pc_norm
from concurrent.futures import ProcessPoolExecutor, as_completed
from preprocess.topology.utils import (
    positive_int,
    save_persistence_diagrams,
    validate_point_cloud_array,
)


def compute_alpha_topological_features(points: np.ndarray, rescale: bool = True):
    alpha_complex = AlphaComplex(points=points)
    st = alpha_complex.create_simplex_tree()
    st.compute_persistence()

    # extract diagrams by dimension
    pds = {dim: st.persistence_intervals_in_dimension(dim) for dim in range(3)}

    if not rescale:
        return pds

    # collect all finite pairs to compute normalization
    finite_vals = (
        np.concatenate(
            [
                pds[dim][np.isfinite(pds[dim]).all(axis=1)]
                for dim in pds
                if len(pds[dim]) > 0
            ],
            axis=0,
        )
        if any(len(pds[dim]) > 0 for dim in pds)
        else np.array([])
    )

    if finite_vals.size == 0:
        return pds

    bmin, bmax = finite_vals[:, 0].min(), finite_vals[:, 1].max()
    scale = bmax - bmin if bmax > bmin else 1.0

    # normalize each PD
    for dim in pds:
        if len(pds[dim]) > 0:
            pd = pds[dim]
            pd = np.where(np.isfinite(pd), (pd - bmin) / scale, pd)
            pds[dim] = pd

    return pds


def process_sample(pcd_path: str, output_dir: str, rescale: bool) -> tuple:
    sample_name = Path(pcd_path).stem
    try:
        points = np.load(pcd_path)
        validate_point_cloud_array(points, sample_name)
        points = pc_norm(points)

        pds = compute_alpha_topological_features(points, rescale=rescale)

        save_persistence_diagrams(
            os.path.join(output_dir, f"{sample_name}.npz"),
            pds,
            dimensions=(0, 1, 2),
        )

    except Exception as e:
        print(f"Failed to process sample {sample_name}: {e}")
        return False, sample_name

    return True, ""


def main(args) -> None:
    input_dir = args.pcd_dir
    output_dir = args.output_dir
    rescale = args.rescale
    n_workers = args.n_workers

    os.makedirs(output_dir, exist_ok=True)

    pcd_paths = sorted(Path(input_dir).glob("*.npy"))

    print(f"==> Computing topological features using Alpha complex (rescale={rescale})")
    failed, success = 0, 0
    failed_samples = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(process_sample, pcd_path, output_dir, rescale)
            for pcd_path in pcd_paths
        ]

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing samples"
        ):
            is_success, fail_sample_name = future.result()
            if is_success:
                success += 1
            else:
                failed += 1
                failed_samples.append(fail_sample_name)

    print(
        f"Failed to process {failed} samples, successfully processed {success} samples."
    )


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute topological features using Alpha complex for any dataset"
    )
    parser.add_argument(
        "--pcd_dir",
        required=True,
        type=str,
        help="Path to the input directory. Point clouds should be in .npy format",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="Path to the output directory. Created if it does not exist",
    )
    parser.add_argument(
        "--rescale",
        action="store_true",
        help="If set, rescales persistence diagrams to [0, 1]. If omitted, keeps original scales.",
    )
    parser.add_argument(
        "--n_workers",
        type=positive_int,
        default=32,
        help="Number of parallel workers to use for processing samples.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
