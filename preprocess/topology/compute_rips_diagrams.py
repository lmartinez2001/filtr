import os
import numpy as np

from tqdm import tqdm
from pathlib import Path
from gudhi import RipsComplex
from datasets.utils import pc_norm
from concurrent.futures import ProcessPoolExecutor, as_completed
from preprocess.topology.utils import (
    positive_float,
    positive_int,
    save_persistence_diagrams,
    validate_point_cloud_array,
)


def compute_rips_features(
    points: np.ndarray, max_edge_length: float, max_dimension: int
):
    rips_complex = RipsComplex(points=points, max_edge_length=max_edge_length)
    # max_dimension must be 2 to save pd0 and pd1
    st = rips_complex.create_simplex_tree(max_dimension=max_dimension)
    st.compute_persistence()

    # extract diagrams by dimension
    pds = {
        dim: st.persistence_intervals_in_dimension(dim) for dim in range(max_dimension)
    }

    return pds


def process_sample(
    pcd_path: str,
    output_dir: str,
    max_edge_length: float,
    max_dimension: int,
) -> tuple:
    """Process a single point cloud. Takes the path to high-res (8192) pcd and computes persistence diagrams."""
    try:
        sample_name = Path(pcd_path).stem
        points = np.load(pcd_path)
        validate_point_cloud_array(points, sample_name)
        points = pc_norm(points)

        pds = compute_rips_features(
            points,
            max_edge_length=max_edge_length,
            max_dimension=max_dimension,
        )

        save_persistence_diagrams(
            os.path.join(output_dir, f"{sample_name}.npz"),
            pds,
            dimensions=(0, 1),
        )

    except Exception as e:
        print(f"Failed to process sample {sample_name}: {e}")
        return False, sample_name

    return True, ""


def main(args) -> None:
    input_dir = args.pcd_dir
    output_dir = args.output_dir
    max_edge_length = args.max_edge_length
    max_dimension = args.max_dimension
    split_file = args.split_file

    os.makedirs(output_dir, exist_ok=True)
    if split_file is None:
        points = [path.name for path in sorted(Path(input_dir).glob("*.npy"))]
    else:
        filenames = np.loadtxt(split_file, dtype=str)
        points = [f + ".npy" for f in filenames]

    existing_files = {path.name for path in Path(output_dir).glob("*.npz")}
    points = [p for p in points if (p.replace(".npy", ".npz") not in existing_files)]
    print(f"==> Ignoring already processed files: {len(existing_files)}")

    pcd_paths = [Path(input_dir) / p for p in points if p.endswith(".npy")]

    print(
        f"==> Computing topological features using Rips complex for {len(pcd_paths)} samples"
    )
    print(f"Max edge length: {max_edge_length}, Max dimension: {max_dimension}")

    failed, success = 0, 0
    failed_samples = []

    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        futures = [
            executor.submit(
                process_sample,
                pcd_path,
                output_dir,
                max_edge_length,
                max_dimension,
            )
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
        description="Compute topological features using Rips complex for any dataset"
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
        "--max_edge_length",
        type=positive_float,
        default=2.0,
        help="Maximum edge length for Rips complex construction.",
    )
    parser.add_argument(
        "--max_dimension",
        type=positive_int,
        default=2,
        choices=(2,),
        help="Maximum homology dimension to compute. Only 2 is supported.",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default=None,
        help="List of files to process. If not provided, process all files in the input directory.",
    )
    parser.add_argument(
        "--n_workers",
        type=positive_int,
        default=8,
        help="Number of parallel workers to use for processing samples.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
