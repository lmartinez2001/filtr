from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm
import os
from collections import defaultdict
import util.misc as utils

def process_sample(sample_path, quantile = None):
    # Placeholder for processing logic
    data = np.load(sample_path)
    pd = data["pd1"]
    if quantile is not None:
        pd, _ = utils.h1_threshold_quantile(pd, quantile)
    stats = {
        "length": len(pd),
        "max": np.max(pd),
        "min": np.min(pd),
    }
    return sample_path, stats

def main(args):
    input_dir = args.input_dir
    output_file = args.output_file
    quantile = args.quantile
    
    samples = os.listdir(input_dir)
    print(f"==> Found {len(samples)} samples in {input_dir}")
    paths  = [os.path.join(input_dir, sample) for sample in samples]

    all_stats = defaultdict(list)
    print(f"==> Processing samples...")
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_sample, sample_path, quantile) for sample_path in paths]

        for future in tqdm(as_completed(futures), total=len(futures)):
            sample_path, stats = future.result()
            for key, value in stats.items():
                all_stats[key].append(value)

    # Save all_stats to output_file
    print(f"==> Saving statistics to {output_file}")
    np.savez_compressed(output_file, **all_stats)



def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="PDTR Stats")
    parser.add_argument(
        "--input_dir",
        required=True,
        type=str,
        help="Input directory containing the persistence diagrams.",
    )
    parser.add_argument(
        "--output_file",
        required=True,
        type=str,
        help="Output file for saving the statistics.",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        required=False,
        help="Quantile to compute for the statistics.",
    )
    return parser.parse_args()


if __name__ == "__main__":   
    args = parse_args()
    main(args)