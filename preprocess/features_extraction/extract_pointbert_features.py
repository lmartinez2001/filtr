import os
import sys
from pathlib import Path

import torch
import numpy as np

from tqdm import tqdm
from easydict import EasyDict as edict


def _find_pointbert_root() -> Path:
    """Locate the vendored PointBERT checkout without modifying the submodule."""
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / "third_party" / "pointbert"

    if (candidate / "models" / "Point_BERT.py").exists():
        return candidate

    raise FileNotFoundError(
        "Could not locate the PointBERT submodule under third_party/pointbert."
    )


POINTBERT_ROOT = _find_pointbert_root()
if str(POINTBERT_ROOT) not in sys.path:
    sys.path.insert(0, str(POINTBERT_ROOT))

from models.Point_BERT import Point_BERT
from utils.config import cfg_from_yaml_file
 

class PointBERTFeatureExtractor:

    def __init__(self, config, pretrained_path):
        self.config = config
        self.model = Point_BERT(config).cuda()
        self.load_pretrained_weights(pretrained_path)
        self.model.eval()
        self.n_transformer_layers = len(self.model.transformer_q.blocks.blocks)

    def load_pretrained_weights(self, checkpoint_path):
        print(f"Loading pretrained weights from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path)

        if "base_model" in ckpt:
            state_dict = ckpt["base_model"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt

        # Remove "module." prefix if present (from DataParallel)
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                cleaned_state_dict[k[7:]] = v
            else:
                cleaned_state_dict[k] = v

        # Load weights
        missing_keys, unexpected_keys = self.model.load_state_dict(
            cleaned_state_dict, strict=False
        )
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")

        print("Pretrained weights loaded successfully")

    @torch.no_grad()
    def extract_features(self, pts, return_all_layers=False):
        """
        Extract features from different stages of the model

        Args:
            pts: Input point cloud tensor [B, N, 3]
            return_all_layers: If True, return features from all transformer layers

        Returns:
            Dictionary containing different feature representations:
            - "encoder_features": Features after point cloud encoder [B, G, encoder_dims]
            - "reduced_features": Features after dimension reduction [B, G, trans_dim]
            - "transformer_features": Features after transformer blocks [B, G+1, trans_dim]
            - "cls_token": CLS token feature [B, trans_dim]
            - "patch_tokens": Patch token features [B, G, trans_dim]
            - "final_features": Final concatenated features [B, trans_dim*2]
            - "layer_features": Features from each transformer layer (if return_all_layers=True)
        """

        # Set model to evaluation mode
        self.model.eval()
        pts = pts.cuda()

        # Use the transformer_q for feature extraction
        transformer = self.model.transformer_q

        # Step 1: Divide point cloud into groups
        neighborhood, center = self.model.group_divider(pts)  # [B, G, N, 3], [B, G, 3]

        # Step 2: Encode input cloud blocks using the Mini-Pointnet encoder
        group_input_tokens = transformer.encoder(neighborhood)  # [B, G, encoder_dims]
        encoder_features = group_input_tokens.clone()

        # Step 3: Reduce dimensions
        group_input_tokens = transformer.reduce_dim(
            group_input_tokens
        )  # [B, G, trans_dim]
        reduced_features = group_input_tokens.clone()

        # Step 4: Prepare tokens
        batch_size = group_input_tokens.size(0)
        cls_tokens = transformer.cls_token.expand(batch_size, -1, -1)
        cls_pos = transformer.cls_pos.expand(batch_size, -1, -1)  # [B, 1, trans_dim]

        # Step 5: Add positional embedding
        pos = transformer.pos_embed(center)

        # Step 6: Create final input
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)  # [B, G+1, trans_dim]
        pos = torch.cat((cls_pos, pos), dim=1)  # [B, G+1, trans_dim]

        # Step 7: Pass through transformer blocks
        layer_features = []
        if return_all_layers:
            # Extract features from each layer
            for i, block in enumerate(transformer.blocks.blocks):
                x = block(x + pos)  # [B, G+1, trans_dim]
                layer_features.append(x.clone())
        else:
            # Just pass through all blocks
            x = transformer.blocks(x, pos)

        # Step 8: Apply layer normalization
        x = transformer.norm(x)
        transformer_features = x.clone()  # [B, G+1, trans_dim]

        # Step 9: Extract different feature types
        cls_token = x[:, 0]  # [B, trans_dim]
        patch_tokens = x[:, 1:]  # [B, G, trans_dim]

        cls_pos = pos[:, 0, :]  # [B, trans_dim]
        patch_pos = pos[:, 1:, :]  # [B, G, trans_dim]

        # Step 10: Create final concatenated features (as used in classification)
        concat_features = torch.cat(
            [cls_token, patch_tokens.max(1)[0]], dim=-1
        )  # [B, trans_dim*2]

        features = {
            "encoder_features": encoder_features,
            "reduced_features": reduced_features,
            "transformer_features": transformer_features,
            "cls_token": cls_token,
            "patch_tokens": patch_tokens,
            "final_features": concat_features,
            "cls_pos_embeddings": cls_pos,
            "patch_pos_embeddings": patch_pos,
            "group_centers": center,
        }

        if return_all_layers:
            for i, layer_feature in enumerate(layer_features):
                features[f"layer_features_{i}"] = (
                    layer_feature  # n_layers x [1, n_patches + cls, hidden_dim]
                )

        return edict(features)

    @torch.no_grad()
    def extract_cls_features_only(self, pts):
        """
        Extract only the CLS token features (most commonly used)

        Args:
            pts: Input point cloud tensor [B, N, 3]

        Returns:
            cls_features: CLS token features [B, trans_dim]
        """
        features = self.extract_features(pts, return_all_layers=False)
        return features.cls_token


def pc_norm(pcd):
    mean = pcd.mean(axis=0)
    pcd -= mean
    max_dist = np.max(np.linalg.norm(pcd, axis=1))
    pcd /= max_dist
    return pcd


def random_translation(pcd, shift_range=0.1):
    shifts = np.random.uniform(-shift_range, shift_range, 3)
    translated_pcd = pcd + shifts
    return translated_pcd

def random_sample_points(pcd: np.ndarray, n_samples: int, rng: np.random.Generator):
    if len(pcd) < n_samples:
        raise ValueError(
            f"Not enough points to sample {n_samples} points. Available: {len(pcd)}"
        )
    n_points = len(pcd)
    indices = rng.choice(n_points, n_samples, replace=False)
    return pcd[indices]


def load_pcd(sample_path: str, n_points: int, rng: np.random.Generator):
    pcd = np.load(sample_path)
    if not isinstance(pcd, np.ndarray):
        raise ValueError(f"Loaded object is not a numpy array: {type(pcd)!r}")
    if pcd.ndim != 2 or pcd.shape[1] != 3:
        raise ValueError(
            f"Expected point cloud array of shape [N, 3], got {pcd.shape}"
        )
    pcd = pc_norm(pcd)
    pcd = random_sample_points(pcd, n_points, rng)
    pcd = torch.from_numpy(pcd).float()  # [N, 3]
    return pcd

def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(description="Extract Point-BERT features")
    parser.add_argument(
        "--model_config",
        type=str,
        default=str(POINTBERT_ROOT / "cfgs" / "Mixup_models" / "Point-BERT.yaml"),
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default="ckpts/Point-BERT.pth",
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--dvae_ckpt",
        type=str,
        required=True,
        help="Path to the dVAE checkpoint used to initialize Point-BERT",
    )
    parser.add_argument(
        "--pcd_dir",
        type=str,
        required=True,
        help="Point cloud directory (Only processes .npy files)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--in_points",
        type=int,
        default=1024,
        help="Input points for feature extraction",
    )
    parser.add_argument(
        "--out_points",
        type=int,
        default=2048,
        help="Output points for feature extraction",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for deterministic point sampling",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    pcd_dir = args.pcd_dir
    output_dir = args.output_dir

    print(f"==> Launch this script from the repository root: {POINTBERT_ROOT.parents[1]}")

    config = cfg_from_yaml_file(args.model_config)
    config.model.dvae_config.ckpt = args.dvae_ckpt
    output_resolution = args.out_points
    config.model.dvae_config.num_group = output_resolution // config.model.dvae_config.group_size
    print(f"==> Output resolution: {output_resolution}")
    

    feature_extractor = PointBERTFeatureExtractor(config.model, args.model_ckpt)
    print("==> Feature extractor initialized")

    all_samples = sorted(s for s in os.listdir(pcd_dir) if s.endswith(".npy"))
    all_sample_paths = [os.path.join(pcd_dir, s) for s in all_samples]


    # all_n_points = args.in_points
    # for n_points in all_n_points:
    n_points = args.in_points
    os.makedirs(
        f"{output_dir}/out_{output_resolution}/in_{n_points}",
        exist_ok=True,
    )

    print(f"==> Processing dataset with {n_points} input points")

    n_layers = feature_extractor.n_transformer_layers
    rng = np.random.default_rng(args.seed)
    failed_samples = []

    for sample_path in tqdm(all_sample_paths, desc="Extracting features"):
        sample_name = os.path.basename(sample_path).replace(".npy", "")
        try:
            pcd = load_pcd(sample_path, n_points, rng)

            features = feature_extractor.extract_features(
                pcd.unsqueeze(0), return_all_layers=True
            )

            layer_features = {
                f"layer_features_{i}": features[f"layer_features_{i}"]
                .squeeze(0)
                .cpu()
                .numpy()
                for i in range(n_layers)
            }

            res = {
                "final_features": features.final_features.squeeze(0).cpu().numpy(),
                "cls_token": features.cls_token.squeeze(0).cpu().numpy(),
                "patch_tokens": features.patch_tokens.squeeze(0).cpu().numpy(),
                "sample_name": sample_name,
                "patch_pos_embeddings": features.patch_pos_embeddings.squeeze(0)
                .cpu()
                .numpy(),
                "cls_pos_embeddings": features.cls_pos_embeddings.squeeze(0)
                .cpu()
                .numpy(),
                "group_centers": features.group_centers.squeeze(0).cpu().numpy(),
            }

            np.savez(
                f"{args.output_dir}/out_{output_resolution}/in_{n_points}/{sample_name}.npz",
                **res,
                **layer_features,
            )
        except Exception as exc:
            failed_samples.append((sample_path, str(exc)))

    if failed_samples:
        print("==> Feature extraction completed with failures:")
        for sample_path, error in failed_samples:
            print(f"  - {sample_path}: {error}")
    else:
        print("==> Feature extraction completed successfully with no failures")
