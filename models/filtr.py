import torch
import logging
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


class FILTR(nn.Module):
    def __init__(
        self,
        decoder: nn.Module,
        num_queries: int,
        in_feature_dim: int,
        aux_loss: bool,
        use_layer_norm_adapter: bool,

    ):
        """
        Parameters:
            decoder: decoder module, with attribute d_model
            num_queries: number of object queries (max PD pairs to predict)
            in_feature_dim: C_in dimension of the precomputed features
            aux_loss: auxiliary decoder losses flag
        """
        super().__init__()
        self.num_queries = num_queries
        self.decoder = decoder
        hidden_dim = decoder.d_model
        self.use_layer_norm_adapter = use_layer_norm_adapter

        logger.info("FILTR initialized with %s queries", num_queries)

        # Query embeddings that carry per persistence pair information (coords and existence)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Linear projections from sequence feature dim(s) to d_model
        self.input_proj_seq = nn.Linear(in_feature_dim, hidden_dim)
        self.pos_proj_seq = nn.Linear(in_feature_dim, hidden_dim)

        # Heads: regress persistence pairs (birth, death) and existence logit
        self.pairs_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.exist_embed = nn.Linear(hidden_dim, 1)

        if self.use_layer_norm_adapter:
            logger.info("Using LayerNorm adapter after input projection")
            self.src_proj_norm = nn.LayerNorm(hidden_dim, eps=1e-6)

        # Use auxiliary decoder losses
        self.aux_loss = aux_loss

    def forward(self, features: Tensor, pos: Tensor):
        """
        Args:
            features: FloatTensor [B, seq_len, in_feature_dim] from pretrained backbone
            pos:      FloatTensor [B, seq_len, in_feature_dim] from pretrained backbone

        Returns:
            dict with keys:
                - "pred_pairs": [B, num_queries, 2] (values in [0, ?])
                - "pred_exist": [B, num_queries] (logits)
                - "aux_outputs": list of dicts (if aux_loss is True)
        """

        # Project to model dim
        src_seq = self.input_proj_seq(features)     # [B, seq_len, d_model]
        if self.use_layer_norm_adapter:
            src_seq = self.src_proj_norm(src_seq)
        pos_seq = self.pos_proj_seq(pos)            # [B, seq_len, d_model]

        # hs: [num_layers or 1, B, num_queries, d_model]
        # discard memore because no backbone
        hs, _ = self.decoder(src_seq, self.query_embed.weight, pos_seq)

        # Regress persistence pairs
        outputs_pairs = self.pairs_embed(hs)  # [num_layers, B, num_queries, 2]
        last_output_pairs = outputs_pairs[-1] # [B, num_queries, 2]
        last_output_pairs = self._format_pairs(last_output_pairs) # [B, num_queries, 2]
        
        # Regress existence logits
        exist_logits = self.exist_embed(hs)  # [num_layers, B, num_queries, 1]
        exist_logits = exist_logits.squeeze(-1)  # [num_layers, B, num_queries]

        out = {
            "pred_pairs": last_output_pairs, #  (B, num_queries, 2)
            "pred_exist": exist_logits[-1]  #   (B, num_queries) take from last layer
            }
        
        # useless but here just in case
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_pairs, exist_logits)

        return out
    
    def _format_pairs(self, outputs_pairs: Tensor):
        """
        Format raw logits to valid persistence pairs
        
        :param outputs_pairs: (B, num_queries, 2) raw output logits
        :type outputs_pairs: Tensor
        """
        b_raw = outputs_pairs[..., 0] # birth (B, num_queries)
        d_raw = outputs_pairs[..., 1] # death (B, num_queries)
        b = torch.sigmoid(b_raw)
        d = b + F.softplus(d_raw)
        return torch.stack([b, d], dim=-1) # (B, num_queries, 2)
    


class FILTREnd2End(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        num_queries: int,
        aux_loss: bool,
    ):
        """
        Parameters:
            decoder: decoder module, with attribute d_model
            num_queries: number of object queries (max PD pairs to predict)
            in_feature_dim: C_in dimension of the precomputed features
            aux_loss: auxiliary decoder losses flag
        """
        super().__init__()
        self.num_queries = num_queries
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder # final decoder 
        hidden_dim = decoder.d_model

        logger.info("FILTR initialized with %s queries", num_queries)

        # Query embeddings that carry per persistence pair information (coords and existence²)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)


        # Heads: regress persistence pairs (birth, death) and existence logit
        self.pairs_embed = MLP(input_dim=hidden_dim, 
                               hidden_dim=hidden_dim, 
                               output_dim=2, 
                               num_layers=3)
        self.exist_embed = nn.Linear(hidden_dim, 1)

        # Use auxiliary decoder losses
        self.aux_loss = aux_loss

    def forward(self, pts):
        """
        Args:
            pts: FloatTensor [B, n_pts, 3] raw point cloud

        Returns:
            dict with keys:
                - "pred_pairs": [B, num_queries, 2] (values in [0, ?])
                - "pred_exist": [B, num_queries] (logits)
                - "aux_outputs": list of dicts (if aux_loss is True)
        """
        bs = pts.shape[0]
        backbone_output = self.backbone(pts) # [B, 1024, 256]

        # pointnet2
        if len(backbone_output) == 2:
            features = backbone_output[0]  # [B, ?, C]
            pts = backbone_output[1]  # [B, ?, 3]
        else:
            features = backbone_output  # [B, ?, C]
        pos_embed = self.encoder.pos_embed(pts)  # [B, 1024, 256]
        encoded = self.encoder(features, pos_embed)  # [B, 1024, 256]
        # pos_embed = self.encoder.pos_embed.unsqueeze(0).repeat(bs,1,1)  # [B, 1024, 256]

        # hs: [num_layers or 1, B, num_queries, d_model]
        hs, _ = self.decoder(src=encoded, query_embed=self.query_embed.weight, pos_embed=pos_embed)

        # Regress persistence pairs
        outputs_pairs = self.pairs_embed(hs)  # [num_layers, B, num_queries, 2]
        output_pairs = self._format_pairs(outputs_pairs[-1])  # [B, num_queries, 2]
        exist_logits = self.exist_embed(hs).squeeze(-1)  # [num_layers, B, num_queries]

        out = {
            "pred_pairs": output_pairs, #  (B, num_queries, 2)
            "pred_exist": exist_logits[-1]  #   (B, num_queries)
            }
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_pairs, exist_logits)

        return out
    
    def _format_pairs(self, outputs_pairs: Tensor):
        b_raw = outputs_pairs[..., 0] # (B, num_queries)
        d_raw = outputs_pairs[..., 1] # (B, num_queries)
        b = torch.sigmoid(b_raw)
        d = b + F.softplus(d_raw)
        return torch.stack([b, d], dim=-1) # (B, num_queries, 2)

    @torch.jit.unused
    def _set_aux_loss(self, outputs_pairs: Tensor, exist_logits: Tensor):
        return [{"pred_pairs": p, "pred_exist": e} for p, e in zip(outputs_pairs[:-1], exist_logits[:-1])]


class SetCriterion(nn.Module):
    def __init__(self, 
                 matcher, 
                 weight_dict: dict, 
                 losses: list):
        super().__init__()
        if not set(losses).issubset({"existence", "recon", "diag", "w2"}):
            raise ValueError(f"Losses {losses} not recognized")

        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses


    def forward(self, outputs: dict, targets: list):
        """        
        :param outputs: Dictionary outputted by the forward function of FILTR. Must contain the key "pred_pairs" and "pred_exist".
        :param targets: List[dict] containing ground truth persistence diagrams. Each diagram is stored in the "pairs" field. Diagrams in the batch are of different sizes.  
        """
        # outputs["pred_pais"] (batch_size, num_queries, 2)
        # outputs["pred_exist"] (batch_size, num_queries)
        # targets list of dicsts with pairs key: (n_gt, 2)
        bs = len(targets)
        device = outputs["pred_pairs"].device

        # ==> MAching
        indices: list = self.matcher(outputs, targets) # Match gt and predicted pairs
        
        # Number of pairs for each gt diagram, since they have different sizes
        num_pairs_list = [targets[b]["pairs"].shape[0] for b in range(bs)]
        num_pairs = torch.tensor(sum(num_pairs_list), dtype=torch.float, device=device)
        normalizer = num_pairs.clamp_min(1.0)

        losses = {}
        for loss_name in self.losses: # existence, recon, diag, ...
            losses.update(self.get_loss(loss_name, outputs, targets, indices, normalizer))

        # ==> Weight
        losses = {k: v * self.weight_dict[k] for k, v in losses.items()}
        return losses


    def existence_loss(self, outputs, targets, indices, num_pairs):
        pred_exist = outputs["pred_exist"] # (B, Q)
        B, Q = pred_exist.shape
        
        target_exist = torch.zeros_like(pred_exist)
        idx = self._get_src_permutation_idx(indices)
        if idx[0].numel() > 0:
            target_exist[idx] = 1.0

        pos_count = target_exist.sum(dim=1)
        neg_count = Q - pos_count
        
        # Keep a meaningful negative-only signal when a sample has no GT pairs.
        w_neg = torch.where(
            pos_count > 0,
            pos_count / (neg_count + 1e-6),
            torch.ones_like(pos_count)
        )
        
        # Broadcast weights: w_pos=1.0, w_neg varies per sample
        w = torch.ones_like(pred_exist)
        # Assign w_neg to negative indices. 
        # (This masking trick is faster than unsqueeze/broadcast logic)
        mask_neg = (target_exist == 0)
        w[mask_neg] = w_neg.unsqueeze(1).expand_as(w)[mask_neg]

        # BCE Loss
        # reduction='none' preserves (B, Q) structure
        loss_unreduced = F.binary_cross_entropy_with_logits(pred_exist, target_exist, weight=w, reduction='none')
        
        # FIXED: Global Normalization
        # Sum everything and divide by num_pairs to match scale of other losses
        # (Alternatively, divide by B to keep it 'per image' classification, but DETR divides by num_pairs)
        return {"existence": loss_unreduced.sum() / num_pairs}
    
    def reconstruction_loss(self, outputs, targets, indices, num_pairs):
        # 1. Align Predictions
        src_idx = self._get_src_permutation_idx(indices) # Tuple(Batch_idx, Pred_idx)
        if src_idx[0].numel() == 0:
            return {"recon": outputs["pred_pairs"].sum() * 0.0}
        src_pairs = outputs["pred_pairs"][src_idx]       # (Total_Matched, 2)

        # 2. Align Targets
        # Concatenate all matching target pairs in the same order
        target_pairs = torch.cat([t["pairs"][J] for t, (_, J) in zip(targets, indices)], dim=0)

        # 3. Compute MSE (Vectorized)
        # Sum reduction, then normalize by GLOBAL num_pairs
        loss = F.mse_loss(src_pairs, target_pairs, reduction='sum') / num_pairs
        return {"recon": loss}
    
    def diagonal_loss(self, outputs: dict, targets, indices: list, num_pairs):
        pred_pairs = outputs["pred_pairs"] # (B, Q, 2)
        
        # 1. Identify Unmatched Predictions
        src_idx = self._get_src_permutation_idx(indices)
        
        # Create a boolean mask of ALL predictions
        mask = torch.ones(pred_pairs.shape[:2], dtype=torch.bool, device=pred_pairs.device)
        # Set matched indices to False
        mask[src_idx] = False
        
        # Select all unmatched pairs across the entire batch at once
        unmatched = pred_pairs[mask] # (Total_Unmatched, 2)
        if unmatched.numel() == 0:
            return {"diag": pred_pairs.sum() * 0.0}
        
        # Constraint Check: Preds > GT, so unmatched is never empty.
        
        # 2. Compute Distance to Diagonal (x-y)
        diff = unmatched[:, 1] - unmatched[:, 0]
        
        # 3. Loss Calculation
        # Sum squared errors, divide by num_pairs (or Total_Unmatched if you prefer, but num_pairs is standard)
        loss = (diff ** 2).sum() / num_pairs
        return {"diag": loss}
    
    def _get_src_permutation_idx(self, indices):
        # Merges list of lists into two tensors: batch indices and src indices
        # [(P0, T0), (P1, T1)] -> ([0, 0, ..., 1, 1, ...], [P0..., P1...])
        src_tensors = [src for (src, _) in indices if src.numel() > 0]
        if not src_tensors:
            device = indices[0][0].device if indices else torch.device("cpu")
            empty_idx = torch.empty(0, dtype=torch.long, device=device)
            return empty_idx, empty_idx

        batch_idx = torch.cat([
            torch.full_like(src, i) for i, (src, _) in enumerate(indices) if src.numel() > 0
        ])
        src_idx = torch.cat(src_tensors)
        return batch_idx, src_idx

    # def existence_loss(self, outputs, targets, indices, num_pairs):
    #     pred_exist = outputs["pred_exist"]  # (B, num_queries)
    #     B, Q = pred_exist.shape

    #     target_exist = torch.zeros_like(pred_exist)
    #     for b, (pred_idx, _) in enumerate(indices):
    #         target_exist[b, pred_idx] = 1.0

    #     # counts per sample
    #     pos_count = target_exist.sum(dim=1)                      # (B,)
    #     neg_count = (Q - pos_count)                                    # (B,)

    #     # class-balanced weights per sample
    #     w_pos = torch.ones_like(pos_count) # (B,)
    #     w_neg = torch.where(neg_count > 0, (pos_count / (neg_count + 1e-12)), torch.zeros_like(neg_count))

    #     # broadcast to [B, Q]
    #     w = torch.where(target_exist.bool(), w_pos.unsqueeze(1), w_neg.unsqueeze(1))

    #     elem = F.binary_cross_entropy_with_logits(pred_exist, target_exist, reduction="none")

    #     # normalized per sample, then averaged over batch
    #     denom = w.sum(dim=1).clamp_min(1.0)                # [B]
    #     per_sample = (elem * w).sum(dim=1) / denom         # [B]
    #     loss = per_sample.mean()                           # scalar

    #     return {"existence": loss}


    # def reconstruction_loss(self, outputs, targets, indices, num_pairs):
    #     pred_pairs = outputs["pred_pairs"] # (B, num_queries, 2)
    #     per_sample_losses = []

    #     for sample, (pred_idx, tgt_idx) in enumerate(indices):
    #         if len(pred_idx) == 0:
    #             continue
    #         p = pred_pairs[sample, pred_idx] # (num_matched, 2)
    #         t = targets[sample]["pairs"][tgt_idx].to(p.device) # (num_matched, 2), same device and dtype
    #         sample_loss = ((p - t) ** 2).mean()
    #         per_sample_losses.append(sample_loss)

    #     if len(per_sample_losses) == 0:
    #         return {"recon": pred_pairs.sum() * 0.0}

    #     loss = torch.stack(per_sample_losses).mean()
    #     return {"recon": loss}


    # def diagonal_loss(self, outputs: dict, targets, indices: list, num_pairs):
    #     pred_pairs = outputs["pred_pairs"]  # (B, num_queries, 2)
    #     n_preds = pred_pairs.shape[1]
    #     sample_losses = []

    #     for sample, (pred_idx, _) in enumerate(indices):
    #         mask = torch.ones(n_preds, device=pred_pairs.device, dtype=torch.bool)
    #         mask[pred_idx] = False  # discard all matched predictions
    #         unmatched = pred_pairs[sample][mask]  # (n_unmatched, 2)
    #         if unmatched.numel() == 0:
    #             continue

    #         # distance to diagonal: d - b
    #         diff = unmatched[:, 1] - unmatched[:, 0]
    #         # squared distance per sample (mean over unmatched)
    #         sample_loss = (diff ** 2).mean()
    #         sample_losses.append(sample_loss)

    #     if len(sample_losses) == 0:
    #         print("[WARNING] No unmatched predictions for diagonal loss")
    #         return {"diag": pred_pairs.sum() * 0.0}

    #     # equal weight per sample
    #     loss = torch.stack(sample_losses).mean()
    #     return {"diag": loss}


    # ==> Other <== 
    def wasserstein2_loss(self, outputs, targets, indices, num_pairs):
        pred_pairs = outputs["pred_pairs"]
        loss_sum = pred_pairs.new_tensor(0.0)
        for b in range(pred_pairs.shape[0]):
            P = pred_pairs[b]
            T = targets[b]["pairs"].to(P.device)
            N = P.shape[0]
            M = T.shape[0]
            if N == 0 and M == 0:
                continue
            C = torch.cdist(T, P, p=2) ** 2 if (M > 0 and N > 0) else pred_pairs.new_zeros((M, N))
            td = 0.5 * (T[:, 1] - T[:, 0]) ** 2 if M > 0 else pred_pairs.new_zeros((0,))
            pd = 0.5 * (P[:, 1] - P[:, 0]) ** 2 if N > 0 else pred_pairs.new_zeros((0,))
            S = M + N
            big = torch.full((S, S), 0.0, device=P.device)
            if M > 0 and N > 0:
                big[:M, :N] = C
            if M > 0:
                big[:M, N:N+M] = torch.diag(td)
            if N > 0:
                big[M:M+N, :N] = torch.diag(pd)
            row, col = linear_sum_assignment(big.detach().cpu().numpy())
            row = torch.as_tensor(row, device=P.device)
            col = torch.as_tensor(col, device=P.device)
            cost = big[row, col].sum()
            loss_sum += cost
        loss = loss_sum / num_pairs
        return {"w2": loss}
    
    
    def get_loss(self, 
                 loss_name: str, 
                 outputs: dict, 
                 targets: list, 
                 indices: list, 
                 num_pairs: int
                 ):
        loss2func = {
            "existence": self.existence_loss, 
            "recon": self.reconstruction_loss,
            "diag": self.diagonal_loss,
            "w2": self.wasserstein2_loss,
            }
        return loss2func[loss_name](outputs, targets, indices, num_pairs)



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
