import torch
from torch import nn
from scipy.optimize import linear_sum_assignment
from torch_linear_assignment import batch_linear_assignment, assignment_to_indices


class HungarianMatcherPersistence(nn.Module):

    def __init__(self, 
                 cost_coord: float, 
                 cost_no_exist: float, 
                 use_exist_in_assignment: bool, 
                 use_cpu: bool):
        """
        Params:
            cost_coord: Weight for the squared 
            cost_no_exist: Weight for the existence penalty term
        """
        super().__init__()
        self.cost_coord = cost_coord
        self.cost_no_exist = cost_no_exist
        self.use_exist_in_assignment = use_exist_in_assignment # use existence prob
        self.use_cpu = use_cpu
        assert cost_coord != 0 or cost_no_exist != 0 or not use_exist_in_assignment

    @torch.no_grad()
    def forward(self, outputs: dict, targets: list) -> tuple:
        """
        Params:
            outputs: dict with keys:
                "pred_pairs": Tensor of shape [batch_size, num_preds, 2]
                    Predicted persistence pairs (birth, death)
                "pred_exist": Tensor of shape [batch_size, num_preds]
                    Logits predicting existence probability for each pair

            targets: list (len = batch_size), each element a dict with:
                "pairs": Tensor of shape [num_gt, 2]
                    Ground-truth persistence pairs (birth, death)

        Returns:
            A list of size batch_size, containing tuples (pred_idx, gt_idx) where:
                - pred_idx: indices of selected predictions
                - gt_idx: sorted indices of corresponding ground truth pairs
        """
        bs = outputs["pred_pairs"].shape[0]
        out_pairs = outputs["pred_pairs"] # (batch_size, num_preds, 2)
        out_exist_logits = outputs.get("pred_exist", None) # (batch_size, num_preds) can be None is head disabled
        device = out_pairs.device

        indices = []
        for sample in range(bs):
            gt_pairs = targets[sample]["pairs"]  # [M_sample, 2]
            pred_pairs = out_pairs[sample]       # [N, 2]
            M_sample = gt_pairs.shape[0]
            # Compute squared L2 distance between gt pairs and predicted pairs
            cost_reg = torch.cdist(gt_pairs, pred_pairs, p=2) ** 2  # [M_sample, N]

            cost_exist = torch.zeros_like(cost_reg)
            if self.use_exist_in_assignment and (out_exist_logits is not None):
                existence_logits = out_exist_logits[sample]  # [N]
                existence_prob = torch.sigmoid(existence_logits).unsqueeze(0).repeat(M_sample, 1) # (M_sample, N)
                cost_exist = self.cost_no_exist * (1.0 - existence_prob)
            
            cost = self.cost_coord * cost_reg + cost_exist # (M_sample, N)
            
            if self.use_cpu:
                gt_idx, pred_idx = linear_sum_assignment(cost.cpu().numpy()) # row_indices, col_indices
            else:
                assignment = batch_linear_assignment(cost.unsqueeze(0))
                gt_idx, pred_idx = assignment_to_indices(assignment)
                gt_idx = gt_idx.squeeze(0)
                pred_idx = pred_idx.squeeze(0)
            
            indices.append((
                torch.as_tensor(pred_idx, dtype=torch.long, device=device),
                torch.as_tensor(gt_idx, dtype=torch.long, device=device)
            ))
        return indices