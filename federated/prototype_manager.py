"""
Server-Side Global Class Prototype Manager for MALFSIL.
Aggregates class feature representations across non-IID clients to preserve global decision boundaries.
"""

from typing import Dict, List, Tuple, Any
import torch


class GlobalPrototypeManager:
    """
    Maintains and updates running global class prototypes by sample-weighted averaging
    of client-computed latent centroids.
    """

    def __init__(self):
        # class_id -> prototype_vector (torch.Tensor)
        self.global_prototypes: Dict[int, torch.Tensor] = {}
        # class_id -> cumulative sample count used for aggregation
        self.prototype_counts: Dict[int, int] = {}

    def update_prototypes(self, client_prototypes_list: List[Dict[int, Tuple[torch.Tensor, int]]]) -> None:
        """
        Aggregate local prototypes from multiple clients:
        client_prototypes: class_id -> (mean_tensor, sample_count)
        """
        class_sums: Dict[int, torch.Tensor] = {}
        class_counts: Dict[int, int] = {}

        for client_proto_dict in client_prototypes_list:
            for cls_id, (proto_tensor, count) in client_proto_dict.items():
                if count <= 0:
                    continue
                if cls_id not in class_sums:
                    class_sums[cls_id] = proto_tensor.clone() * count
                    class_counts[cls_id] = count
                else:
                    class_sums[cls_id] += proto_tensor * count
                    class_counts[cls_id] += count

        # Compute new weighted centroids
        for cls_id, total_weighted_sum in class_sums.items():
            tot_count = class_counts[cls_id]
            if tot_count > 0:
                new_proto = total_weighted_sum / tot_count
                if cls_id not in self.global_prototypes:
                    self.global_prototypes[cls_id] = new_proto.detach().cpu()
                    self.prototype_counts[cls_id] = tot_count
                else:
                    # Exponential running update
                    old_proto = self.global_prototypes[cls_id]
                    old_c = self.prototype_counts[cls_id]
                    combined_proto = (old_proto * old_c + new_proto.detach().cpu() * tot_count) / (old_c + tot_count)
                    self.global_prototypes[cls_id] = combined_proto
                    self.prototype_counts[cls_id] = old_c + tot_count

    def get_prototypes(self) -> Dict[int, torch.Tensor]:
        """Return global prototype dictionary."""
        return {k: v.clone() for k, v in self.global_prototypes.items()}

    def state_dict(self) -> Dict[str, Any]:
        return {
            "global_prototypes": {k: v.cpu() for k, v in self.global_prototypes.items()},
            "prototype_counts": self.prototype_counts,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        raw_protos = state.get("global_prototypes", {})
        self.global_prototypes = {int(k): v for k, v in raw_protos.items()}
        raw_counts = state.get("prototype_counts", {})
        self.prototype_counts = {int(k): v for k, v in raw_counts.items()}
