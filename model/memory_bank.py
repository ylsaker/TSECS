import torch
import torch.nn as nn
from model.cross_attention import CrossAttention

class MemoryBank:
    def __init__(self, momentum_weight, sim_thresold, dif_thresold):
        self.memory_bank = None
        self.momentum_weight = momentum_weight
        self.sim_thresold = sim_thresold
        self.dif_thresold = dif_thresold

    def get_memory_bank(self,):
        return self.memory_bank

    def nearest_center_normeuc(self, prototypes):
        # n, dim; cluster, dim
        #   x       prototype
        # n, dim
        x_tmp = nn.functional.normalize(self.memory_bank, dim=-1, p=2)
        # cluster, dim
        prototypes_tmp = nn.functional.normalize(prototypes, dim=-1, p=2)
        dist = x_tmp @ prototypes_tmp.t()
        nearest_value, nearest_idx = (dist).max(dim=-1)
        return {'nearest':(nearest_value, nearest_idx)}

    def update_cluster_center(self, new_cluster_center):
        if self.memory_bank == None:
            self.memory_bank = new_cluster_center.detach()
        else:
            result = self.nearest_center_normeuc(new_cluster_center)
            nearest_value, nearest_idx = result['nearest']
            new_add_dif_value = []
            for i, idx in enumerate(nearest_idx):
                if nearest_value[i] > self.sim_thresold:
                    self.memory_bank[i] = self.momentum_weight * self.memory_bank[i] + (1-self.momentum_weight) * new_cluster_center[idx]
                if nearest_value[i] < self.dif_thresold:
                    new_add_dif_value.append(new_cluster_center[idx].detach())
            if new_add_dif_value:
                self.memory_bank = torch.cat([self.memory_bank, torch.stack(new_add_dif_value)], dim=0)
