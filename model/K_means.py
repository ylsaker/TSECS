import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from model.memory_bank import *
from model.cross_attention import *

class Kmeans(nn.Module):
    def __init__(self, n_clusters=20, max_iter=100, metric='euc'):
        super(Kmeans, self).__init__()
        self.n_cluster = n_clusters
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.cls_onehot = torch.eye(n_clusters)
        self.nearest_center = self.nearest_center_cos if metric == 'cos' else self.nearest_center_euc

    # @torch.no_grad()
    def forward(self, x):
        self.cls_onehot = self.cls_onehot.to(x.device)
        # prototypes = torch.randn([self.n_cluster, x.size(-1)]).to(x.device)
        prototypes = x[:self.n_cluster]
        for _ in range(self.max_iter):
            nearst_idx = self.nearest_center(x, prototypes)
            prototypes_new = self.update_center(x, nearst_idx)
            variation = ((prototypes_new - prototypes)**2).mean()
            prototypes = prototypes_new
            if variation < 1e-4:
                break
        return prototypes, nearst_idx

    def nearest_center_cos(self, x, prototypes):
        # print(x.size())
        # exit()
        num = x.size(0)
        # n, dim; cluster, dim
        #   x       prototype
        # n, cluster, dim
        x_tmp = x/((x**2).sum(-1, True).sqrt()+1e-7)
        # n, cluster, dim
        prototypes_tmp = prototypes/((prototypes**2).sum(-1, True).sqrt()+1e-7)
        sim = x_tmp@prototypes_tmp.t()
        nearest_value, nearst_idx = sim.max(dim=-1)
        return nearst_idx

    def nearest_center_euc(self, x, prototypes):
        num = x.size(0)
        # n, dim; cluster, dim
        #   x       prototype
        # n, cluster, dim
        x_tmp = x.unsqueeze(1).expand(-1, self.n_cluster, -1)
        # n, cluster, dim
        prototypes_tmp = prototypes.unsqueeze(0).expand(num, -1, -1)
        dist = ((x_tmp - prototypes_tmp)**2).sum(-1).sqrt()
        nearest_value, nearst_idx = (-dist).max(dim=-1)
        return nearst_idx

    def update_center(self, x, nearst_idx):
        # clusters, num
        one_hot = F.one_hot(nearst_idx, num_classes=self.n_cluster).t().type(torch.float32)
        # clusters, dim
        prototypes = torch.matmul(one_hot, x) / (one_hot.sum(1, True) + 1e-6)
        return prototypes


class KmeansUnitspace(Kmeans):
    def __init__(self, n_clusters=20, max_iter=100, metric='euc'):
        super(KmeansUnitspace, self).__init__(n_clusters, max_iter, metric)
        self.nearest_center = self.nearest_center_normeuc
        self.bn = nn.BatchNorm1d(640)
        
    def nearest_center_normeuc(self, x, prototypes):
        num = x.size(0)
        # n, dim; cluster, dim
        #   x       prototype
        # n, cluster, dim
        x_tmp = nn.functional.normalize(x.unsqueeze(1).expand(-1, self.n_cluster, -1), p=2, dim=-1)
        # n, cluster, dim
        prototypes_tmp = nn.functional.normalize(prototypes.unsqueeze(0).expand(num, -1, -1), p=2, dim=-1)
        dist = ((x_tmp - prototypes_tmp)**2).sum(-1).sqrt()
        nearest_value, nearst_idx = (-dist).max(dim=-1)
        return nearst_idx

    def forward(self, x):
        self.cls_onehot = self.cls_onehot.to(x.device)
        prototypes = x[:self.n_cluster]
        for _ in range(self.max_iter):
            nearst_idx = self.nearest_center(x, prototypes)
            prototypes_new = self.update_center(x, nearst_idx)
            variation = ((prototypes_new - prototypes) ** 2).mean()
            prototypes = prototypes_new
            if variation < 1e-5:
                break
        return prototypes, nearst_idx


class KmeansUnitspaceV2(KmeansUnitspace):
    def __init__(self, n_clusters=20, max_iter=200, metric='euc'):
        super(KmeansUnitspaceV2, self).__init__(n_clusters, max_iter, metric)
    
    def nearest_center_normeuc(self, x, prototypes):
        # n, dim; cluster, dim
        #   x       prototype
        # n, dim
        x_tmp = nn.functional.normalize(x, dim=-1, p=2)
        # cluster, dim
        prototypes_tmp = nn.functional.normalize(prototypes, dim=-1, p=2)
        dist = x_tmp @ prototypes_tmp.t()
        nearest_value, nearst_idx = (dist).max(dim=-1)
        return nearst_idx

    def forward(self, x):
        self.cls_onehot = self.cls_onehot.to(x.device)
        prototypes = x[:self.n_cluster]
        for _ in range(self.max_iter):
            nearst_idx = self.nearest_center_normeuc(x, prototypes)
            prototypes_new = self.update_center(x, nearst_idx)
            variation = ((prototypes_new - prototypes) ** 2).mean()
            prototypes = prototypes_new
            if variation < 1e-5:
                break
        return prototypes, nearst_idx[0]


class DynamicKmeans(KmeansUnitspaceV2):
    def __init__(self, n_clusters=20, max_iter=200, metric='euc', basis_thredshold=100):
        super(DynamicKmeans, self).__init__(n_clusters=n_clusters, max_iter=max_iter, metric=metric)
        self.basis_thredshold = basis_thredshold
    
    def get_lans(self, x):
        _, lan, res = torch.linalg.svd(x, True) # U S V^H
        lan = lan.view(-1)
        sv_means = lan.mean()
        return lan, sv_means
    
    def forward(self, x):
        self.cls_onehot = self.cls_onehot.to(x.device)
        lan, sv_means = self.get_lans(x)
        self.n_cluster = (lan > self.basis_thredshold).type(torch.long).sum()
        prototypes = x[:self.n_cluster]
        for _ in range(self.max_iter):
            nearst_idx = self.nearest_center_normeuc(x, prototypes)
            prototypes_new = self.update_center(x, nearst_idx)
            variation = ((prototypes_new - prototypes) ** 2).mean()
            prototypes = prototypes_new
            if variation < 1e-5:
                break
        return prototypes, sv_means
    
    
class DynamicKmeansV2(DynamicKmeans):
    def __init__(self, n_clusters=20, max_iter=200, metric='euc', basis_thredshold=100):
        super(DynamicKmeansV2, self).__init__(n_clusters, max_iter, 
                                              metric, basis_thredshold)

    def get_lans(self, x):
        _, lan, res = torch.linalg.svd(x, True)
        lan = lan.view(-1)
        sv_means = lan.mean()
        lan = torch.sigmoid((lan - lan.mean()) / lan.var().sqrt())
        lan = lan / lan.max()
        return lan, sv_means


class DynamicKmeansV3(DynamicKmeansV2):
    def __init__(self, n_clusters=20, max_iter=200, metric='euc', basis_thredshold=100, largest_num=1000, scale=1):
        super(DynamicKmeansV3, self).__init__(n_clusters, max_iter, 
                                              metric, basis_thredshold)
        self.crossattention = CrossAttention(640, 8, largest_num, False, scale)
        self.largest_num = largest_num

    def forward(self, x):
        self.cls_onehot = self.cls_onehot.to(x.device)
        lan, sv_means = self.get_lans(x)
        self.n_cluster = (lan > self.basis_thredshold).type(torch.long).sum()
        prototypes = x[:self.n_cluster]
        prototypes = self.crossattention(prototypes)
        for _ in range(self.max_iter):
            nearst_idx = self.nearest_center_normeuc(x, prototypes)
            prototypes_new = self.update_center(x, nearst_idx)
            variation = ((prototypes_new - prototypes) ** 2).mean()
            prototypes = prototypes_new
            if variation < 1e-5:
                break
        if self.crossattention.memory_bank == None or self.crossattention.memory_bank.size(0) > self.largest_num:
            self.crossattention.memory_bank = prototypes.detach()
        else:
            self.crossattention.memory_bank = torch.cat([self.crossattention.memory_bank, prototypes.detach()], dim=0)
        return prototypes, sv_means


class DynamicKmeansV4(DynamicKmeansV2):
    def __init__(self, n_clusters=20, max_iter=200, metric='euc', basis_thredshold=100, largest_num=1000, scale=1):
        super(DynamicKmeansV4, self).__init__(n_clusters, max_iter, 
                                              metric, basis_thredshold)
        self.crossattention = CrossAttention(640, 8, True, scale)

    def forward(self, x):
        self.cls_onehot = self.cls_onehot.to(x.device)
        lan, sv_means = self.get_lans(x)
        self.n_cluster = (lan > self.basis_thredshold).type(torch.long).sum()
        prototypes = x[:self.n_cluster]
        prototypes = self.crossattention(prototypes)
        for _ in range(self.max_iter):
            nearst_idx = self.nearest_center_normeuc(x, prototypes)
            prototypes_new = self.update_center(x, nearst_idx)
            variation = ((prototypes_new - prototypes) ** 2).mean()
            prototypes = prototypes_new
            if variation < 1e-5:
                break
        self.crossattention.memory_bank = prototypes.detach()
        return prototypes, sv_means

class DynamicKmeansV5(DynamicKmeansV2):
    def __init__(self, n_clusters=20, max_iter=200, metric='euc', basis_thredshold=100, largest_num=1000, scale=1):
        super(DynamicKmeansV5, self).__init__(n_clusters, max_iter, 
                                              metric, basis_thredshold)
        self.crossattention = CrossAttentionV2(640, 8, False, scale)