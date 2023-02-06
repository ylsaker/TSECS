from torch import nn
import torch

class SVD_(nn.Module):
    def __init__(self, basis_num=32, basis_thredshold=100):
        super(SVD_, self).__init__()
        self.basis_num = basis_num
        self.basis_thredshold = basis_thredshold
        
    def forward(self, x):
        _, lan, res = torch.linalg.svd(x, False)
        lan = lan.view(-1)
        ret_num = (lan > self.basis_thredshold).type(torch.long).sum()
        return res[:ret_num], lan.view(-1)[:ret_num]