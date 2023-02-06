from torch import nn
from model.gaussian_layer import GaussianLayerV2
from model.BaseModel import BaseModelFSL
import numpy as np
import torch
import math


class BaseSimilarityEncoder(BaseModelFSL):
    """
    Encoding the image-to-class similarity matrix(2-D) to similarity pattern(1-D).
    """
    def __init__(self, k_size=1, sigma=0.5, iter=2, drop_rate=0.001):
        super(BaseSimilarityEncoder, self).__init__()
        self.k_size = k_size
        self.sigma = sigma
        self.iter = iter
        self.pool_size = 2
        self.gauss_kernel = None
        self.drop_rate = drop_rate
        self.pooling = nn.MaxPool3d([1, self.pool_size, self.pool_size], stride=[1, self.pool_size, self.pool_size])

    def get_gauss_kernel(self, device):
        row = 2 * self.k_size + 1
        col = 2 * self.k_size + 1
        A = []
        sigma = torch.tensor(self.sigma, dtype=torch.float32)
        for i in range(row):
            r = []
            for j in range(col):
                fenzi = (i + 1 - self.k_size - 1) ** 2 + (j + 1 - self.k_size - 1) ** 2
                r.append(torch.exp(-fenzi / (2 * sigma)) / (2 * np.pi * sigma))
            A.append(torch.stack(r, 0))
        A = torch.stack(A, 0)
        A = A / A.sum()
        gauss_kernel = A.view(1, 1, 1, 2 * self.k_size + 1, 2 * self.k_size + 1).type(torch.float32).to(device)
        self.gauss_kernel = gauss_kernel

    def encoder(self, x, layer):
        # filter
        encoding = torch.nn.functional.conv3d(x, self.gauss_kernel.detach(), stride=[1, 1, 1], padding=[0, self.k_size, self.k_size])
        qm, pm, psize_, H_, W_ = encoding.size()
        encoding = encoding.view(qm, 1, -1, H_, W_)
        encoding = self.pooling(encoding)
        return encoding

    def pass_through(self, x, layer):
        # if self.drop_rate > 0:
            # x = nn.functional.dropout(x, p=self.drop_rate ** layer, training=self.training, inplace=True)
        return x

    def forward_(self, patterns):
        qm, q_size, way, p_size_shot = patterns.size()
        qh = qw = ph = pw = int(math.sqrt(q_size))
        shot = p_size_shot // (ph * pw)
        # generate the Gaussian kernel
        if self.gauss_kernel is None:
            self.get_gauss_kernel(patterns.device)
        encodings = torch.stack([patterns[:, :, i, :].permute(0, 2, 1).view(qm, 1, shot * ph * pw, qh, qw) for i in range(way)], 0)
        for j in range(self.iter):
            encoding_list = []
            for i in range(way):
                encoding = self.encoder(encodings[i], j)
                encoding_list.append(encoding)
            # process similarity features
            encodings = self.pass_through(torch.stack(encoding_list, 0), j + 1)
        # way, qm, 1, shot*H*W, 1, 1 -> way, qm, shot, H*W -> qm, shot, way, H*W
        patterns = encodings.view(way, qm, -1, qh * qw).permute(1, 2, 0, 3)
        logits = patterns.sum([-1, -3])
        return {'patterns': patterns, 'logits': logits}


class SimilarityEncoderIMSE(BaseSimilarityEncoder):
    def __init__(self, k_size=1, sigma=0.5, iter=2):
        super(SimilarityEncoderIMSE, self).__init__(k_size, sigma, iter)
        self.pool_size = 2


class SimilarityEncoderV2(BaseSimilarityEncoder):
    """
    Args:
        nn (_type_): _description_
    """
    def __init__(self, k_size=1, sigma=0.5):
        super(SimilarityEncoderV2, self).__init__(k_size, sigma)

    def forward_(self, patterns):
        qm, q_size, way, p_size_shot = patterns.size()
        qh = qw = int(math.sqrt(q_size))
        # generate the Gaussian kernel
        if self.gauss_kernel is None:
            self.get_gauss_kernel(patterns.device)
        encodings = torch.stack([patterns[:, :, i, :].permute(0, 2, 1).view(qm, 1, p_size_shot, qh, qw) for i in range(way)], 0)
        # iteratively encoding
        for j in range(2):
            encoding_list = []
            for i in range(way):
                encoding = self.encoder(encodings[i], j)
                encoding_list.append(encoding)
            # process similarity features
            encodings = self.pass_through(torch.stack(encoding_list, 0), j)
        # way, qm, 1, shot*H*W, 1, 1 -> way, qm, shot, H*W -> qm, shot, way, H*W
        patterns = encodings.view(way, qm, -1, p_size_shot).permute(1, 2, 0, 3)
        # compute classification scores
        logits = patterns.sum([-1, -3])
        # print(logits[:75]); exit()
        return patterns, logits


class SimilarityEncoderV3(BaseSimilarityEncoder):
    """_summary_
    For IMSE.
    Args:
        BaseSimilarityEncoder (_type_): _description_
    """
    def __init__(self, k_size=1, sigma=0.5, iter=2):
        super(SimilarityEncoderV3, self).__init__(k_size=k_size, sigma=sigma, iter=iter)

    def encoder(self, encoding, layer):
        # filter
        encoding = torch.nn.functional.conv3d(encoding, self.gauss_kernel.detach(), stride=[1, 1, 1], padding=[0, 0, 0])
        qm, pm, psize_, H_, W_ = encoding.size()
        encoding = encoding.view(qm, 1, -1, H_, W_)
        if H_ > 1 and H_ % 2 != 0:
            size_ = H_ + H_ % 2
            encoding = nn.functional.adaptive_avg_pool3d(encoding, [psize_, size_, size_])
        elif H_ > 1:
            encoding = self.pooling(encoding)
        return encoding


class FilterPatternSimilarityEncoder(BaseSimilarityEncoder):
    def __init__(self, k_size=1, sigma=0.5):
        super(FilterPatternSimilarityEncoder, self).__init__(k_size, sigma)

    def encoder(self, encoding, layer):
        # filter
        for _ in range(2):
            encoding = torch.nn.functional.conv3d(encoding, self.gauss_kernel.detach(), stride=[1, 1, 1], padding=[0, 1, 1])
        qm, _, psize_, H_, W_ = encoding.size()
        encoding = encoding.view(qm, 1, -1, H_, W_)
        encoding = self.pooling(encoding)
        h, w = encoding.size()[3:]
        return encoding

    def forward_(self, patterns):
        qm, q_size, way, p_size_shot = patterns.size()
        qh = qw = ph = pw = int(math.sqrt(q_size))
        shot = p_size_shot // (ph * pw)
        # generate the Gaussian kernel
        if self.gauss_kernel is None:
            self.get_gauss_kernel(patterns.device)
        encodings = torch.stack([patterns[:, :, i, :].permute(0, 2, 1).view(qm, 1, shot * ph * pw, qh, qw) for i in range(way)], 0)

        for j in range(2):
            encoding_list = []
            for i in range(way):
                encoding = self.encoder(encodings[i], j)
                encoding_list.append(encoding)
            # process similarity features
            encodings = self.pass_through(torch.stack(encoding_list, 0), j)
        # way, qm, 1, shot*H*W, 1, 1 -> way, qm, shot, H*W -> qm, shot, way, H*W
        patterns = encodings.view(way, qm, -1, qh * qw).permute(1, 2, 0, 3)
        logits = patterns.sum([-1, -3])
        return patterns, logits


class SimilarityPatternBN1D(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self, eps=1):
        """_summary_
        Args:
            eps (_type_, optional): _description_. Defaults to 1e-10.
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.ones([1]))
        self.beta = nn.Parameter(torch.zeros([1]))
        self.eps = eps

    def forward(self, x):
        """_summary_

        Args:
            x (Tensor): (Batchsize, SP-dim, -1)
        """
        mean_ = x.mean([0, 2], True)
        var_ = torch.var(x, [0, 2], keepdim=True)
        x = (x - mean_) / (var_ + self.eps).sqrt() * self.gamma + self.beta
        return x


class SimilarityEncoderBN(BaseSimilarityEncoder):
    def __init__(self, k_size=1, sigma=0.5):
        super(SimilarityEncoderBN, self).__init__(k_size, sigma)
        self.bn = nn.ModuleList([
            nn.Sequential(
                SimilarityPatternBN1D(),
            ),
            nn.Sequential(
                SimilarityPatternBN1D(),
            ),
        ])
        self.sigma = nn.Parameter(torch.tensor(self.sigma, dtype=torch.float32))

    def pass_through(self, x_, layer):
        # way, qm, _, p_size_shot, H, W = x_.size()
        # # way, qm, 1, p_size_shot, H, W -> qm, _, p_size_shot, way, H, W -> qm, p_size_shot, way*H*W
        # x = x_.permute(1, 2, 3, 0, 4, 5).contiguous().view(qm, p_size_shot, way*H*W)
        # # qm, _, p_size_shot, way, H, W -> way, qm, 1, p_size_shot, H, W
        # x = self.bn[layer](x).view(qm, _, p_size_shot, way, H, W).permute(3, 0, 1, 2, 4, 5)
        return x_

    def get_gauss_kernel(self):
        row = 2 * self.k_size + 1
        col = 2 * self.k_size + 1
        A = []
        for i in range(row):
            r = []
            for j in range(col):
                fenzi = (i + 1 - self.k_size - 1) ** 2 + (j + 1 - self.k_size - 1) ** 2
                r.append(torch.exp(-fenzi / (2 * self.sigma)) / 
                         (2 * np.pi * self.sigma))
            A.append(torch.stack(r, 0))
        A = torch.stack(A, 0)
        A = A / A.sum()
        gauss_kernel = A.view(1, 1, 1, 2 * self.k_size + 1, 2 * self.k_size + 1)
        return gauss_kernel.detach()

    def encoder(self, encoding, layer):
        # filter
        encoding = torch.nn.functional.conv3d(encoding, self.get_gauss_kernel(), stride=[1, 1, 1],  padding=[0, 1, 1])
        qm, pm, psize_, H_, W_ = encoding.size()
        encoding = encoding.view(qm, 1, -1, H_, W_)
        encoding = self.pooling(encoding)
        return encoding

    def forward_(self, patterns):
        qm, q_size, way, p_size_shot = patterns.size()
        qh = qw = int(math.sqrt(q_size))
        # generate the Gaussian kernel
        encodings = torch.stack([patterns[:, :, i, :].permute(0, 2, 1).view(qm, 1, p_size_shot, qh, qw) for i in range(way)], 0)
        for j in range(2):
            encoding_list = []
            for i in range(way):
                encoding = self.encoder(encodings[i], j)
                encoding_list.append(encoding)
            # process similarity features
            encodings = self.pass_through(torch.stack(encoding_list, 0), j)
        # way, qm, 1, shot*H*W, 1, 1 -> way, qm, shot, H*W -> qm, shot, way, H*W
        patterns = encodings.view(way, qm, -1, p_size_shot).permute(1, 2, 0, 3)
        logits = patterns.sum([-1, -3])
        return patterns, logits


class SimilarityEncoderASP(SimilarityEncoderBN):
    def __init__(self, k_size=1, sigma=0.5, template_num=5):
        super(SimilarityEncoderASP, self).__init__(k_size, sigma)
        self.gaussian_layers = nn.ModuleList([GaussianLayerV2(template_num=template_num),
                                              GaussianLayerV2(template_num=template_num)])
        self.template_num = template_num

    def pass_through(self, x_, layer):
        way, qm, _, p_size_shot, H, W = x_.size()
        # way, qm, 1, p_size_shot, H, W -> qm, _, p_size_shot, way, H, W -> 
        # qm, p_size_shot, way*H*W
        x = x_.permute(1, 2, 3, 0, 4, 5).contiguous().view(qm, p_size_shot,
                                                           way * H * W)
        # qm, _, p_size_shot, way, H, W -> way, qm, 1, p_size_shot, H, W
        x = x.view(qm, _, p_size_shot, way, H, W).permute(3, 0, 1, 2, 4, 5)
        return x

    def fusion_block(self, res):
        return res.max(1, True)[0]

    def encoder(self, x, layer):
        encoding = self.gaussian_layers[layer](x)
        encoding = self.fusion_block(encoding)

        qm, pm, psize_, H_, W_ = encoding.size()
        encoding = encoding.view(qm, 1, -1, H_, W_)
        encoding = torch.relu(self.pooling(encoding))
        return encoding
