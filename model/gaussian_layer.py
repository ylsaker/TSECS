import torch
from torch import nn
from numpy import pi as PI

class GaussianLayer(nn.Module):
    def __init__(self, kernel_size=1, template_num=5):
        super(GaussianLayer, self).__init__()
        self.k = kernel_size
        self.sigma_parameters = nn.Parameter(torch.tensor(torch.ones([template_num])))
        self.gaussian_kernels = None
        self.get_params()

    def get_params(self):
        row = 2 * self.k + 1
        col = 2 * self.k + 1
        mat = []
        for i in range(row):
            r = []
            for j in range(col):
                fenzi = (i + 1 - self.k - 1) ** 2 + (j + 1 - self.k - 1) ** 2
                r.append(fenzi)
            mat.append(r)
        self.mat = torch.tensor(mat, dtype=torch.float64, requires_grad=False).view(1, 1, 1, 2 * self.k + 1,
                                                                                    2 * self.k + 1)

    def get_gauss_kernel(self):
        kernel_list = []
        for s in self.sigma_parameters:
            sigma = 2 * (nn.functional.relu(s / self.sigma_parameters.norm()) + 1e-5)
            gauss_kernel = ((-self.mat.to(sigma.device)) / sigma).exp() / (PI * sigma)
            gauss_kernel = gauss_kernel / gauss_kernel.sum()
            kernel_list.append(gauss_kernel)
        kernel_list = torch.cat(kernel_list, 0).type(torch.float32)
        self.gaussian_kernels = kernel_list

    def forward(self, x):
        res = torch.nn.functional.conv3d(x, self.gaussian_kernels, stride=1, padding=[0, self.k, self.k])
        return res


class GaussianLayerV2(GaussianLayer):
    def __init__(self, kernel_size=1, template_num=5):
        super(GaussianLayerV2, self).__init__(kernel_size, template_num)
        self.gaussian_kernels = None

    def get_gauss_kernel(self):
        kernel_list = []
        for s in self.sigma_parameters:
            sigma = 2 * (torch.sigmoid(s / self.sigma_parameters.norm()) + 1e-5)
            gauss_kernel = ((-self.mat.to(sigma.device)) / sigma).exp() / (PI*sigma)
            gauss_kernel = gauss_kernel / gauss_kernel.sum()
            kernel_list.append(gauss_kernel)
        kernel_list = torch.cat(kernel_list, 0).type(torch.float32)
        self.gaussian_kernels = kernel_list

    def forward(self, x):
        self.get_gauss_kernel()
        res = torch.nn.functional.conv3d(x, self.gaussian_kernels, stride=1, padding=[0, self.k, self.k])
        return res