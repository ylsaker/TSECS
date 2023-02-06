import torch

import torch
from PIL import ImageEnhance
import time
from tqdm import tqdm
from torch import nn
from torch.nn.utils.weight_norm import WeightNorm
import numpy as np
transformtypedict=dict(Brightness=ImageEnhance.Brightness, Contrast=ImageEnhance.Contrast, Sharpness=ImageEnhance.Sharpness, Color=ImageEnhance.Color)


class ImageJitter(object):
    def __init__(self, transformdict):
        self.transforms = [(transformtypedict[k], transformdict[k]) for k in transformdict]


    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha*(randtensor[i]*2.0 -1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')

        return out

def cal_cov_loss(cov_source, cov_target):
    cov_source = torch.stack([get_cov(c) for c in torch.stack(cov_source, 0).to(0)], 0)
    cov_target = torch.stack([get_cov(c) for c in torch.stack(cov_target, 0).to(0)], 0)
    loss = ((cov_source-cov_target)**2).sum([-1, -2]).mean(0)
    return loss


def cal_FSL_cov_loss(cov_source):
    cov_source = torch.stack([get_cov(c) for c in torch.stack(cov_source, 0).to(cov_source[0].device)], 0)
    lam = torch.stack([torch.trace(cov_s)/cov_source.size(-1) for cov_s in cov_source], 0)
    eye = torch.eye(cov_source.size(-1)).repeat(cov_source.size(0), 1, 1).to(cov_source.device)
    iden = eye*lam.view(lam.size(0), 1, 1)
    loss = ((cov_source-iden)**2).sum([-1, -2]).mean(0)
    return loss

# def cal_cov_loss(cov_source, cov_target):
#     cov_source = torch.stack(cov_source, 0).to(0)
#     cov_target = torch.stack(cov_target, 0).to(0)
#     loss = ((cov_source-cov_target)**2).sum([-1, -2]).mean(0)
#     return loss

def get_cov(x):
    mean = x.mean(0, True)
    va = x - mean
    cov = torch.matmul(va.t(), va)/(x.size(1)-1)+0.01*torch.eye(n=x.size(-1)).type(x.dtype).cuda(x.device)
    return cov

def get_cov_unbias(imse_set_source, imse_set_target):
    loss_list = []
    qm = imse_set_source[0].size(0)
    bound = qm // 5
    idxs = list(range(qm))
    for class_idx in range(len(imse_set_source)):
        unbias_idx = idxs[0:class_idx*bound]+idxs[class_idx*bound+bound:qm]
        tmp_source_sp = imse_set_source[class_idx][unbias_idx]
        mean_s = tmp_source_sp.mean(0, True)
        mean_q = imse_set_target[class_idx].mean(0, True)
        sub_sq = mean_s-mean_q
        cov_s = get_cov(tmp_source_sp)
        cov_t = get_cov(imse_set_target[class_idx])
        # term3 = torch.mm(torch.mm(sub_sq, cov_s.inverse()), sub_sq.t()) - sub_sq.size(-1)
        loss_list.append(0.5*(KL_loss(cov_s, cov_t) + (sub_sq**2).sum()))
    loss = torch.stack(loss_list, 0).mean()
    # print(loss)
    return loss

def get_window_mask():
    H = W = 10
    idxs = torch.zeros([H, W])
    mask = []
    # ks = self.kernel_size
    ks = 1
    for i in range(H):
        for j in range(W):
            i_tmp = idxs.clone()
            i_tmp[max(0, i - ks):min(H-1, i+ks)+1, max(0, j - ks):min(H-1, j+ks)+1] += 1

            mask.append(i_tmp.view(-1))
    mask = torch.stack(mask, 0)
    return mask

# def cal_cov_loss(encoding_source, cov_target):
#     msk = get_window_mask().to(0)
#     encoding_source = torch.stack(encoding_source, 0).to(0)
#     mean = encoding_source.mean(1, True)
#     va = encoding_source - mean
#     cov_source = torch.bmm(va.permute(0, 2, 1), va)/\
#           (encoding_source.size(1) - 1)\
#                  +0.01*torch.eye(n=encoding_source.size(-1)).type(encoding_source.dtype).cuda(encoding_source.device)
#     lam = torch.stack([torch.trace(cov_s)/cov_source.size(-1) for cov_s in cov_source], 0)
#     eye = torch.eye(cov_source.size(-1)).repeat(cov_source.size(0), 1, 1).to(cov_source.device)
#     iden = eye*lam.view(lam.size(0), 1, 1) + (msk-eye)*cov_source.abs()
#     # print(cov_source[0], iden[0])
#     # exit()
#     loss = ((cov_source-iden)**2).sum([-1, -2]).mean(0)
#     return loss

# def cal_cov_loss(encoding_source, cov_target):
#     encoding_source = torch.stack(encoding_source, 0).to(0)
#     mean = encoding_source.mean(1, True)
#     va = encoding_source - mean
#     cov_source = torch.bmm(va.permute(0, 2, 1), va)/\
#           (encoding_source.size(1) - 1)\
#                  +0.01*torch.eye(n=encoding_source.size(-1)).type(encoding_source.dtype).cuda(encoding_source.device)
#     lam = torch.stack([torch.trace(cov_s)/cov_source.size(-1) for cov_s in cov_source], 0)
#     eye = torch.eye(cov_source.size(-1)).repeat(cov_source.size(0), 1, 1).to(cov_source.device)
#     iden = eye*lam.view(lam.size(0), 1, 1)
#     # print(cov_source[0], iden[0])
#     # exit()
#     loss = ((cov_source-iden)**2).sum([-1, -2]).mean(0)
#     return loss

def cal_cov_classloss(encoding_source, cov_target):
    msk = get_window_mask().to(0)
    encoding_source = torch.stack([es[i*15:i*15+15, :] for i, es in enumerate(encoding_source)], 0).to(0)
    mean = encoding_source.mean(1, True)
    va = encoding_source - mean
    cov_source = torch.bmm(va.permute(0, 2, 1), va)/\
          (encoding_source.size(1) - 1)\
                 +0.01*torch.eye(n=encoding_source.size(-1)).type(encoding_source.dtype).cuda(encoding_source.device)
    lam = torch.stack([torch.trace(cov_s)/cov_source.size(-1) for cov_s in cov_source], 0)
    eye = torch.eye(cov_source.size(-1)).repeat(cov_source.size(0), 1, 1).to(cov_source.device)
    iden = eye*lam.view(lam.size(0), 1, 1) + (msk-eye)*cov_source.abs()
    loss = ((cov_source-iden)**2).sum([-1, -2]).mean(0)
    return loss

def cal_cov_classloss_covdist(encoding_source_all, cov_target):
    msk = get_window_mask().to(0)
    # encoding_source_all --> 5, qm, 100
    # encoding_source --> 5, 15, 100
    encoding_source = torch.stack([es[i*15:i*15+15, :] for i, es in enumerate(encoding_source_all)], 0).to(0)
    # mean --> 5, 1, 100
    mean = encoding_source.mean(1, True)
    va = encoding_source - mean
    cov_source = torch.bmm(va.permute(0, 2, 1), va)/\
          (encoding_source.size(1) - 1)\
                 +0.01*torch.eye(n=encoding_source.size(-1)).type(encoding_source.dtype).cuda(encoding_source.device)
    dist = []

    for i in range(5):
        v = (encoding_source_all[i] - mean[i]).unsqueeze(1)
        # print(v.size(), cov_source[i].size())
        dist.append(torch.matmul(torch.matmul(v, cov_source[i].unsqueeze(0).expand(v.size(1), -1, -1)), v.permute(0, 2, 1)))
    class_dist = -torch.stack(dist, 0).squeeze().t().softmax(-1).log()
    # print(class_dist)
    dist_loss = torch.stack([class_dist[i*15:i*15+15, :].mean() for i in range(5)], 0)

    lam = torch.stack([torch.trace(cov_s)/cov_source.size(-1) for cov_s in cov_source], 0)
    eye = torch.eye(cov_source.size(-1)).repeat(cov_source.size(0), 1, 1).to(cov_source.device)
    iden = eye*lam.view(lam.size(0), 1, 1) + (msk-eye)*cov_source.abs()
    loss = ((cov_source-iden)**2).sum([-1, -2]).mean(0)
    # print(dist)
    return loss, dist_loss.mean()


def KL_loss(cov_source, cov_target):
    loss = ((cov_source-cov_target)**2).sum()
    return loss

def KL_cov_loss(sp_source, sp_target):
    loss = get_cov_unbias(sp_source, sp_target)
    return loss


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear( indim, outdim, bias = False)
        self.class_wise_learnable_norm = False  #See the issue#4&8 in the github
        if self.class_wise_learnable_norm:
            WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm

        if outdim <=200:
            self.scale_factor = 2 #a fixed scale factor to scale the output of cos value into a reasonably large input for softmax, for to reproduce the result of CUB with ResNet10, use 4. see the issue#31 in the github
        else:
            self.scale_factor = 10 #in omniglot, a larger scale factor is required to handle >1000 output classes.

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+ 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized) #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor* (cos_dist)

        return scores


@torch.no_grad()
def meta_test(val_loader, model, epoch_index, best_prec1, opt, train='val'):
    batch_time = AverageMeter()
    ganloss = AverageMeter()
    classloss = AverageMeter()
    top1_m1 = AverageMeter()
    top1_m2 = AverageMeter()
    # switch to evaluate mode
    # model.eval()
    accuracies = []

    end = time.time()
    print('====='*20)
    for episode_index, (query_images, query_targets, query_modal, support_images, support_targets, _) in tqdm(enumerate(val_loader)):

        # Convert query and support images
        query_images = torch.squeeze(query_images.type(torch.FloatTensor))
        support_images = support_images
        input_var1 = query_images.cuda()
        input_var2 = support_images.cuda()

        # Calculate the output
        query_modal = query_modal.cuda()
        query_targets = query_targets.cuda()
        loss_acc = model(query_x=input_var1, support_x=input_var2, query_y=query_targets, query_m=query_modal, train=train)

        # Measure accuracy and record loss
        ganloss.update(loss_acc['gan_loss'].item(), query_images.size(0))
        classloss.update(loss_acc['class_loss'].item(), query_images.size(0))
        top1_m1.update(loss_acc['class_acc'][0][0].item(), query_images.size(0))
        top1_m2.update(loss_acc['class_acc'][1][0].item(), query_images.size(0))
        accuracies.append(loss_acc['class_acc'][1][0].item())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # ============== print the intermediate results ==============#
        if episode_index % opt.print_freq == 0 and episode_index != 0:
            print('Test-({0}): [{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Prec@1 {top1_m1.val:.3f} ({top1_m1.avg:.3f}, {top1_m2.avg:.3f} -> {mean:.3f})'.format(
                epoch_index, episode_index, len(val_loader), batch_time=batch_time, loss=classloss,
                top1_m1=top1_m1, top1_m2=top1_m2, mean=(top1_m1.avg+top1_m2.avg)/2))

            # 'Eposide-({0}): [{1}/{2}]\t'
            # 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            # 'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            # 'classLoss {loss.val:.3f} ({loss.avg:.3f})\t'
            # 'genLoss {ganloss.val:.3f} ({ganloss.avg:.3f})\t'
            # 'Prec@1 {top1_m1.val:.3f} ({top1_m1.avg:.3f}, {top1_m2.avg:.3f} -> {mean:.3f})'

            # print('Test-({0}): [{1}/{2}]\t'
            #       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #       'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
            #       'Prec@1 {top1_m1.val:.3f} ({top1_m1.avg:.3f}, {top1_m2.avg:.3f} -> {mean:.3f})'.format(
            #     epoch_index, episode_index, len(val_loader), batch_time=batch_time, loss=classloss, top1_m1=top1_m1,
            #     top1_m2=top1_m2, mean=(top1_m1.avg+top1_m2.avg)/2) , file=F_txt)

    print(' * Prec@1 {mean:.3f} Best_prec1 {best_prec1:.3f}'.format(mean=(top1_m1.avg+top1_m2.avg)/2, best_prec1=best_prec1))
    # print(' * Prec@1 {mean:.3f} Best_prec1 {best_prec1:.3f}'.format(mean=(top1_m1.avg+top1_m2.avg)/2, best_prec1=best_prec1), file=F_txt)

    return top1_m2.avg, accuracies



def accuracy(output, target, topk=(1,)):
    """Computes the precaccuracy(output, target, topk=(1,3))ision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res[0]

def baselinepp(val_loader, model, epoch_index, best_prec1, opt, train='val'):
    top1_m1 = AverageMeter()
    top1_m2 = AverageMeter()
    # switch to evaluate mode
    # model.eval()
    accuracies = []

    end = time.time()
    print('====='*20)
    for episode_index, (query_images, query_targets, query_modal, support_images, support_targets,) in tqdm(enumerate(val_loader)):

        # Convert query and support images
        query_images = torch.squeeze(query_images.type(torch.FloatTensor))
        support_images = support_images[0].cuda()
        input_var1 = query_images.cuda()
        support_targets = support_targets.cuda().squeeze()

        input_var2 = support_images.view(-1, 3, 84, 84)

        # query_modal = query_modal.cuda()
        query_targets = query_targets.cuda()
        query_targets1, query_targets2 = query_targets.view(2, -1)
        linear_module = distLinear(indim=640*25, outdim=5)
        # linear_module = nn.Linear(640, 5, True)

        linear_module.cuda()


        set_optimizer = torch.optim.SGD(linear_module.parameters(), lr=0.01, momentum=0.9, dampening=0.9,
                                       weight_decay=0.001)

        loss_func = nn.CrossEntropyLoss()
        loss_func = loss_func.cuda()
        batch_size = 4
        support_size = 5 * opt.shot_num
        # emb_s = model(input_var2).detach().mean(dim=[-1, -2])
        # emb_q = model(input_var1).detach().mean(dim=[-1, -2])
        emb_s = model(input_var2).detach().view(-1, 640*25)
        emb_q = model(input_var1).detach().view(-1, 640*25)
        for epoch in range(100):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size, batch_size):
                set_optimizer.zero_grad()
                selected_id = torch.from_numpy(rand_id[i: min(i + batch_size, support_size)]).cuda()
                z_batch = emb_s[selected_id]
                y_batch = support_targets[selected_id]
                scores = linear_module(z_batch)
                loss = loss_func(scores, y_batch)
                loss.backward()
                set_optimizer.step()
        pred_q = linear_module(emb_q).view(2, -1, 5)
        acc_1 = accuracy(pred_q[0], query_targets1).item()
        acc_2 = accuracy(pred_q[1], query_targets2).item()

        top1_m1.update(acc_1)
        accuracies.append(acc_2)

        top1_m2.update(acc_2)
        # ============== print the intermediate results ==============#
        if episode_index % opt.print_freq == 0 and episode_index != 0:
            print('Test-({0}): [{1}/{2}]\t'
                  'Prec@1 {top1_m1.val:.3f} ({top1_m1.avg:.3f}, {top1_m2.avg:.3f} -> {mean:.3f})'.format(
                epoch_index, episode_index, len(val_loader),
                top1_m1=top1_m1, top1_m2=top1_m2, mean=(top1_m1.avg+top1_m2.avg)/2))
    return top1_m2.avg, accuracies

from torch.nn import init
def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

