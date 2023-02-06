from torch import nn
import torch

def mask_mining_loss(ori_logits, msk_logits, labels):
    logits = msk_logits - ori_logits
    celoss = nn.functional.cross_entropy(logits, labels)
    return celoss

def mask_mining_lossV2(ori_logits, msk_logits, labels):
    ce_ori = nn.functional.cross_entropy(ori_logits, labels, reduce=False)
    ce_msk = nn.functional.cross_entropy(msk_logits, labels, reduce=False)
    res = nn.functional.relu(ce_msk-ce_ori).mean()
    return res

def mask_mining_lossV3(ori_logits, msk_logits, labels):
    ce_ori = nn.functional.cross_entropy(ori_logits, labels, reduce=False)
    ce_msk = nn.functional.cross_entropy(msk_logits, labels, reduce=False)
    res = nn.functional.relu(ce_msk-ce_ori).exp().mean()
    return res


def mask_mining_lossV4(ori_logits, msk_logits, labels):
    ce_ori = nn.functional.cross_entropy(ori_logits, labels, reduce=False)
    ce_msk = nn.functional.cross_entropy(msk_logits, labels, reduce=False) + 0.3
    res = nn.functional.relu(ce_msk-ce_ori).exp().mean()
    return res


def mask_mining_lossV5(ori_logits, msk_logits, labels):
    # ce_ori = nn.functional.cross_entropy(ori_logits, labels, reduce=False)
    ce_msk = nn.functional.cross_entropy(msk_logits, labels, reduce=False) + 0.3
    res = ce_msk.mean()
    return res


def mask_mining_lossV6(ori_logits, msk_logits, labels, margin=0.2):
    logits = msk_logits - ori_logits - margin
    celoss = nn.functional.cross_entropy(logits, labels)
    return celoss

def mask_mining_lossV6P(ori_logits, msk_logits, labels, margin=0.2):
    sift = nn.functional.one_hot(labels, ori_logits.size(-1)).detach()
    same_prob = (msk_logits - ori_logits) - margin*sift
    divergence = nn.functional.cross_entropy(same_prob, labels)
    return divergence

def mask_mining_lossV7(ori_logits, msk_logits, labels):
    labels = nn.functional.one_hot(labels, ori_logits.size(-1))
    ori_same_class = (ori_logits * labels).sum(-1)
    msk_same_class = (msk_logits * labels).sum(-1)
    ori_diff_class = -(ori_logits * (1-labels)).sum(-1)/(ori_logits.size(-1)-1)
    msk_diff_class = -(msk_logits * (1-labels)).sum(-1)/(ori_logits.size(-1)-1)

    logits = (msk_same_class - ori_same_class) + msk_diff_class - ori_diff_class
    res = (-logits).exp().mean()
    return res


class UDASoftLabel_MultiScale_V2(nn.Module):
    def __init__(self, topk=4):
        super(UDASoftLabel_MultiScale_V2, self).__init__()
        self.topk = topk

    def forward(self, q, S):
        loss = self.topk_loss(q, S)
        return loss

    def avgpool(self, inp, t):
        res = inp
        for i in range(t):
            res = torch.nn.functional.avg_pool2d(res, 2, 2)
        return res

    def topk_loss(self, q, S):
        S = torch.cat(S, 0)
        pm, C, h, w = S.size()
        q = self.avgpool(q, 2).permute(0, 2, 3, 1).reshape([-1, C])
        S_source = self.avgpool(S, 2).permute(0, 2, 3, 1).reshape([-1, C])
        S = torch.cat([self.avgpool(S, 1).permute(0, 2, 3, 1).reshape([-1, C]), S_source], 0)

        q2s, _ = self.cosine_similar(q, S).topk(k=15, dim=-1)
        all_similarity = q2s.softmax(-1)

        topk_v, _ = all_similarity.topk(k=self.topk, dim=-1)
        loss = -(torch.log(topk_v).mean(-1)).mean()
        return loss

    def cosine_similar(self, q, p):
        innerproduct = torch.einsum('ik,sk->is', q, p)
        q2 = torch.sqrt(torch.einsum('ik,ik->i', q, q))
        p2 = torch.sqrt(torch.einsum('ik,ik->i', p, p))
        q2p2 = torch.einsum('i,s->is', q2, p2)
        res = innerproduct / q2p2
        return res




