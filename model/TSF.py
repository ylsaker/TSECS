import os
from turtle import pos

import torch
from cv2 import log
from numpy import broadcast, source
from model.BaseModel import BaseModelFSL
from model.K_means import *
from torch import nn
from model.similarity_encoder import *
from model.memory_bank import *



TO_LD_MAT = lambda x: x.permute(0, 2, 3, 1).contiguous().view(-1, x.size(1))


class TSFL(BaseModelFSL):
    def __init__(self, cluster_num, component_size=2, sigma=0.3, neighbor_k=3, kl_eps=1e-2,
                 word_eps=0, KL_margin=1, semi_margin=1, eps=1e-10):
        super(TSFL, self).__init__()
        self.clusters = cluster_num
        self.component_size = component_size
        self.sigma = sigma
        self.neighbor_k = neighbor_k
        self.kl_eps = kl_eps 
        self.word_eps = word_eps
        self.KL_margin = KL_margin
        self.semi_margin = semi_margin
        self.gamma = nn.Parameter(torch.ones([1]))
        self.beta = nn.Parameter(torch.zeros([1]))
        self.eps = eps
        self.encoder = BaseSimilarityEncoder(1, sigma)  # Encoding the image-to-class similarity matrix(2-D) to similarity pattern(1-D).
        self.way_num = None
        self.logits = None
        self.train_mode = False
        self.K_means = Kmeans(cluster_num)
        
    def emb_bn2(self, query_emb, support_emb):
        x = torch.cat([query_emb, support_emb]).permute(0, 2, 1)
        mean_ = x.mean([0, 2], True)
        var_ = torch.var(x, [0, 2], keepdim=True)
        x = (x - mean_) / (var_ + self.eps).sqrt() * self.gamma + self.beta
        return x.permute(0, 2, 1)

    def get_bow(self, query_t, support):
        _, C, H, W = query_t.size()
        x = torch.cat([query_t, support.view(-1, C, H, W)]).permute(0, 2, 3, 1).contiguous().view(-1, C)
        words, ass = self.K_means(x)
        return (words, ass), x.mean()

    def project_to_wordspace(self, bow, query, support):
        way, shot, _, _, _ = support.size()
        qm, C, H, W = query.size()
        query = query.permute(0, 2, 3, 1).view(qm, H * W, C)
        support = support.permute(0, 1, 3, 4, 2).view(way, shot, H * W, C)
        # qm, HW, cluster
        query_emb = torch.einsum('ijk,pk->ijp', query / (query.norm(dim=-1, keepdim=True) + 1e-10),
                                 bow / (bow.norm(dim=-1, keepdim=True) + 1e-10)).view(qm, -1, self.clusters)
        # way, shot x HW, cluster
        support_emb = torch.einsum('injk,pk->injp', support / (support.norm(dim=-1, keepdim=True) + 1e-10),
                                   bow / (bow.norm(dim=-1, keepdim=True) + 1e-10)).view(way * shot, -1, self.clusters)
        
        suooprt_lds = nn.functional.normalize(support_emb, p=2, dim=-1)
        simi = torch.einsum('ijk,pqk->ijpq', suooprt_lds, suooprt_lds).mean(-1)
        divergence = 0.
        for i in range(way):
            divergence += (simi[i, :, i] - torch.cat([simi[i, :, :i], simi[i, :, i+1:]], -1).mean(-1)).mean()
        self.set_meter('bow_divergence', divergence.detach())
        return query_emb, support_emb

    def process_output(self, res):
        return res
    
    def get_patterns(self, query_emb, support_emb):
        patterns = torch.einsum('ijk,lpk->ijlp', query_emb/(query_emb.norm(dim=-1, keepdim=True) + 1e-10), support_emb/(support_emb.norm(dim=-1, keepdim=True) + 1e-10))
        topk_v, topk_i = patterns.topk(self.neighbor_k, sorted=False)
        patterns = torch.zeros_like(patterns).scatter(-1, topk_i, topk_v)
        return patterns
    
    def get_prototype(self, x, way, shot):
        return x
    
    def compute_rloss(self, x):
        _, _, d = x.size()
        x = x.view(-1, d)
        mean_ = x.mean(0, True)
        var_ = x - mean_
        cov_ = var_.t() @ var_ / (x.size(0) - 1)
        # trace_ = torch.trace(cov_) / cov_.size(0)
        # rword_loss = (cov_ - torch.eye(cov_.size(0)).to(cov_.device) * trace_).pow(2).sum()
        return cov_.abs().sum()
    
    def emb_bn1(self, x, *args, **kwargs):
        return x
    
    def KL_divergence(self, cov1, cov2, mean_1, mean_2):
        trace_ = torch.trace(cov1.inverse()@cov2)
        logs_ = torch.logdet(cov1) - torch.logdet(cov2)
        mean_sub = mean_1 - mean_2
        manhanobis = (mean_sub @ cov1.inverse() @ mean_sub.t()).squeeze()
        res = torch.relu((trace_ + logs_ + manhanobis - mean_2.size(-1)) / 2 - self.KL_margin)
        return res
    
    def get_cluster_info(self, query):
        def cov(x):
            mean = x.mean(0, True)
            vars = (x - mean)
            cov = vars.t() @ vars/ (x.size(0) - 1)
            # cov = vars.t()@vars/ (x.size(0) - 1)
            return cov

        source, target = query[:query.size(0)//2].view(-1, query.size(-1)), query[query.size(0)//2:].view(-1, query.size(-1))
        cov_s, cov_t = cov(source), cov(target)
        try:
            KL = self.KL_divergence(cov_s, cov_t, source.mean(0, True), target.mean(0, True))
        except:
            KL = self.KL_divergence(cov_s + self.word_eps * torch.eye(cov_s.size(0)).cuda(), 
                                    cov_t + self.word_eps * torch.eye(cov_s.size(0)).cuda(), source.mean(0, True), target.mean(0, True))
        # KL = self.KL_divergence(cov_t, cov_s, target.mean(0, True), source.mean(0, True)) + self.KL_divergence(cov_s, cov_t, source.mean(0, True), target.mean(0, True))
        wass = self.warserstain(cov_s, cov_t, source.mean(0, True), target.mean(0, True))
        return KL, wass
    
    def warserstain(self, cov1, cov2, mean_1, mean_2):
        return (cov1-cov2).pow(2).sum() + (mean_1-mean_2).pow(2).sum()

    def combine_words(self, x):
        N, _, C = x.size()
        H = W = int(math.sqrt(x.size(1)))
        x = x.view(N, H, W, C)
        tmp = []
        for r in range(0, H, self.component_size):
            row_feat = x.narrow(1, r, self.component_size)
            for c in range(0, W, self.component_size):
                cell_component = row_feat.narrow(2, c, self.component_size)
                tmp.append(cell_component.contiguous().view(N, 1, -1))
        tmp = torch.cat(tmp, 1)
        return tmp
    
    def semi_KL(self, cov1, cov2, mean1, mean2):
        trace_ = torch.trace(cov1.inverse()@cov2)
        logs_ = torch.logdet(cov1) - torch.logdet(cov2)
        mean_sub = mean1 - mean2
        manhanobis = (mean_sub @ cov1.inverse() @ mean_sub.t()).squeeze()
        res = torch.relu((trace_ + logs_ + manhanobis - mean2.size(-1)) / 2 - self.semi_margin)
        return res
    
    def rspa(self, cov1):
        I = torch.trace(cov1) / cov1.size(-1) * torch.eye(cov1.size(-1)).to(cov1.device)
        res = (cov1 - I).pow(2).sum()
        return res
    
    def get_cov_loss(self, data_x):
        def cov(x):
            x = x.permute(2, 1, 0, 3)
            x = x.contiguous().view(x.size(0) * x.size(1), 1, -1, x.size(-1))
            mean = x.mean([1, 2], True)
            vars = (x - mean).view(x.size(0), -1, x.size(-1))
            cov = torch.bmm(vars.permute(0, 2, 1), vars) / (x.size(1) * x.size(2) - 1)
            return cov, mean
        n = data_x.size(0) // 2
        I = torch.eye(data_x.size(-1)).to(data_x.device) * self.kl_eps
        cov_s, mean_s = cov(data_x[:n])
        cov_t, mean_t = cov(data_x[n:])
        semi_kl = 0.
        rspa_loss = 0.
        for cs, ct, ms, mt in zip(cov_s, cov_t, mean_s, mean_t):
             semi_kl += self.semi_KL(cs + I, ct + I, ms[0], mt[0])
             rspa_loss += self.rspa(cs + I)
        semi_kl = semi_kl / cov_t.size(0)
        return rspa_loss, semi_kl

    def get_queryt(self, query, train):
        qm, C, H, W = query.size()
        if not train:
            query_t = query.view(2, -1, C, H, W)[1]
        else:
            query_t = query
        return query_t
    
    def process_logits(self, support_emb):
        pass
    
    def forward_(self, query, support, train=True):
        self.train_mode = train
        way, shot, _, _, _ = support.size()
        self.way_num = way
        qm, C, H, W = query.size()

        query_t = self.get_queryt(query, train)
        (clusters, _), _ = self.get_bow(query_t, support)
        self.cluster_I = torch.eye(self.clusters).cuda()

        self.clusters = clusters.size(0)

        # protect raw features to the word space
        query_emb, support_emb = self.project_to_wordspace(clusters, query, support)

        # pass the BN
        emb_data = self.emb_bn1(torch.cat([query_emb, support_emb]).permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        query_emb, support_emb = emb_data[:qm], emb_data[qm:].view(way, -1, self.clusters)

        # compute word-block-level features
        query_emb_high = self.combine_words(query_emb)
        support_emb_high = self.combine_words(support_emb.view(way*shot, H*W, -1)).view(way*shot, -1, self.component_size**2*self.clusters)
        support_emb_high = self.get_prototype(support_emb_high, way, shot)

        # pass the BN
        componet_data = self.emb_bn2(query_emb_high, support_emb_high)
        query_emb_high, support_emb_high = componet_data[:qm], componet_data[qm:].contiguous().view(way, -1, self.component_size**2*self.clusters)

        # pass the IMSE
        patterns = self.get_patterns(query_emb_high, support_emb_high)

        # simi_vectors: (qm, shot, way, H*W)
        encoder_res = self.encoder(patterns)
        simi_vectors, self.logits = encoder_res['patterns'], encoder_res['logits']
        pattern_l2 = simi_vectors.pow(2).sqrt().sum()
        rspa_loss, semi_kl = self.get_cov_loss(simi_vectors)

        # process logits
        self.process_logits(support_emb)

        # compute the KL for the domain-shared word space
        cluster_KL, cluster_wass = self.get_cluster_info(query_emb)
        word_KL, word_wass = self.get_cluster_info(query_emb_high)

        # experiment index to be showed
        self.set_meter('semi_kl', semi_kl.clone().detach())
        self.set_meter('cluster_KL', cluster_KL.clone().detach())
        self.set_meter('word_KL', word_KL.clone().detach())

        return {
            'logits': self.logits, 'rspa_loss': rspa_loss,
            'cluster_KL':cluster_KL, 'semi_kl': semi_kl, 'word_KL': word_KL,
            'cluster_wass': cluster_wass, 'pattern_l2': pattern_l2}


class TSFLv2(TSFL):  # 相比于TSFL, 修改了self.K_means和self.encoder
    def __init__(self, cluster_num, component_size=2,
                 sigma=0.3, neighbor_k=3, kl_eps=1e-2, word_eps=0,
                 KL_margin=1, semi_margin=1, eps=1e-10):
        super(TSFLv2, self).__init__(cluster_num, component_size,
                                   sigma, neighbor_k, kl_eps,
                                   word_eps, KL_margin,
                                   semi_margin)
        self.K_means = KmeansUnitspaceV2(cluster_num)
        self.encoder = SimilarityEncoderV3(1, sigma, 2)
        self.gamma = nn.Parameter(torch.ones([1]))
        self.beta = nn.Parameter(torch.zeros([1]))
        self.eps = eps

    def emb_bn2(self, query_emb, support_emb):
        x = torch.cat([query_emb, support_emb]).permute(0, 2, 1)
        mean_ = x.mean([0, 2], True)
        var_ = torch.var(x, [0, 2], keepdim=True)
        x = (x - mean_) / (var_ + self.eps).sqrt() * self.gamma + self.beta
        return x.permute(0, 2, 1)

    def get_bow(self, query_t, support):
        _, C, H, W = query_t.size()
        x = torch.cat([query_t, support.view(-1, C, H, W)]).permute(0, 2, 3, 1).contiguous().view(-1, C)
        words, ass = self.K_means(x)
        return (words, ass), x.mean()

from model.semantic_component import *
class TSFL_LocalSemantic(TSFL):
    def __init__(self, cluster_num, component_size=2,
                sigma=0.3, neighbor_k=3, kl_eps=1e-2, word_eps=0,
                KL_margin=1, semi_margin=1, scope_size=2):
        super(TSFL_LocalSemantic, self).__init__(cluster_num, component_size,
                                    sigma, neighbor_k, kl_eps,
                                    word_eps, KL_margin,
                                    semi_margin)

        self.K_means = SemanticComponentExtractor(cluster_num, scope_size)

    def get_bow(self, query_t, support):

        _, C, H, W = query_t.size()
        x = torch.cat([TO_LD_MAT(query_t), TO_LD_MAT(support.view(-1, C, H, W))])
        words = self.K_means(x)['cluster_centers']
        return (words, 0), x.mean()


class TSFL_LocalSemanticV2(TSFL):
    def __init__(self, cluster_num, component_size=2,
                sigma=0.3, neighbor_k=3, kl_eps=1e-2, word_eps=0,
                KL_margin=1, semi_margin=1, scope_size=2, focus=50):
        super().__init__(cluster_num, component_size,
                                    sigma, neighbor_k, kl_eps,
                                    word_eps, KL_margin,
                                    semi_margin)

        self.K_means = SemanticComponentExtractorV2(cluster_num, scope_size, focus)

    def get_bow(self, query_t, support):
        _, C, H, W = query_t.size()
        x = torch.cat([TO_LD_MAT(query_t), TO_LD_MAT(support.view(-1, C, H, W))])
        words = self.K_means(x)['cluster_centers']
        return (words, 0), x.mean()


class TSFL_LocalSemanticAdaptative(TSFL):
    def __init__(self, cluster_num, component_size=2,
                sigma=0.3, neighbor_k=3, kl_eps=1e-2, word_eps=0,
                KL_margin=1, semi_margin=1, scope_size=2, focus=50):
        super().__init__(cluster_num, component_size,
                                    sigma, neighbor_k, kl_eps,
                                    word_eps, KL_margin,
                                    semi_margin)

        self.K_means = SemanticComponentExtractorAdaptative(cluster_num, scope_size, focus)

    def get_bow(self, query_t, support):
        _, C, H, W = query_t.size()
        x = torch.cat([TO_LD_MAT(query_t), TO_LD_MAT(support.view(-1, C, H, W))])
        words = self.K_means(x)['cluster_centers']

        norm_cluster_centers = nn.functional.normalize(words, p=2, dim=-1)
        inter_sim = norm_cluster_centers @ norm_cluster_centers.t()
        prob_ = inter_sim.softmax(-1)
        cluster_angle_loss = -(prob_ * prob_.log() *
                               (1 - torch.eye(words.size(0)).to(words.device))).sum(-1).mean()
        self.set_output('cluster_angle_loss', cluster_angle_loss)

        return (words, 0), x.mean()


class TSFL_LocalSemanticAdaptativeV2(TSFL_LocalSemanticAdaptative):
    def __init__(self, cluster_num, component_size=2,
                sigma=0.3, neighbor_k=3, kl_eps=1e-2, word_eps=0,
                KL_margin=1, semi_margin=1, scope_size=2, focus=50):
        super().__init__(cluster_num, component_size,
                                    sigma, neighbor_k, kl_eps,
                                    word_eps, KL_margin,
                                    semi_margin)
        self.K_means = SemanticComponentExtractorAdaptativeV2(cluster_num, scope_size, focus)


class TSFL_FineKLv2(TSFLv2):
    def __init__(self, cluster_num, component_size=2,
                 sigma=0.3, neighbor_k=3, kl_eps=1e-2, word_eps=0,
                 KL_margin=1, semi_margin=1):
        super(TSFL_FineKLv2, self).__init__(cluster_num, component_size,
                                   sigma, neighbor_k, kl_eps,
                                   word_eps, KL_margin,
                                   semi_margin)
        self.K_means = KmeansUnitspaceV2(cluster_num)
        self.encoder = SimilarityEncoderV3(1, sigma, 2)

    def get_cluster_info(self, query):
        def cov(x):
            mean = x.mean(0, True)
            vars = (x - mean)
            cov = vars.t() @ vars / (x.size(0) - 1) + \
                self.word_eps * torch.eye(mean.size(-1)).to(mean.device)
            return cov
        source = query[:query.size(0) // 2]
        target = query[query.size(0) // 2:]
        cluster_sets_tmp = [[] for _ in range(self.way_num)]
        pseudo_target_label = self.logits.clone().view(2, -1, self.way_num)[0].argmax(-1)
        for t_sample, pseudo_l in zip(target, pseudo_target_label):
            cluster_sets_tmp[pseudo_l].append(t_sample)
        source = source.view(self.way_num, -1, source.size(-1))
        fine_KL = 0
        fine_wass = 0
        valid_num = 0
        for c, sets in enumerate(cluster_sets_tmp):
            if sets:
                valid_num += 1
                c_target = torch.cat(sets)
                mean_t = c_target.mean(0, True)
                mean_s = source[c].mean(0, True)
                cov_t = cov(c_target)
                cov_s = cov(source[c])
                fine_KL += self.KL_divergence(cov_s, cov_t,
                                              mean_s, mean_t)
                fine_wass += self.warserstain(cov_s, cov_t,
                                              mean_s, mean_t)
        fine_KL = fine_KL / valid_num
        fine_wass = fine_wass / valid_num
        return fine_KL, fine_wass


class TSFL_FineKLv3(TSFL_FineKLv2):
    def __init__(self, cluster_num, component_size=2,
                 sigma=0.3, neighbor_k=3, kl_eps=1e-2, word_eps=0,
                 KL_margin=1, semi_margin=1):
        super(TSFL_FineKLv3, self).__init__(cluster_num, component_size,
                                   sigma, neighbor_k, kl_eps,
                                   word_eps, KL_margin,
                                   semi_margin)

    def emb_bn2(self, query_emb, support_emb):
        x = torch.cat([query_emb, support_emb]).permute(0, 2, 1)
        return x.permute(0, 2, 1)


class TSFL_FineKLv4(TSFL_FineKLv2):
    def __init__(self, cluster_num, component_size=2,
                 sigma=0.3, neighbor_k=3, kl_eps=1e-2, word_eps=0,
                 KL_margin=1, semi_margin=1):
        super(TSFL_FineKLv4, self).__init__(cluster_num, component_size,
                                   sigma, neighbor_k, kl_eps,
                                   word_eps, KL_margin,
                                   semi_margin)


class TSFL_Dynamic(TSFLv2):
    def __init__(self, cluster_num, component_size=2, sigma=0.3,
                 neighbor_k=3, kl_eps=1e-2, word_eps=0,
                 KL_margin=1, semi_margin=1, basis_thredshold=50):
        super(TSFL_Dynamic, self).__init__(cluster_num, component_size,
                                   sigma, neighbor_k, kl_eps,
                                   word_eps, KL_margin,
                                   semi_margin)
        self.K_means = DynamicKmeans(cluster_num, basis_thredshold=basis_thredshold)
        self.encoder = SimilarityEncoderV3(1, sigma, 2)

    def get_bow(self, query_t, support):
        _, C, H, W = query_t.size()
        x = torch.cat([query_t, support.view(-1, C, H, W)]).permute(0, 2, 3, 1).contiguous().view(-1, C)
        bow, sv_means = self.K_means(x)
        self.set_meter('word_num', torch.tensor([bow.size(0)]))
        self.set_meter('sv_means', sv_means)
        return (bow, sv_means), _


class TSFL_DynamicV2(TSFL_Dynamic):
    def __init__(self, cluster_num, component_size=2, sigma=0.3,
                 neighbor_k=3, kl_eps=1e-2, word_eps=0,
                 KL_margin=1, semi_margin=1, basis_thredshold=0.5):
        super(TSFL_DynamicV2, self).__init__(cluster_num, component_size,
                                   sigma, neighbor_k, kl_eps, word_eps,
                                   KL_margin, semi_margin, basis_thredshold)
        self.K_means = DynamicKmeansV2(cluster_num, basis_thredshold=basis_thredshold)


class TSFL_DynamicFineKL(TSFL_FineKLv4):
    def __init__(self, cluster_num, component_size=2, sigma=0.3,
                neighbor_k=3, kl_eps=1e-2, word_eps=0,
                KL_margin=1, semi_margin=1, basis_thredshold=50):
        super(TSFL_DynamicFineKL, self).__init__(cluster_num, component_size,
                                   sigma, neighbor_k, kl_eps,
                                   word_eps, KL_margin,
                                   semi_margin)
        self.K_means = DynamicKmeansV2(cluster_num, basis_thredshold=basis_thredshold)
        self.encoder = SimilarityEncoderV3(1, sigma, 2)

    def get_bow(self, query_t, support):
        _, C, H, W = query_t.size()
        x = torch.cat([query_t, support.view(-1, C, H, W)]).permute(0, 2, 3, 1).contiguous().view(-1, C)
        bow, sv_means = self.K_means(x)
        self.set_meter('word_num', torch.tensor([bow.size(0)]))
        self.set_meter('sv_means', sv_means)
        return (bow, sv_means), _


class TSFL_ClassDynamicFineKL(TSFL_DynamicFineKL):
    def __init__(self, cluster_num, component_size=2, sigma=0.3,
                neighbor_k=3, kl_eps=1e-2, word_eps=0,
                KL_margin=1, semi_margin=1, basis_thredshold=50):
        super(TSFL_ClassDynamicFineKL, self).__init__(cluster_num, component_size,
                                   sigma, neighbor_k, kl_eps,
                                   word_eps, KL_margin,
                                   semi_margin, basis_thredshold)

    def get_bow(self, query_t, support):
        _, C, H, W = query_t.size()
        # x = torch.cat([query_t, support.view(-1, C, H, W)]).permute(0, 2, 3, 1).contiguous().view(-1, C)
        bows = []
        sv_means = []
        for c in range(len(support)):
            x = support[c].view(-1, C, H*W).permute(0, 2, 1).contiguous().view(-1, C)
            bow, sv_mean = self.K_means(x)
            sv_means.append(sv_mean)
            bows.append(bow)
        x = query_t.view(-1, C, H*W).permute(0, 2, 1).contiguous().view(-1, C)
        bow, sv_mean = self.K_means(x)
        sv_means.append(sv_mean)
        bows.append(bow)
        sv_means = torch.stack(sv_means)
        bows = torch.cat(bows)
        self.set_meter('word_num', torch.tensor([bows.size(0)]))
        self.set_meter('sv_means', sv_means.mean())
        return (bow, sv_means), _


class TSFL_Broadcast(TSFL_ClassDynamicFineKL):
    def __init__(self, cluster_num, component_size=2,
                 sigma=0.3, neighbor_k=3, kl_eps=1e-2, word_eps=0,
                 KL_margin=1, semi_margin=1, basis_thredshold=50, epsilon=2.5, tri_margin=1.0):
        super(TSFL_Broadcast, self).__init__(cluster_num, component_size,
                                   sigma, neighbor_k, kl_eps,
                                   word_eps, KL_margin,
                                   semi_margin, basis_thredshold)
        self.epsilon = epsilon
        self.tri_margin = tri_margin
        self.ce_loss = nn.CrossEntropyLoss()

    def euc(self, input1, input2):
        input1_tmp = input1.mean(dim=1)
        input2_tmp = input2.mean(dim=1)
        n1, n2 = input1_tmp.size(0), input2_tmp.size(1)
        input1_tmp = input1_tmp.repeat(1, n2, 1)
        input2_tmp = input2_tmp.repeat(n1, 1, 1)
        return torch.einsum("ijk,pqk->ijpq", input1, input2)

    def get_triplet_loss(self, similarity):
        triplet_loss = torch.tensor(0.).to(similarity.device)
        poses, _ = torch.max(similarity, dim=1)
        most_sim = similarity.topk(k=2, dim=1, largest=True, sorted=True)[0][:,1]
        # poses = poses.unsqueeze(1).repeat(1, 4)
        losses = most_sim - poses + self.tri_margin
        for loss in losses:
            if loss > 0.:
                triplet_loss += loss
        return triplet_loss

    def select_q_target(self, logits):
        target_logits, target_label = torch.max(logits, dim=1)
        selected_index = []
        selected_label = []
        for i, l in enumerate(target_logits):
            if l >= self.epsilon:
                selected_index.append(i+75)
                selected_label.append(target_label[i])
        return selected_index, selected_label

    def get_target_center(self, query, support, query_index, query_label):
        bins = [[],[],[],[],[]]
        class_center = []
        for i, idx in enumerate(query_index):
            bins[query_label[i]].append(query[idx])
        for i, bin in enumerate(bins):
            if bin:
                # bin.append(support[i])
                class_center.append(torch.stack(bin).mean(dim=0))
            else:
                class_center.append(support[i])
        return torch.stack(class_center)

    def get_target_pattern(self, target_logits, query_emb_high, support_emb_high):
        target_ep, target_ep_label = self.select_q_target(target_logits)
        target_ep_center = self.get_target_center(query_emb_high, support_emb_high, target_ep, target_ep_label)
        patterns = self.get_patterns(query_emb_high[75:], target_ep_center)
        return {"patterns": patterns, "target_ep": target_ep, "target_ep_label": target_ep_label}

    def forward_(self, query, support, train=True):
        self.train_mode = train
        way, shot, _, _, _ = support.size()
        self.way_num = way
        qm, C, H, W = query.size()

        query_t = self.get_queryt(query, train)
        (clusters, _), _ = self.get_bow(query_t, support)
        self.cluster_I = torch.eye(self.clusters).cuda()

        self.clusters = clusters.size(0)

        # protect raw features to the word space
        query_emb, support_emb = self.project_to_wordspace(clusters, query, support)

        # pass the BN
        emb_data = self.emb_bn1(torch.cat([query_emb, support_emb]).permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        query_emb, support_emb = emb_data[:qm], emb_data[qm:].view(way, -1, self.clusters)

        # compute word-block-level features
        query_emb_high = self.combine_words(query_emb)
        support_emb_high = self.combine_words(support_emb.view(way*shot, H*W, -1)).view(way*shot, -1, self.component_size**2*self.clusters)
        support_emb_high = self.get_prototype(support_emb_high, way, shot)

        # pass the BN
        componet_data = self.emb_bn2(query_emb_high, support_emb_high)
        query_emb_high, support_emb_high = componet_data[:qm], componet_data[qm:].contiguous().view(way, -1, self.component_size**2*self.clusters)

        # pass the IMSE
        patterns = self.get_patterns(query_emb_high, support_emb_high)

        # simi_vectors: (qm, shot, way, H*W)
        encoder_res = self.encoder(patterns)
        simi_vectors, self.logits = encoder_res['patterns'], encoder_res['logits']
        pattern_l2 = simi_vectors.pow(2).sqrt().sum()
        rspa_loss, semi_kl = self.get_cov_loss(simi_vectors)

        # process logits
        self.process_logits(support_emb)

        # compute the KL for the domain-shared word space
        cluster_KL, cluster_wass = self.get_cluster_info(query_emb)
        word_KL, word_wass = self.get_cluster_info(query_emb_high)
        target_logits = self.logits[75:] # target to support
        # select query T'
        results = self.get_target_pattern(target_logits, query_emb_high, support_emb_high)
        # dis_matric = self.euc(query_emb_high[75:], target_ep_center)
        patterns, target_ep, target_ep_label = results["patterns"], results["target_ep"], results["target_ep_label"]
        encoder_res = self.encoder(patterns)
        simi_vectors, logits = encoder_res['patterns'], encoder_res['logits']
        triplet_loss = self.get_triplet_loss(logits)
        self.logits[75:] = logits
        # experiment index to be showed
        self.set_meter('semi_kl', semi_kl.clone().detach())
        self.set_meter('cluster_KL', cluster_KL.clone().detach())
        self.set_meter('word_KL', word_KL.clone().detach())
        self.set_meter('triplet_loss', triplet_loss.clone().detach())
        return {
            'logits': self.logits, 'rspa_loss': rspa_loss,
            'cluster_KL':cluster_KL, 'semi_kl': semi_kl, 'word_KL': word_KL,
            'cluster_wass': cluster_wass, 'pattern_l2': pattern_l2, 'triplet_loss': triplet_loss
            }


class TSFL_Broadcastv2(TSFL_Broadcast):
    def __init__(self, cluster_num, component_size=2,
                 sigma=0.3, neighbor_k=3, kl_eps=1e-2, word_eps=0,
                 KL_margin=1, semi_margin=1, basis_thredshold=50, epsilon=2.5, tri_margin=1.):
        super(TSFL_Broadcastv2, self).__init__(cluster_num, component_size,
                                               sigma, neighbor_k, kl_eps,
                                               word_eps, KL_margin,
                                               semi_margin, basis_thredshold, epsilon, tri_margin)

    def get_source_center(self, query, support):
        source_center = []
        for i, sup in enumerate(support):
            source_center.append(torch.cat([sup.unsqueeze(0), query[15*i: 15*(i+1)]], dim=0).mean(dim=0))
        return torch.stack(source_center)


class TSFL_Broadcastv3(TSFL_Broadcast):
    def __init__(self, cluster_num, component_size=2,
                 sigma=0.3, neighbor_k=3, kl_eps=1e-2, word_eps=0,
                 KL_margin=1, semi_margin=1, basis_thredshold=50, epsilon=2.5, tri_margin=1., select_iter=2):
        super(TSFL_Broadcastv3, self).__init__(cluster_num, component_size,
                                               sigma, neighbor_k, kl_eps,
                                               word_eps, KL_margin,
                                               semi_margin, basis_thredshold, epsilon, tri_margin)
        self.iter = select_iter

    def iter_select_target(self, logits, selected_index, selected_label):
        target_logits, target_label = torch.max(logits, dim=1)
        for i, l in enumerate(target_logits):
            if l > self.epsilon and i not in selected_index:
                selected_index.append(i)
                selected_label.append(target_label[i])
        return selected_index, selected_label

    def iter_get_target_pattern(self, target_logits, query_emb_high, support_emb_high, target_ep, target_ep_label):
        target_ep, target_ep_label = self.iter_select_target(target_logits, target_ep, target_ep_label)
        target_ep_center = self.get_target_center(query_emb_high, support_emb_high, target_ep, target_ep_label)
        patterns = self.get_patterns(query_emb_high[75:], target_ep_center)
        return {"patterns": patterns, "target_ep": target_ep, "target_ep_label": target_ep_label}

    def forward_(self, query, support, train=True):
        self.train_mode = train
        way, shot, _, _, _ = support.size()
        self.way_num = way
        qm, C, H, W = query.size()

        query_t = self.get_queryt(query, train)
        (clusters, _), _ = self.get_bow(query_t, support)
        self.cluster_I = torch.eye(self.clusters).cuda()

        self.clusters = clusters.size(0)

        # protect raw features to the word space
        query_emb, support_emb = self.project_to_wordspace(clusters, query, support)

        # pass the BN
        emb_data = self.emb_bn1(torch.cat([query_emb, support_emb]).permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        query_emb, support_emb = emb_data[:qm], emb_data[qm:].view(way, -1, self.clusters)

        # compute word-block-level features
        query_emb_high = self.combine_words(query_emb)
        support_emb_high = self.combine_words(support_emb.view(way*shot, H*W, -1)).view(way*shot, -1, self.component_size**2*self.clusters)
        support_emb_high = self.get_prototype(support_emb_high, way, shot)

        # pass the BN
        componet_data = self.emb_bn2(query_emb_high, support_emb_high)
        query_emb_high, support_emb_high = componet_data[:qm], componet_data[qm:].contiguous().view(way, -1, self.component_size**2*self.clusters)

        # pass the IMSE
        patterns = self.get_patterns(query_emb_high, support_emb_high)

        # simi_vectors: (qm, shot, way, H*W)
        encoder_res = self.encoder(patterns)
        simi_vectors, self.logits = encoder_res['patterns'], encoder_res['logits']
        pattern_l2 = simi_vectors.pow(2).sqrt().sum()
        rspa_loss, semi_kl = self.get_cov_loss(simi_vectors)

        # process logits
        self.process_logits(support_emb)

        # compute the KL for the domain-shared word space
        cluster_KL, cluster_wass = self.get_cluster_info(query_emb)
        word_KL, word_wass = self.get_cluster_info(query_emb_high)
        target_logits = self.logits[75:] # target to support
        # select query T'
        results = self.get_target_pattern(target_logits, query_emb_high, support_emb_high)
        patterns, target_ep, target_ep_label = results["patterns"], results["target_ep"], results["target_ep_label"]
        # dis_matric = self.euc(query_emb_high[75:], target_ep_center)
        encoder_res = self.encoder(patterns)
        simi_vectors, target_logits = encoder_res['patterns'], encoder_res['logits']
        for i in range(self.iter):
            results = self.iter_get_target_pattern(target_logits, query_emb_high, support_emb_high, target_ep, target_ep_label)
            patterns, target_ep, target_ep_label = results["patterns"], results["target_ep"], results["target_ep_label"]
            # dis_matric = self.euc(query_emb_high[75:], target_ep_center)
            encoder_res = self.encoder(patterns)
            simi_vectors, target_logits = encoder_res['patterns'], encoder_res['logits']

        triplet_loss = self.get_triplet_loss(target_logits)
        # experiment index to be showed
        self.set_meter('semi_kl', semi_kl.clone().detach())
        self.set_meter('cluster_KL', cluster_KL.clone().detach())
        self.set_meter('word_KL', word_KL.clone().detach())
        self.set_meter('triplet_loss', triplet_loss.clone().detach())
        return {
            'logits': self.logits, 'rspa_loss': rspa_loss,
            'cluster_KL':cluster_KL, 'semi_kl': semi_kl, 'word_KL': word_KL,
            'cluster_wass': cluster_wass, 'pattern_l2': pattern_l2, 'triplet_loss': triplet_loss
            }

class TSFL_Broadcastv4(TSFL_Broadcastv3):
    def __init__(self, cluster_num, component_size=2,
                 sigma=0.3, neighbor_k=3, kl_eps=1e-2, word_eps=0,
                 KL_margin=1, semi_margin=1, basis_thredshold=50, epsilon=2.5, weight_en=0.1, tri_margin=1., select_iter=2):
        super(TSFL_Broadcastv4, self).__init__(cluster_num, component_size,
                                               sigma, neighbor_k, kl_eps,
                                               word_eps, KL_margin, semi_margin,
                                               basis_thredshold,
                                               epsilon, tri_margin, select_iter)
        self.weight_en = weight_en

    def weight_enrase(self, ):
        self.epsilon = self.epsilon * (1+self.weight_en)



class TSFL_Broadcastv5(TSFL_Broadcast):
    def __init__(self, cluster_num, component_size=2,
                 sigma=0.3, neighbor_k=3, kl_eps=1e-2, word_eps=0,
                 KL_margin=1, semi_margin=1, basis_thredshold=50, epsilon=2.5, tri_margin=1.0):
        super(TSFL_Broadcastv5, self).__init__(cluster_num, component_size,
                                   sigma, neighbor_k, kl_eps,
                                   word_eps, KL_margin,
                                   semi_margin, basis_thredshold, epsilon, tri_margin)

    def forward_(self, query, support, train=True):
        self.train_mode = train
        way, shot, _, _, _ = support.size()
        self.way_num = way
        qm, C, H, W = query.size()

        query_t = self.get_queryt(query, train)
        (clusters, _), _ = self.get_bow(query_t, support)
        self.cluster_I = torch.eye(self.clusters).cuda()

        self.clusters = clusters.size(0)

        # protect raw features to the word space
        query_emb, support_emb = self.project_to_wordspace(clusters, query, support)

        # pass the BN
        emb_data = self.emb_bn1(torch.cat([query_emb, support_emb]).permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        query_emb, support_emb = emb_data[:qm], emb_data[qm:].view(way, -1, self.clusters)

        # compute word-block-level features
        query_emb_high = self.combine_words(query_emb)
        support_emb_high = self.combine_words(support_emb.view(way*shot, H*W, -1)).view(way*shot, -1, self.component_size**2*self.clusters)
        support_emb_high = self.get_prototype(support_emb_high, way, shot)

        # pass the BN
        componet_data = self.emb_bn2(query_emb_high, support_emb_high)
        query_emb_high, support_emb_high = componet_data[:qm], componet_data[qm:].contiguous().view(way, -1, self.component_size**2*self.clusters)

        # pass the IMSE
        patterns = self.get_patterns(query_emb_high, support_emb_high)

        # simi_vectors: (qm, shot, way, H*W)
        encoder_res = self.encoder(patterns)
        simi_vectors, self.logits = encoder_res['patterns'], encoder_res['logits']
        pattern_l2 = simi_vectors.pow(2).sqrt().sum()
        rspa_loss, semi_kl = self.get_cov_loss(simi_vectors)

        # process logits
        self.process_logits(support_emb)

        # compute the KL for the domain-shared word space
        cluster_KL, cluster_wass = self.get_cluster_info(query_emb)
        word_KL, word_wass = self.get_cluster_info(query_emb_high)
        target_logits = self.logits[75:] # target to support
        # select query T'
        patterns = self.get_target_pattern(target_logits, query_emb_high, support_emb_high)
        # dis_matric = self.euc(query_emb_high[75:], target_ep_center)
        encoder_res = self.encoder(patterns)
        simi_vectors, logits = encoder_res['patterns'], encoder_res['logits']
        self.logits[75:] = logits
        triplet_loss = self.get_triplet_loss(logits)
        # experiment index to be showed
        self.set_meter('semi_kl', semi_kl.clone().detach())
        self.set_meter('cluster_KL', cluster_KL.clone().detach())
        self.set_meter('word_KL', word_KL.clone().detach())
        self.set_meter('triplet_loss', triplet_loss.clone().detach())
        return {
            'logits': self.logits, 'rspa_loss': rspa_loss,
            'cluster_KL':cluster_KL, 'semi_kl': semi_kl, 'word_KL': word_KL,
            'cluster_wass': cluster_wass, 'pattern_l2': pattern_l2, 'triplet_loss': triplet_loss
            }


class TSFL_BroadcastV6(TSFL_Broadcastv4):
    def __init__(self, cluster_num, component_size=2, sigma=0.3,
                neighbor_k=3, kl_eps=1e-2, word_eps=0,
                KL_margin=1, semi_margin=1, basis_thredshold=50, epsilon=2.5, weight_en=0.1, tri_margin=1., select_iter=2):
        super(TSFL_BroadcastV6, self).__init__(cluster_num, component_size,
                                                      sigma, neighbor_k, kl_eps,
                                                      word_eps, KL_margin, semi_margin,
                                                      basis_thredshold, epsilon,
                                                      weight_en, tri_margin, select_iter)

    def get_source_center(self, query, support):  # 源域中心
        source_center = []
        for i, sup in enumerate(support):
            source_center.append(torch.cat([sup.unsqueeze(0), query[15*i: 15*(i+1)]], dim=0).mean(dim=0))
        return torch.stack(source_center)

    def get_target_center(self, query, support, query_index, query_label):  # 目标域中心
        bins = [[],[],[],[],[]]
        class_center = []
        for i, idx in enumerate(query_index):
            bins[query_label[i]].append(query[idx])
        for i, bin in enumerate(bins):
            if bin:
                # bin.append(support[i])
                class_center.append(torch.stack(bin).mean(dim=0))
            else:
                # print(str(i)+" is empty")
                if support[i].size(0) > 25:
                    class_center.append(support[i].contiguous().view(5, 25, -1).mean(dim=0))
                else:
                    class_center.append(support[i])
        return torch.stack(class_center)

    def get_target_centerv2(self, query, support, query_index, query_label, train):  # 无用
        bins = [[],[],[],[],[]]
        class_center = []
        for i, idx in enumerate(query_index):
            bins[query_label[i]].append(query[idx])
        for i, bin in enumerate(bins):
            if bin:
                # bin.append(support[i])
                class_center.append(torch.stack(bin).mean(dim=0))
            else:
                if train:
                    # print(str(i)+" is empty")
                    class_center.append(support[i])
                else:
                    # print(str(i)+" is empty")
                    class_center.append(support[i])
        return torch.stack(class_center)

    def forward_(self, query, support, train=True, opt=None):  # =====多加了opt参数 ===== #
        self.train_mode = train
        way, shot, _, _, _ = support.size()
        self.way_num = way
        qm, C, H, W = query.size()

        query_t = self.get_queryt(query, train)
        (clusters, _), _ = self.get_bow(query_t, support)

        self.clusters = clusters.size(0)

        # protect raw features to the word space
        query_emb, support_emb = self.project_to_wordspace(clusters, query, support)

        # pass the BN
        emb_data = self.emb_bn1(torch.cat([query_emb, support_emb]).permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        query_emb, support_emb = emb_data[:qm], emb_data[qm:].view(way, -1, self.clusters)

        # compute word-block-level features
        query_emb_high = self.combine_words(query_emb)
        support_emb_high = self.combine_words(support_emb.view(way*shot, H*W, -1)).view(way*shot, -1, self.component_size**2*self.clusters)
        support_emb_high = self.get_prototype(support_emb_high, way, shot)

        # pass the BN
        componet_data = self.emb_bn2(query_emb_high, support_emb_high)
        query_emb_high, support_emb_high = componet_data[:qm], componet_data[qm:].contiguous().view(way, -1, self.component_size**2*self.clusters)

        # pass the IMSE
        patterns = self.get_patterns(query_emb_high, support_emb_high)

        # simi_vectors: (qm, shot, way, H*W)
        encoder_res = self.encoder(patterns)
        simi_vectors, self.logits = encoder_res['patterns'], encoder_res['logits']
        pattern_l2 = simi_vectors.pow(2).sqrt().sum()
        rspa_loss, semi_kl = self.get_cov_loss(simi_vectors)

        # process logits
        self.process_logits(support_emb)

        # compute the KL for the domain-shared word space
        cluster_KL, cluster_wass = self.get_cluster_info(query_emb)
        word_KL, word_wass = self.get_cluster_info(query_emb_high)

        target_logits = self.logits[75:] # target to support

        # select query T'   #找到最初的target伪标签
        results = self.get_target_pattern(target_logits, query_emb_high, support_emb_high)

        # ===========保存参数logits  query_emb_high  support_emb_high=========== #
        if opt.save_para_flag and opt.save_epoch_item >= 0 and ((opt.asp_count % opt.per_save_count == 0) or (opt.asp_count == 1)):
            self.save_mid_para(opt, 'logits_before', self.logits, tsf_iter=-1)
            self.save_mid_para(opt, 'query_emb_high', query_emb_high, tsf_iter=-1)
            self.save_mid_para(opt, 'support_emb_high', support_emb_high, tsf_iter=-1)
        # ===========保存参数=========== #

        patterns, target_ep, target_ep_label = results["patterns"], results["target_ep"], results["target_ep_label"]
        # ===========保存参数target_ep, target_ep_label  看迭代标签的变化=========== #
        if opt.save_para_flag and opt.save_epoch_item >= 0 and ((opt.asp_count % opt.per_save_count == 0) or (opt.asp_count == 1)):
            self.save_mid_para(opt, 'patterns', patterns, tsf_iter=20)  # 此处用20表示第一次寻找target伪标签
            self.save_mid_para(opt, 'target_ep', target_ep, para_type='list', tsf_iter=20)
            self.save_mid_para(opt, 'target_ep_label', target_ep_label, para_type='list', tsf_iter=20)
        # ===========保存参数=========== #
        # self.weight_enrase()
        for i in range(self.iter):  # 寻找的迭代次数
            results = self.iter_get_target_pattern(target_logits, query_emb_high, support_emb_high, target_ep, target_ep_label)
            patterns, target_ep, target_ep_label = results["patterns"], results["target_ep"], results["target_ep_label"]
            # self.weight_enrase()

            # ===========保存参数target_ep, target_ep_label  看迭代标签的变化=========== #
            if opt.save_para_flag and opt.save_epoch_item >= 0 and ((opt.asp_count % opt.per_save_count == 0) or (opt.asp_count == 1)):
                self.save_mid_para(opt, 'patterns', patterns, tsf_iter=i)
                self.save_mid_para(opt, 'target_ep', target_ep, para_type='list', tsf_iter=i)
                self.save_mid_para(opt, 'target_ep_label', target_ep_label, para_type='list', tsf_iter=i)
            # ===========保存参数=========== #

        encoder_res = self.encoder(patterns)
        simi_vectors, target_logits = encoder_res['patterns'], encoder_res['logits']
        triplet_loss = self.get_triplet_loss(target_logits)
        # experiment index to be showed
        self.logits[75:] = target_logits
        self.set_meter('semi_kl', semi_kl.clone().detach())
        self.set_meter('cluster_KL', cluster_KL.clone().detach())
        self.set_meter('word_KL', word_KL.clone().detach())
        self.set_meter('triplet_loss', triplet_loss.clone().detach())
        return {
            'logits': self.logits, 'rspa_loss': rspa_loss,
            'cluster_KL':cluster_KL, 'semi_kl': semi_kl, 'word_KL': word_KL,
            'cluster_wass': cluster_wass, 'pattern_l2': pattern_l2, 'triplet_loss': triplet_loss
            }

    def save_mid_para(self, opt, para_name, para, para_type='tensor', tsf_iter=-1):
        """
            IMSE/ASP----
                        epoch----
                                train
                                validate
                                test
                                meta_test----
                                             5类1shot的相似度矩阵  (150, 5, 100, 100), 需要append
                                             稀疏化的5类1shot的相似度矩阵  (150, 5, 100, 100), 需要append
                                             高斯核（ASP需要保存没轮的sigma和kernel, IMSE的不会变, 保不保存都可以） 大小不统一
                                             高斯滤波后的相似度矩阵---- 大小不统一  相同类不同轮数需要append
                                                                   记录不同的轮数
                                             ---ASP中5个高斯核融合后的结果----  相同类不同轮数需要append
                                                                       记录不同的轮数
                                             池化后的相似度矩阵----  相同类不同轮数需要append
                                                                记录不同的轮数(与高斯轮数应该是对应的)
                                             保存最终SPs结果   (150, 5, 100), 需要append
                                             ---ASP中经过msk的SPs结果   (150, 5, 100), 需要append
        """
        # root = '/home/lyz/NAS/lyz/pyProject/experiment_results/TSFL_Broadcast/save_para/'
        root = opt.save_para_img_path
        if opt.only_test_flag:  # 只测试时保存地址
            root = opt.save_test_para_img_path
        # 新建三个文件夹IMSE(metric)、mode_type、para_name
        metric_path = os.path.join(root, f'{opt.metric}')
        if not os.path.exists(metric_path):
            os.makedirs(metric_path)
        ### ====== 感觉就不用记录epoch了, 因为asp_count已经算了所有轮数 ====== ###
        # if opt.only_test_flag:  # 只测试时需要记录轮数, 感觉也不需要啊
        epoch_path = os.path.join(metric_path, f'{opt.save_epoch_item}')
        if not os.path.exists(epoch_path):
            os.makedirs(epoch_path)
        epoch_path = metric_path
        mode_path = os.path.join(epoch_path, f'{opt.mode_type}')
        if not os.path.exists(mode_path):
            os.makedirs(mode_path)
        para_path = os.path.join(mode_path, f'{para_name}')
        if not os.path.exists(para_path):
            os.makedirs(para_path)
        # if type(para) is not torch.Tensor:  # 不是tensor的转成tensor
        #     para = torch.tensor(para)
        if tsf_iter != -1:  # 对于多轮高斯滤波, 里面再新建一个轮数文件夹
            para_path = os.path.join(para_path, f'{tsf_iter}')
            if not os.path.exists(para_path):
                os.makedirs(para_path)
        if para_type == 'tensor':
            torch.save(para.cpu(), os.path.join(para_path, f"{opt.asp_count}.pt"))
        elif para_type == 'list':
            # 都搞成tensor, 统一处理
            torch.save(torch.tensor(para), os.path.join(para_path, f"{opt.asp_count}.pt"))
            # np.save(os.path.join(epoch_path, f"{opt.asp_count}.npy"), para)
        else:
            raise ValueError("'para_type' must be 'tensor' or 'list'!")


class TSFL_BroadcastV7(TSFL_BroadcastV6):
    def __init__(self, cluster_num, component_size=2, sigma=0.3,
                neighbor_k=3, kl_eps=1e-2, word_eps=0,
                KL_margin=1, semi_margin=1, basis_thredshold=50, epsilon=2.5,
                weight_en=0.1, tri_margin=1., select_iter=2, largest_num=100, scale=1.):
        super(TSFL_BroadcastV7, self).__init__(cluster_num, component_size,
                                                      sigma, neighbor_k, kl_eps,
                                                      word_eps, KL_margin, semi_margin,
                                                      basis_thredshold, epsilon,
                                                      weight_en, tri_margin, select_iter)
        self.crossattention = CrossAttention(640, 8, largest_num,False, scale)

    def forward_(self, query, support, train=True):
        self.train_mode = train
        way, shot, _, _, _ = support.size()
        self.way_num = way
        qm, C, H, W = query.size()

        query_t = self.get_queryt(query, train)
        (clusters, _), _ = self.get_bow(query_t, support)
        if train:
            clusters = self.crossattention(clusters)

        self.clusters = clusters.size(0)

        # protect raw features to the word space
        query_emb, support_emb = self.project_to_wordspace(clusters, query, support)

        # pass the BN
        emb_data = self.emb_bn1(torch.cat([query_emb, support_emb]).permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        query_emb, support_emb = emb_data[:qm], emb_data[qm:].view(way, -1, self.clusters)

        # compute word-block-level features
        query_emb_high = self.combine_words(query_emb)
        support_emb_high = self.combine_words(support_emb.view(way*shot, H*W, -1)).view(way*shot, -1, self.component_size**2*self.clusters)
        support_emb_high = self.get_prototype(support_emb_high, way, shot)
        
        # pass the BN
        componet_data = self.emb_bn2(query_emb_high, support_emb_high)
        query_emb_high, support_emb_high = componet_data[:qm], componet_data[qm:].contiguous().view(way, -1, self.component_size**2*self.clusters)

        # pass the IMSE
        patterns = self.get_patterns(query_emb_high, support_emb_high)

        # simi_vectors: (qm, shot, way, H*W)
        encoder_res = self.encoder(patterns)
        simi_vectors, self.logits = encoder_res['patterns'], encoder_res['logits']
        pattern_l2 = simi_vectors.pow(2).sqrt().sum()
        rspa_loss, semi_kl = self.get_cov_loss(simi_vectors)
        
        # process logits
        self.process_logits(support_emb)
        
        # compute the KL for the domain-shared word space
        cluster_KL, cluster_wass = self.get_cluster_info(query_emb)
        word_KL, word_wass = self.get_cluster_info(query_emb_high)

        target_logits = self.logits[75:] # target to support
    
        # select query T' 
        results = self.get_target_pattern(target_logits, query_emb_high, support_emb_high)
        patterns, target_ep, target_ep_label = results["patterns"], results["target_ep"], results["target_ep_label"]
        self.weight_enrase()
        for i in range(self.iter): 
            results = self.iter_get_target_pattern(target_logits, query_emb_high, support_emb_high, target_ep, target_ep_label)
            patterns, target_ep, target_ep_label = results["patterns"], results["target_ep"], results["target_ep_label"]
            self.weight_enrase()
        encoder_res = self.encoder(patterns)
        simi_vectors, target_logits = encoder_res['patterns'], encoder_res['logits']
        triplet_loss = self.get_triplet_loss(target_logits)
        # experiment index to be showed
        self.logits[75:] = target_logits
        self.set_meter('semi_kl', semi_kl.clone().detach())
        self.set_meter('cluster_KL', cluster_KL.clone().detach())
        self.set_meter('word_KL', word_KL.clone().detach())
        self.set_meter('triplet_loss', triplet_loss.clone().detach())
        return {
            'logits': self.logits, 'rspa_loss': rspa_loss, 
            'cluster_KL':cluster_KL, 'semi_kl': semi_kl, 'word_KL': word_KL,
            'cluster_wass': cluster_wass, 'pattern_l2': pattern_l2, 'triplet_loss': triplet_loss
            }


class TSFL_BroadcastV8(TSFL_BroadcastV6):
    def __init__(self, cluster_num, component_size=2, sigma=0.3, 
                neighbor_k=3, kl_eps=1e-2, word_eps=0,
                KL_margin=1, semi_margin=1, basis_thredshold=50, epsilon=2.5, 
                weight_en=0.1, tri_margin=1., select_iter=2, largest_num=100, scale=1.):
        super(TSFL_BroadcastV8, self).__init__(cluster_num, component_size, 
                                                      sigma, neighbor_k, kl_eps, 
                                                      word_eps, KL_margin, semi_margin, 
                                                      basis_thredshold, epsilon, 
                                                      weight_en, tri_margin, select_iter)
        
    def get_bow(self, query_t, support):
        _, C, H, W = query_t.size()
        # x = torch.cat([query_t, support.view(-1, C, H, W)]).permute(0, 2, 3, 1).contiguous().view(-1, C)
        input_data = []
        
        for c in range(len(support)):
            x = support[c].view(-1, C, H*W).permute(0, 2, 1).contiguous().view(-1, C)
            input_data.append(x)
        input_data.append(query_t.view(-1, C, H*W).permute(0, 2, 1).contiguous().view(-1, C))
        bows, sv_means = self.K_means(torch.cat(input_data, dim=0))
        
        # sv_means = torch.stack(sv_means)
        # bows = torch.cat(bows)
        self.set_meter('word_num', torch.tensor([bows.size(0)]))
        self.set_meter('sv_means', sv_means.mean())
        return (bows, sv_means), _


class TSFL_BroadcastV9(TSFL_BroadcastV6):
    def __init__(self, cluster_num, component_size=2, sigma=0.3, 
                neighbor_k=3, kl_eps=1e-2, word_eps=0,
                KL_margin=1, semi_margin=1, basis_thredshold=50, epsilon=2.5, 
                weight_en=0.1, tri_margin=1., select_iter=2, largest_num=100, scale=1., bias=0.):
        super(TSFL_BroadcastV9, self).__init__(cluster_num, component_size, 
                                                      sigma, neighbor_k, kl_eps, 
                                                      word_eps, KL_margin, semi_margin, 
                                                      basis_thredshold, epsilon, 
                                                      weight_en, tri_margin, select_iter)
        self.K_means = DynamicKmeansV3(cluster_num, basis_thredshold=basis_thredshold, largest_num=largest_num, scale=scale, bias=bias)

    
class TSFL_BroadcastV10(TSFL_BroadcastV6):   # 效果最好
    def __init__(self, cluster_num, component_size=2, sigma=0.3, 
                neighbor_k=3, kl_eps=1e-2, word_eps=0,
                KL_margin=1, semi_margin=1, basis_thredshold=50, epsilon=2.5, 
                weight_en=0.1, tri_margin=1., select_iter=2, scale=1.):
        super(TSFL_BroadcastV10, self).__init__(cluster_num, component_size, 
                                                sigma, neighbor_k, kl_eps, 
                                                word_eps, KL_margin, semi_margin, 
                                                basis_thredshold, epsilon, 
                                                weight_en, tri_margin, select_iter)
        self.K_means = DynamicKmeansV4(cluster_num, basis_thredshold=basis_thredshold, scale=scale)

class TSFL_BroadcastV11(TSFL_BroadcastV6):
    def __init__(self, cluster_num, component_size=2, sigma=0.3, 
                neighbor_k=3, kl_eps=1e-2, word_eps=0,
                KL_margin=1, semi_margin=1, basis_thredshold=50, epsilon=2.5, 
                weight_en=0.1, tri_margin=1., select_iter=2, scale=1., bias=0.):
        super(TSFL_BroadcastV11, self).__init__(cluster_num, component_size, 
                                                      sigma, neighbor_k, kl_eps, 
                                                      word_eps, KL_margin, semi_margin, 
                                                      basis_thredshold, epsilon, 
                                                      weight_en, tri_margin, select_iter)
        self.K_means = DynamicKmeansV5(cluster_num, basis_thredshold=basis_thredshold, scale=scale, bias=bias)
        