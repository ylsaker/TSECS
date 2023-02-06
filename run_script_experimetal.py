from numpy import False_
from experiment_manage import ExperimentServer
from main import main
import argparse, os
import make_args

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0')
parser.add_argument('--metric', default='TSFL_BroadcastV6')
p_args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = p_args.gpu

parameters_dir = {
    'metric': [p_args.metric], 'data_name': ['visda'], 'backbone': ['ResNet12'],
    'shot_num': [5], 'epochs': [6],
    'way_num': [5],
    # 'query_num': [10],
    'interim': [1],
    'random_target': [False],  
    'episode_train_num': [280],
    'episode_val_num': [100],
    'episode_test_num': [100],
    'print_freq': [50],
    # set data and domain
    'workers': [12],
    'source_domain': ['sketch'],
    'target_domain': ['clipart'],
    # 'target_domain': ['clipart', 'painting'],

    # losses' weights
    'loss_weight': [1.],
    'bow_gloss': [0],
    'componet_num': [10],
    'soft_weight': [0.],
    'cmp_cov_loss': [0],
    'cluster_KL': [0.06163],
    'triplet_loss': [0.01947416970785845], # broadcast 0.0133  三元组损失
    'target_loss':[0.],

    'semi_kl': [0.11],
    'kl_eps': [0.001],
    'semi_margin': [0],
    'tri_margin': [1.997668748230611], # align support 2.03 align source center 1.797  三元组损失marhin

    # cross attention
    'largest_num':[160.],  # 不用参数
    'scale':[2.57],  # 仿射变换乘
    'bias': [67.],  # 仿射变换加
    # use DA technique
    'bow_DA': [False],
    'DA': [True],
    'temperature': [2.39], 
    # select target
    'epsilon': [2.1742288924764392], # align support 2.29  target作为伪标签的阈值
    'select_iter': [1], # 选择目标域类中心的轮数
    'weight_en': [1e-3],
    # model structures
    'basis_thredshold': [0.7],
    # 'template_num': [5],
    'sigma': [0.5],
    'ld_num': [100],
    'discri': ['LD'],
    'kernel_size': [1],
    'neighbor_k': [3],

    # set optimizer
    'SGD': [False],
    'iter': [1],
    'lr': [	0.00019],
    'ilr': [0.00135],
    'drop_rate': [0.],

    # set bn
    'on_bn': [True],
    'bn_epoch': [10],
    'pretrained': [True],

}

para_spaces = {
    'cluster_KL' : {
        '_type': 'uniform',
        '_value': [1e-2, 1e-1]
    },
    'semi_kl' : {
        '_type': 'uniform',
        '_value': [1e-1, 0.5]
    },
}

exp_server = ExperimentServer(parameters_dir, main, False, para_spaces, 'path to experiment_result')
exp_server.opt.gpu = p_args.gpu
exp_server.run()
