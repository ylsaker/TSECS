import argparse

import torch.backends.cudnn as cudnn
import os
def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='/home/hsq/Projects/datasets/MMMMdataset', help='/miniImageNet')
    parser.add_argument('--split_PATH', type=str, default='/home/hsq/Projects/mini-imagenet-split')

    parser.add_argument('--data_name', default='officehome', help='miniImageNet|StanfordDog|StanfordCar|CubBird')
    parser.add_argument('--mode', default='train', help='train|val|test')
    parser.add_argument('--outf', default='./results/DN4')
    parser.add_argument('--resume', default='', type=str, help='path to the lastest checkpoint (default: none)')
    parser.add_argument('--basemodel', default='Conv64F', help='Conv64F|ResNet256F')
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--s2p', type=str, default='False')

    #  Few-shot parameters  #final_test_episode_num
    parser.add_argument('--imageSize', type=int, default=84)
    parser.add_argument('--episodeSize', type=int, default=1, help='the mini-batch size of training')
    parser.add_argument('--testepisodeSize', type=int, default=1, help='one episode is taken as a mini-batch')
    parser.add_argument('--epochs', type=int, default=10, help='the total number of training epoch')
    parser.add_argument('--final_test_episode_num', type=int, default=600, help='the total number of training episodes')
    parser.add_argument('--episode_train_num', type=int, default=1000, help='the total number of training episodes')
    parser.add_argument('--episode_val_num', type=int, default=500, help='the total number of evaluation episodes')
    parser.add_argument('--episode_test_num', type=int, default=500, help='the total number of testing episodes')
    parser.add_argument('--way_num', type=int, default=5, help='the number of way/class')
    parser.add_argument('--shot_num', type=int, default=5, help='the number of shot')
    parser.add_argument('--query_num', type=int, default=15, help='the number of queries')
    parser.add_argument('--neighbor_k', type=int, default=3, help='the number of k-nearest neighbors')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.005')
    parser.add_argument('--ilr', type=float, default=0.0001, help='learning rate, default=0.005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='the number of gpus')
    parser.add_argument('--nc', type=int, default=3, help='input image channels')
    parser.add_argument('--clamp_lower', type=float, default=-0.01)
    parser.add_argument('--clamp_upper', type=float, default=0.01)
    parser.add_argument('--print_freq', '-p', default=100, type=int, metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--gpu', type=str, default='0,1,2,3')
    parser.add_argument('--metric', type=str, default='ImgtoClassSpatial_V19Cov')
    parser.add_argument('--discri', type=str, default='LD')
    # parser.add_argument('--initial', type=str, default='1')
    parser.add_argument('--pretrain', type=int, default=0)
    parser.add_argument('--interim', type=int, default=1)
    parser.add_argument('--aug_support', type=str, default='False')
    parser.add_argument('--target_topk', type=int, default=10)
    parser.add_argument('--ld_num', type=int, default=25)

    parser.add_argument('--usa', type=str, default='v3')
    parser.add_argument('--decay_rate', type=float, default=0.5)
    parser.add_argument('--cov_weight', type=float, default=100)
    parser.add_argument('--multigpu', type=bool, default=False)

    parser.add_argument('--softlabel', default=False)
    parser.add_argument('--use_gda', type=str, default='False')
    parser.add_argument('--cov_align', default=False)
    parser.add_argument('--pretrained', default=True)
    parser.add_argument('--backbone', type=str, default='ResNet12')

    parser.add_argument('--loss_weight', type=float, default=0.95)
    parser.add_argument('--soft_weight', type=float, default=0.5)
    parser.add_argument('--source_domain', type=str, default='')
    parser.add_argument('--target_domain', type=str, default='')
    parser.add_argument('--DA', default=False)
    parser.add_argument('--combine_uda_fsl', default=False)
    parser.add_argument('--sigma', type=float, default=0.8)
    parser.add_argument('--affine', type=float, default=0.8)
    parser.add_argument('--gaus', type=float, default=0.8)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--iter', type=int, default=16)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--select_n', type=int, default=10)
    parser.add_argument('--pow', type=int, default=2)
    parser.add_argument('--task_batch', type=int, default=1)
    parser.add_argument('--fg_loss', type=bool, default=False)
    parser.add_argument('--fg_loss_weight', type=float, default=1)
    parser.add_argument('--cov_class_softmax_loss', default=False)

    parser.add_argument('--on_bn', default=False)
    parser.add_argument('--cluster', default=False)
    parser.add_argument('--classification', default=False)
    parser.add_argument('--cluster_loss', default=False)
    parser.add_argument('--rotate', default=False)
    parser.add_argument('--finetune', default=False)

    opt = parser.parse_args()
    opt.cuda = True
    cudnn.benchmark = True
    # os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    if opt.aug_support == 'True':
        opt.aug_support = True
    else:
        opt.aug_support = False
    if opt.s2p == 'True':
        opt.s2p = True
    else:
        opt.s2p = False

    if not opt.pretrained:
        opt.outf += '_pretrained'

    return opt