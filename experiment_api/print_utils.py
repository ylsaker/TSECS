import torch
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, n=15):
        self.n = n
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.histories = []

    def epoch_avg(self, epo=-1):
        return sum(self.histories[epo])/len(self.histories[epo])

    def update(self, val, epoch_sof=False):
        if epoch_sof:
            self.histories.append([val])
        else:
            self.histories[-1].append(val)
        self.val = val
        self.sum += val * self.n
        self.count += self.n
        self.avg = self.sum / self.count


def print_and_record(text, outf):
    print(text)
    with open(outf, 'a+') as f:
        f.write(text + '\n')



def accumulate_and_output_meters(loss_acc, meters_dict, epoch_index, episode_index, episodelen, mode):
    for ind_k in loss_acc:
        v = loss_acc[ind_k]
        if ind_k not in meters_dict:
            meters_dict[ind_k] = AverageMeter()
        try:
            meters_dict[ind_k].update(v.squeeze().item(), epoch_sof=episode_index == 0)
        except Exception as E:
            print(E)
            print(ind_k)
            exit()


    # measure elapsed time
    out_line = '{}-({}): \t[{}/{}] \t'.format(mode, epoch_index, episode_index, episodelen)
    if episode_index == 1:
        out_line = '===================\tEPOCH {} {}ing\t====================\n'.format(epoch_index, mode) + out_line
    for ind_k in meters_dict:
        # out_line += '{}: {:.2f} \t'.format(ind_k, meters_dict[ind_k].epoch_avg())
        out_line += '{}: {:.2f} \t'.format(ind_k, meters_dict[ind_k].avg)
    return out_line


if __name__ == '__main__':
    keys = ['loss', 'acc', 'kl', 'time', 'pic']
    meters_dict = {}
    episodelen = 20
    for i in range(episodelen):
        loss_acc = {}
        for k in keys:
            loss_acc[k] = torch.randn(1)
        accumulate_and_output_meters(loss_acc, meters_dict, 0, i, episodelen, 'test')
    for i in range(episodelen):
        loss_acc = {}
        for k in keys:
            loss_acc[k] = torch.randn(1)
        accumulate_and_output_meters(loss_acc, meters_dict, 0, i, episodelen, 'test')
    for i in range(episodelen):
        loss_acc = {}
        for k in keys:
            loss_acc[k] = torch.randn(1)
        accumulate_and_output_meters(loss_acc, meters_dict, 0, i, episodelen, 'test')
