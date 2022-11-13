from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import numpy as np
import math, random

def str2bool(str):
    return True if str.lower() == 'true' else False

def add_sparse_args(parser):
    parser.add_argument('--sparse', type=str2bool, default=True, help='Enable sparse mode. Default: True.')
    parser.add_argument('--adv', type=bool, default=False, help='adv sparse mode. Default: True.')
    parser.add_argument('--last_dense', type=str2bool, default=False, help='adv sparse mode. Default: True.')
    parser.add_argument('--init-prune-epoch', type=int, default=0, help='The pruning rate / death rate.')
    parser.add_argument('--final-prune-epoch', type=int, default=1000, help='The density of the overall sparse network.')
    parser.add_argument('--fix', type=str2bool, default=False, help='Fix sparse connectivity during training. Default: True.')
    parser.add_argument('--sparse_init', type=str, default='remain_sort', help='sparse initialization: ERK, snip, Grasp')
    parser.add_argument('--growth', type=str, default='random', help='Growth mode. Choose from: momentum, random, random_unfired, and gradient.')
    parser.add_argument('--death', type=str, default='magnitude', help='Death mode / pruning mode. Choose from: magnitude, SET, threshold.')
    parser.add_argument('--redistribution', type=str, default='none', help='Redistribution mode. Choose from: momentum, magnitude, nonzeros, or none.')
    parser.add_argument('--death-rate', type=float, default=0.5, help='The pruning rate / death rate.')
    parser.add_argument('--density', type=float, default=0.2, help='The density of the overall sparse network.')
    parser.add_argument('--final_density', type=float, default=0.1, help='The density of the overall sparse network.')
    parser.add_argument('--update_frequency', type=int, default=5, metavar='N', help='how many iterations to train between parameter exploration')
    parser.add_argument('--decay-schedule', type=str, default='cosine', help='The decay schedule for the pruning rate. Default: cosine. Choose from: cosine, linear.')
    parser.add_argument('--keep_size', type=int, default=5, help='The pruning rate / death rate.')
    parser.add_argument('--c_size', type=int, default=3, help='The pruning rate / death rate.')
class CosineDecay(object):
    def __init__(self, death_rate, T_max, eta_min=0.001, last_epoch=-1):
        self.sgd = optim.SGD(torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=death_rate)
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(self.sgd, T_max, eta_min, last_epoch)

    def step(self):
        self.cosine_stepper.step()

    def get_dr(self):
        return self.sgd.param_groups[0]['lr']

class LinearDecay(object):
    def __init__(self, death_rate, factor=0.99, frequency=600):
        self.factor = factor
        self.steps = 0
        self.frequency = frequency

    def step(self):
        self.steps += 1

    def get_dr(self, death_rate):
        if self.steps > 0 and self.steps % self.frequency == 0:
            return death_rate*self.factor
        else:
            return death_rate


class Masking(object):
    def __init__(self, optimizer, death_rate=0.3, growth_death_ratio=1.0, death_rate_decay=None, death_mode='magnitude', growth_mode='momentum', redistribution_mode='momentum', threshold=0.001, train_loader=None, T_max=0., args=None):
        growth_modes = ['random', 'momentum', 'momentum_neuron', 'gradient']
        if growth_mode not in growth_modes:
            print('Growth mode: {0} not supported!'.format(growth_mode))
            print('Supported modes are:', str(growth_modes))

        self.args = args
        self.device = torch.device("cuda")
        self.growth_mode = growth_mode
        self.death_mode = death_mode
        self.growth_death_ratio = growth_death_ratio
        self.redistribution_mode = redistribution_mode
        self.death_rate_decay = death_rate_decay
        self.last_dense = args.last_dense

        self.masks = {}
        self.modules = []
        self.names = []
        self.optimizer = optimizer

        # stats
        self.name2zeros = {}
        self.num_remove = {}
        self.name2nonzeros = {}
        self.death_rate = death_rate
        self.baseline_nonzero = None
        self.steps = 0
        self.explore_step = 0

        self.pruned_masks = {}
        self.regrowed_masks = {}
        self.pre_masks = None
        self.decay_flag = True

        self.total_nozeros = 0
        self.total_weights = 0
        self.loader = train_loader
        self.regrow_ratio = 1.01
        self.adv = self.args.adv
        self.curr_density = 0.0
        self.regrow_ones = 0
        self.T_max = T_max

        self.num_rm_list = {}

        # if fix, then we do not explore the sparse connectivity
        if self.args.fix:
            self.prune_every_k_steps = None
        else:
            self.prune_every_k_steps = self.args.update_frequency * len(self.loader)

    def init(self, mode='ERK', density=0.05, erk_power_scale=1.0):
        self.density = density
        if mode == 'GMP':
            self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name] = torch.ones_like(weight, dtype=torch.float32, requires_grad=False).cuda()
                    self.baseline_nonzero += (self.masks[name] != 0).sum().int().item()

        elif mode == 'remain_random':
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    mask = self.masks[name]
                    if mask.shape[-1] < 5:
                        ones_mask = torch.ones_like(mask, dtype=torch.float32).cuda()
                        self.masks[name][:] = ones_mask
                        continue

                    nn = int(self.args.k_size / self.args.c_size)
                    k_s = [nn * (i + 1) for i in range(self.args.c_size)]
                    mask_chunk = torch.chunk(mask, self.args.c_size, dim=0)
                    mask_list = []
                    for i, w_c in enumerate(mask_chunk):
                        kernel_size = k_s[i]
                        search_num = (mask_chunk[i] == 0).sum().item()*kernel_size/self.args.k_size
                        num_remain = int(search_num*self.density)

                        ind = list(range(kernel_size))
                        x = torch.rand(mask_chunk[i].shape).cuda()
                        x[:, :, ind] += 10.0

                        new_mask = torch.zeros_like(x, dtype=torch.float32).cuda()
                        y, idx = torch.sort(torch.abs(x).flatten(), descending=True)  ## big to small
                        new_mask.data.view(-1)[idx[:num_remain]] = 1.0

                        mask_list.append(new_mask)

                    mask_new = torch.cat(tuple(mask_list), 0)
                    self.masks[name][:] = mask_new

        elif mode == 'remain_sort':
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    mask = self.masks[name]
                    if mask.shape[-1] < 5:
                        ones_mask = torch.ones_like(mask, dtype=torch.float32).cuda()
                        self.masks[name][:] = ones_mask
                        continue

                    nn = int(self.args.k_size / self.args.c_size)
                    k_s = [nn * (i + 1) for i in range(self.args.c_size)]
                    mask_chunk = torch.chunk(mask, self.args.c_size, dim=0)
                    mask_list = []
                    for i, w_c in enumerate(mask_chunk):
                        kernel_size = k_s[i]
                        search_num = (mask_chunk[i] == 0).sum().item()*kernel_size/self.args.k_size
                        num_remain = int(search_num*self.density)

                        x = torch.rand(mask_chunk[i].shape).cuda()
                        x = x.permute(2, 0, 1)

                        new_mask = torch.zeros_like(x, dtype=torch.float32).cuda()
                        new_mask.data.view(-1)[:num_remain] = 1.0

                        new_mask = new_mask.permute(1, 2, 0)
                        mask_list.append(new_mask)

                    mask_new = torch.cat(tuple(mask_list), 0)
                    self.masks[name][:] = mask_new

        self.apply_mask()
        self.fired_masks = copy.deepcopy(self.masks) # used for ITOP
        # self.print_nonzero_counts()

        total_size = 0
        for name, weight in self.masks.items():
            total_size  += weight.numel()

        sparse_size = 0
        for name, weight in self.masks.items():
            sparse_size += (weight != 0).sum().int().item()


    def step(self):
        self.optimizer.step()
        self.apply_mask()

        if self.decay_flag:
            self.death_rate_decay.step()
            self.death_rate = self.death_rate_decay.get_dr()
        else:
            self.death_rate = 0.001

        self.steps += 1

        if self.prune_every_k_steps is not None:

            if self.steps % self.prune_every_k_steps == 0:
                ## low to high regrow
                self.explore_step += 1
                self.truncate_weights_channel()

                self.cal_nonzero_counts()
                self.curr_density = self.total_nozeros / self.total_weights
                print('curr_density: {0:.4f}, final_density:{1:.4f}'.format(self.curr_density, self.args.final_density))
                print('curr_w_density: {0:.4f}'.format(self.w_ratio))

                _, _ = self.fired_masks_update()
                if self.explore_step > 1:
                    self.print_nonzero_counts()
                self.pre_masks = copy.deepcopy(self.pruned_masks)


    def add_module(self, module, density, sparse_init='ER'):
        self.modules.append(module)
        self.module = module
        for name, tensor in module.named_parameters():
            if 'conv' not in name:
                continue
            self.names.append(name)
            self.masks[name] = torch.zeros_like(tensor, dtype=torch.float32, requires_grad=False).cuda()

        print('Removing biases...')
        self.remove_weight_partial_name('bias')
        print('Removing 2D batch norms...')
        self.remove_type(nn.BatchNorm2d)
        print('Removing 1D batch norms...')
        self.remove_type(nn.BatchNorm1d)
        self.init(mode=sparse_init, density=density)

    def cal_nonzero_counts(self):
        self.total_nozeros = 0
        self.total_weights = 0

        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                self.total_nozeros += (mask != 0).sum().item()
                self.total_weights += mask.numel()

                ## get weight density ratio
                if mask.shape[-1] > 5:
                    nozeronum = (tensor.data != 0).sum().item()
                    numele = tensor.data.nelement()
                    self.w_ratio = nozeronum / numele
                    aa = 0

    def remove_weight(self, name):
        if name in self.masks:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name].shape,
                                                                      self.masks[name].numel()))
            self.masks.pop(name)
        elif name + '.weight' in self.masks:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name + '.weight'].shape,
                                                                      self.masks[name + '.weight'].numel()))
            self.masks.pop(name + '.weight')
        else:
            print('ERROR', name)

    def remove_weight_partial_name(self, partial_name):
        removed = set()
        for name in list(self.masks.keys()):
            if partial_name in name:

                print('Removing {0} of size {1} with {2} parameters...'.format(name, self.masks[name].shape,
                                                                                   np.prod(self.masks[name].shape)))
                removed.add(name)
                self.masks.pop(name)

        print('Removed {0} layers.'.format(len(removed)))

        i = 0
        while i < len(self.names):
            name = self.names[i]
            if name in removed:
                self.names.pop(i)
            else:
                i += 1

    def remove_type(self, nn_type):
        for module in self.modules:
            for name, module in module.named_modules():
                if isinstance(module, nn_type):
                    self.remove_weight(name)

    def masks_reset(self):
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                self.masks[name] = torch.zeros_like(tensor, dtype=torch.float32, requires_grad=False).cuda()

        print('Masks Reseted...')

    def apply_mask(self):
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name in self.masks:
                    tensor.data = tensor.data*self.masks[name]
                    # reset momentum
                    if 'momentum_buffer' in self.optimizer.state[tensor]:
                        self.optimizer.state[tensor]['momentum_buffer'] = self.optimizer.state[tensor]['momentum_buffer']*self.masks[name]

    def truncate_weights_channel(self):
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                self.name2nonzeros[name] = mask.sum().item()
                self.name2zeros[name] = mask.numel() - self.name2nonzeros[name]

                if mask.shape[-1]<5: continue

                # death
                if self.death_mode == 'magnitude':
                    new_mask, remove_list = self.magnitude_death_channel(mask, weight, name, c_size=self.args.c_size)
                elif self.death_mode == 'SET':
                    new_mask = self.magnitude_and_negativity_death(mask, weight, name)
                elif self.death_mode == 'Taylor_FO':
                    new_mask = self.taylor_FO(mask, weight, name)
                elif self.death_mode == 'threshold':
                    new_mask = self.threshold_death(mask, weight, name)

                self.num_rm_list[name] = remove_list
                self.masks[name][:] = new_mask

                ## pick up the remain weights(before regrow)
                self.pruned_masks[name] = new_mask

        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                new_mask = self.masks[name].data.byte()
                if self.masks[name].shape[-1] < 5: continue

                # growth
                if self.growth_mode == 'random':
                    new_mask = self.random_growth_channel(name, new_mask, weight, c_size=self.args.c_size, num_rm_list=self.num_rm_list[name])
                    # new_mask = self.gradient_growth_channel(name, new_mask, weight, c_size=self.args.c_size,
                    #                                       num_rm_list=self.num_rm_list[name])

                elif self.growth_mode == 'random_unfired':
                    new_mask = self.random_unfired_growth(name, new_mask, weight)

                elif self.growth_mode == 'momentum':
                    new_mask = self.momentum_growth(name, new_mask, weight)

                elif self.growth_mode == 'gradient':
                    new_mask = self.gradient_growth_channel(name, new_mask, weight, c_size=self.args.c_size, num_rm_list=self.num_rm_list[name])

                new_nonzero = new_mask.sum().item()

                # exchanging masks
                self.masks.pop(name)
                self.masks[name] = new_mask.float()

                ## pick up the remain weights(after regrow)
                self.regrowed_masks[name] = new_mask.float()

        # self.apply_mask()

    '''
                    DEATH
    '''

    def threshold_death(self, mask, weight, name):
        return (torch.abs(weight.data) > self.threshold)

    def taylor_FO(self, mask, weight, name):

        num_remove = math.ceil(self.death_rate * self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]
        k = math.ceil(num_zeros + num_remove)

        x, idx = torch.sort((weight.data * weight.grad).pow(2).flatten())
        mask.data.view(-1)[idx[:k]] = 0.0

        return mask

    def magnitude_death(self, mask, weight, name):

        num_remove = math.ceil(self.death_rate*self.name2nonzeros[name])
        if num_remove == 0.0: return weight.data != 0.0
        num_zeros = self.name2zeros[name]

        x, idx = torch.sort(torch.abs(weight.data.view(-1)))
        n = idx.shape[0]

        k = math.ceil(num_zeros + num_remove)
        threshold = x[k-1].item()

        return (torch.abs(weight.data) > threshold)

    def magnitude_death_fix(self, mask, weight, name, ori_size=3):
        in_c, ou_c, k_s = mask.shape
        one_index= int((k_s-1)/2)
        remove_size = 0

        mask_new = torch.zeros_like(weight.data, dtype=torch.float32, requires_grad=False).cuda()
        mask_new[:, :, one_index] = 1.0
        mask_new[:, :, 0] = 1.0
        mask_new[:, :, -1] = 1.0

        return mask_new, remove_size

    def magnitude_death_kernel(self, mask, weight, name, ori_size=3):
        keep_size = math.ceil((1.0-self.death_rate) * ori_size)
        # keep_size = 1
        remove_size = ori_size - keep_size
        if mask.shape[-1] <= ori_size:
            return mask, 0
        if remove_size == 0:
            return mask, remove_size

        mask_new = torch.zeros_like(weight.data, dtype=torch.float32, requires_grad=False).cuda()
        value, index_ = torch.topk(torch.abs(weight.data), keep_size, dim=-1)
        mask_new.scatter_(-1, index_, 1.0)

        return mask_new, remove_size

    def magnitude_death_channel(self, mask, weight, name, c_size=3):

        weight_chunk = torch.chunk(weight, c_size, dim=0)
        mask_chunk = list(torch.chunk(mask, c_size, dim=0))
        mask_list = []
        remove_list = []
        for i, w_c in enumerate(weight_chunk):
            num_nonzeros = (mask_chunk[i] != 0).sum().item()
            num_zeros = (mask_chunk[i] == 0).sum().item()
            num_remove = math.ceil(self.death_rate * num_nonzeros)

            x, idx = torch.sort(torch.abs(w_c.data.view(-1)))
            k = math.ceil(num_zeros + num_remove)
            threshold = x[k - 1].item()

            mask_list.append(torch.abs(w_c.data) > threshold)
            remove_list.append(num_remove)

        mask_new = torch.cat(tuple(mask_list), 0)
        return mask_new, remove_list


    def random_growth_kernel(self, name, mask, weight, ori_size=3):
        regrow_num = self.num_remove[name]
        if regrow_num == 0:
            return mask

        x = torch.rand(weight.data.shape).cuda()
        new_mask = torch.zeros_like(x, dtype=torch.float32).cuda()
        value, index_ = torch.topk(x, regrow_num, dim=-1)

        new_mask.scatter_(-1, index_, 1.0)
        new_mask_ = new_mask.byte() | mask
        return new_mask_

    def random_growth_channel(self, name, mask, weight, c_size=3, num_rm_list=None):

        # k_s = [9, 19, 39]
        nn = int(self.args.k_size / c_size)
        k_s = [nn*(i+1) for i in range(c_size)]
        weight_chunk = torch.chunk(weight, c_size, dim=0)
        mask_chunk = list(torch.chunk(mask, c_size, dim=0))
        num_rm_list = num_rm_list
        mask_list = []
        for i, w_c in enumerate(weight_chunk):
            num_remove = num_rm_list[i]
            num_remain = num_remove + (mask_chunk[i] != 0).sum().item()
            kernel_size = k_s[i]

            weight_new = copy.deepcopy(w_c.data) + 20.
            weight_new = weight_new * mask_chunk[i]

            ind = list(range(kernel_size))
            x = torch.rand(mask_chunk[i].shape).cuda()
            x[:, :, ind] += 10.0
            weight_add = weight_new + x

            new_mask = torch.zeros_like(weight_add, dtype=torch.float32).cuda()
            y, idx = torch.sort(torch.abs(weight_add).flatten(), descending=True)  ## big to small
            new_mask.data.view(-1)[idx[:num_remain]] = 1.0

            mask_list.append(new_mask)

        mask_new = torch.cat(tuple(mask_list), 0)
        return mask_new

    def gradient_growth_channel(self, name, mask, weight, c_size=3, num_rm_list=None):

        # k_s = [9, 19, 39]
        nn = int(self.args.k_size / c_size)
        k_s = [nn*(i+1) for i in range(c_size)]
        grad = self.get_gradient_for_weights(weight)
        grad_chunk = torch.chunk(grad, c_size, dim=0)
        mask_chunk = list(torch.chunk(mask, c_size, dim=0))
        num_rm_list = num_rm_list
        mask_list = []
        for i, g_c in enumerate(grad_chunk):
            num_remove = num_rm_list[i]
            g_c = g_c * (mask_chunk[i] == 0).float()
            kernel_size = k_s[i]

            ind = list(range(kernel_size))
            g_c_abs = torch.abs(g_c)
            g_c_abs[:, :, ind] += 10.0

            y, idx = torch.sort(g_c_abs.flatten(), descending=True)
            mask_chunk[i].data.view(-1)[idx[:num_remove]] = 1.0

            mask_list.append(mask_chunk[i])

        mask_new = torch.cat(tuple(mask_list), 0)
        return mask_new

    def gradient_growth_kernel(self, name, mask, weight, ori_size=3):
        regrow_num = self.num_remove[name]
        if regrow_num == 0:
            return mask

        grad = self.get_gradient_for_weights(weight)
        grad = grad * (mask == 0).float()

        mask_new = torch.zeros_like(weight.data, dtype=torch.float32, requires_grad=False).cuda()
        value, index_ = torch.topk(torch.abs(grad), regrow_num, dim=-1)
        mask_new.scatter_(-1, index_, 1.0)

        new_mask_ = mask_new.byte() | mask
        return new_mask_

    def random_growth_tofix(self, name, mask, weight, ori_size=3):
        if mask.shape[-1] <= ori_size:
            keep_size = mask.shape[-1]
        else:
            keep_size = ori_size

        weight_new = copy.deepcopy(weight.data) + 10.
        weight_new = weight_new * mask
        x = torch.rand(weight.data.shape).cuda()
        weight_add = weight_new + x

        new_mask = torch.zeros_like(weight_add, dtype=torch.float32).cuda()
        value, index_ = torch.topk(weight_add, keep_size, dim=-1)
        new_mask.scatter_(-1, index_, 1.0)
        new_mask_ = new_mask.byte()

        return new_mask_

    def gradient_growth_tofix(self, name, mask, weight, ori_size=3):
        if mask.shape[-1] <= ori_size:
            keep_size = mask.shape[-1]
        else:
            keep_size = ori_size

        weight_new = copy.deepcopy(weight.data) + 10.
        weight_new = weight_new * mask
        grad = self.get_gradient_for_weights(weight)
        grad = grad * (mask == 0).float()
        weight_add = torch.abs(grad) + weight_new

        new_mask = torch.zeros_like(weight_add, dtype=torch.float32).cuda()
        value, index_ = torch.topk(weight_add, keep_size, dim=-1)
        new_mask.scatter_(-1, index_, 1.0)
        new_mask_ = new_mask.byte()

        return new_mask_


    def magnitude_and_negativity_death(self, mask, weight, name):
        num_remove = math.ceil(self.death_rate*self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]

        # find magnitude threshold
        # remove all weights which absolute value is smaller than threshold
        x, idx = torch.sort(weight[weight > 0.0].data.view(-1))
        k = math.ceil(num_remove/2.0)
        if k >= x.shape[0]:
            k = x.shape[0]

        threshold_magnitude = x[k-1].item()

        # find negativity threshold
        # remove all weights which are smaller than threshold
        x, idx = torch.sort(weight[weight < 0.0].view(-1))
        k = math.ceil(num_remove/2.0)
        if k >= x.shape[0]:
            k = x.shape[0]
        threshold_negativity = x[k-1].item()


        pos_mask = (weight.data > threshold_magnitude) & (weight.data > 0.0)
        neg_mask = (weight.data < threshold_negativity) & (weight.data < 0.0)


        new_mask = pos_mask | neg_mask
        return new_mask

    '''
                    GROWTH
    '''

    def random_unfired_growth(self, name, new_mask, weight):
        total_regrowth = self.num_remove[name]
        n = (new_mask == 0).sum().item()
        if n == 0: return new_mask
        num_nonfired_weights = (self.fired_masks[name]==0).sum().item()

        if total_regrowth <= num_nonfired_weights:
            idx = (self.fired_masks[name].flatten() == 0).nonzero()
            indices = torch.randperm(len(idx))[:total_regrowth]

            # idx = torch.nonzero(self.fired_masks[name].flatten())
            new_mask.data.view(-1)[idx[indices]] = 1.0
        else:
            new_mask[self.fired_masks[name]==0] = 1.0
            n = (new_mask == 0).sum().item()
            expeced_growth_probability = ((total_regrowth-num_nonfired_weights) / n)
            new_weights = torch.rand(new_mask.shape).cuda() < expeced_growth_probability
            new_mask = new_mask.byte() | new_weights
        return new_mask

    def random_growth(self, name, new_mask, weight):
        total_regrowth = self.num_remove[name]
        n = (new_mask==0).sum().item()
        if n == 0: return new_mask
        expeced_growth_probability = (total_regrowth/n)
        new_weights = torch.rand(new_mask.shape).cuda() < expeced_growth_probability
        new_mask_ = new_mask.byte() | new_weights.byte()
        if (new_mask_!=0).sum().item() == 0:
            new_mask_ = new_mask
        return new_mask_

    def momentum_growth(self, name, new_mask, weight):
        total_regrowth = self.num_remove[name]
        grad = self.get_momentum_for_weight(weight)
        grad = grad*(new_mask==0).float()
        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0

        return new_mask

    def gradient_growth(self, name, new_mask, weight):
        total_regrowth = self.num_remove[name]
        grad = self.get_gradient_for_weights(weight)
        grad = grad*(new_mask==0).float()

        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0

        return new_mask



    def momentum_neuron_growth(self, name, new_mask, weight):
        total_regrowth = self.num_remove[name]
        grad = self.get_momentum_for_weight(weight)

        M = torch.abs(grad)
        if len(M.shape) == 2: sum_dim = [1]
        elif len(M.shape) == 4: sum_dim = [1, 2, 3]

        v = M.mean(sum_dim).data
        v /= v.sum()

        slots_per_neuron = (new_mask==0).sum(sum_dim)

        M = M*(new_mask==0).float()
        for i, fraction  in enumerate(v):
            neuron_regrowth = math.floor(fraction.item()*total_regrowth)
            available = slots_per_neuron[i].item()

            y, idx = torch.sort(M[i].flatten())
            if neuron_regrowth > available:
                neuron_regrowth = available
            threshold = y[-(neuron_regrowth)].item()
            if threshold == 0.0: continue
            if neuron_regrowth < 10: continue
            new_mask[i] = new_mask[i] | (M[i] > threshold)

        return new_mask

    '''
                UTILITY
    '''
    def get_momentum_for_weight(self, weight):
        if 'exp_avg' in self.optimizer.state[weight]:
            adam_m1 = self.optimizer.state[weight]['exp_avg']
            adam_m2 = self.optimizer.state[weight]['exp_avg_sq']
            grad = adam_m1/(torch.sqrt(adam_m2) + 1e-08)
        elif 'momentum_buffer' in self.optimizer.state[weight]:
            grad = self.optimizer.state[weight]['momentum_buffer']
        return grad

    def get_gradient_for_weights(self, weight):
        grad = weight.grad.clone()
        return grad

    def print_nonzero_counts(self):
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                if 'max' in name or 'bottleneck' in name or 'shortcut' in name or 'head' in name:
                    continue
                mask = self.masks[name]
                num_nonzeros = (mask != 0).sum().item()

                ## compare pruned mask
                pre_masks_neg = self.pre_masks[name].data < 1.0
                pruned_masks_neg = self.pruned_masks[name].data < 1.0
                comp_1 = self.pre_masks[name].data.byte() & self.pruned_masks[name].data.byte()
                comp_2 = pre_masks_neg.byte() & pruned_masks_neg.byte()
                diff = self.pre_masks[name].numel()-(comp_1.sum().item() + comp_2.sum().item())

                val = '{0}: {1}->{2}, density: {3:.3f}, diff: {4}'.format(name, self.name2nonzeros[name], num_nonzeros, num_nonzeros/float(mask.numel()), diff)
                print(val)


        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                print('Death rate: {0}\n'.format(self.death_rate))
                break

    def fired_masks_update(self):
        ntotal_fired_weights = 0.0
        ntotal_weights = 0.0
        layer_fired_weights = {}
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                self.fired_masks[name] = self.masks[name].data.byte() | self.fired_masks[name].data.byte()
                ntotal_fired_weights += float(self.fired_masks[name].sum().item())
                ntotal_weights += float(self.fired_masks[name].numel())
                layer_fired_weights[name] = float(self.fired_masks[name].sum().item())/float(self.fired_masks[name].numel())
                print('Layerwise percentage of the fired weights of', name, 'is:', layer_fired_weights[name])
        total_fired_weights = ntotal_fired_weights/ntotal_weights
        print('The percentage of the total fired weights is:', total_fired_weights)
        return layer_fired_weights, total_fired_weights

    ### add function
    def death_decay_update(self, decay_flag=True):
        self.decay_flag = decay_flag