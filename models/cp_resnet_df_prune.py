# coding: utf-8
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._six import container_abcs
from torch.nn.modules.conv import _ConvNd
from torch.utils.checkpoint import checkpoint_sequential
import numpy as np
import shared_globals
from librosa.filters import mel as librosa_mel_fn

from itertools import repeat
from utils_funcs import update_dict


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)

apply_prune = False


class Conv2dPrune(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        print("damped prune ", kernel_size)
        try:
            super(Conv2dPrune, self).__init__(
                in_channels, out_channels, kernel_size, stride, padding, dilation,
                False, _pair(0), groups, bias)
        except:
            super(Conv2dPrune, self).__init__(
                in_channels, out_channels, kernel_size, stride, padding, dilation,
                False, _pair(0), groups, bias, padding_mode='zeros')
        prune_mask = torch.ones_like(self.weight)
        prune_mask.requires_grad = False
        self.register_buffer('prune_mask', prune_mask)

    def update_prune_by_threshold(self, threshold):
        self.prune_mask.fill_(1.)
        self.prune_mask[self.weight.abs() < threshold] = 0

    def update_prune_by_percentage(self, pecentage):
        k = 1 + round(float(pecentage) * (self.weight.numel() - 1))
        threshold = self.weight.view(-1).abs().kthvalue(k).values.item()
        self.update_prune_by_threshold(threshold)

    def copy_prune_to_weight(self):
        self.weight.mul_(self.prune_mask)

    def conv2d_forward(self, input, weight):
        damper = get_damper(weight)
        # print(damper)
        if apply_prune:
            return F.conv2d(input, weight * self.prune_mask * damper, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(input, weight * damper, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

    def forward(self, input):
        return self.conv2d_forward(input, self.weight)


cach_damp = {}


def get_damper(w):
    k = (w.shape[2], w.shape[3])
    if cach_damp.get(k) is None:
        t = torch.ones_like(w[0, 0]).reshape(1, 1, w.shape[2], w.shape[3])
        center2 = (w.shape[2] - 1.) / 2
        center3 = (w.shape[3] - 1.) / 2
        minscale = 0.1
        if center2 >= 1:
            for i in range(w.shape[2]):
                distance = np.abs(i - center2)
                sacale = -(1 - minscale) * distance / center2 + 1.
                t[:, :, i, :] *= sacale
        # if center2 >= 1:
        #     for i in range(w.shape[3]):
        #         distance = np.abs(i - center3)
        #         sacale = -(1 - minscale) * distance/center3 + 1.
        #         t[:, :, :, i] *= sacale
        cach_damp[k] = t.detach()
    return cach_damp.get(k)


def initialize_weights(module):
    if isinstance(module, Conv2dPrune):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity="relu")

        # nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()


layer_index_total = 0


def initialize_weights_fixup(module):
    # if isinstance(module, AttentionAvg):
    #     print("AttentionAvg init..")
    #     module.forw_conv[0].weight.data.zero_()
    #     module.atten[0].bias.data.zero_()
    #     nn.init.kaiming_normal_(module.atten[0].weight.data, mode='fan_in', nonlinearity="sigmoid")
    if isinstance(module, BasicBlock):
        # He init, rescaled by Fixup multiplier
        b = module
        n = b.conv1.kernel_size[0] * b.conv1.kernel_size[1] * b.conv1.out_channels
        print(b.layer_index, math.sqrt(2. / n), layer_index_total ** (-0.5))
        b.conv1.weight.data.normal_(0, (layer_index_total ** (-0.5)) * math.sqrt(2. / n))
        b.conv2.weight.data.zero_()
        if b.shortcut._modules.get('conv') is not None:
            convShortcut = b.shortcut._modules.get('conv')
            n = convShortcut.kernel_size[0] * convShortcut.kernel_size[1] * convShortcut.out_channels
            convShortcut.weight.data.normal_(0, math.sqrt(2. / n))
    if isinstance(module, Conv2dPrune):
        pass
        # nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity="relu")
        # nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()


first_RUN = True

prune_threshold = 100


def apply_prune_per_threshold(module):
    if isinstance(module, Conv2dPrune):
        module.update_prune_by_threshold(prune_threshold)


prune_percentage = 0.8


def apply_prune_percentage(module):
    if isinstance(module, Conv2dPrune):
        module.update_prune_by_percentage(prune_percentage)


total_params = 0
total_pruned_params = 0


def apply_sum_prunes(module):
    global total_params, total_pruned_params
    if isinstance(module, Conv2dPrune):
        total_params += module.prune_mask.numel()
        total_pruned_params += (module.prune_mask == 0).sum().item()


def calc_padding(kernal):
    try:
        return kernal // 3
    except TypeError:
        return [k // 3 for k in kernal]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, k1=3, k2=3):
        super(BasicBlock, self).__init__()
        global layer_index_total
        self.layer_index = layer_index_total
        layer_index_total = layer_index_total + 1
        self.conv1 = Conv2dPrune(
            in_channels,
            out_channels,
            kernel_size=k1,
            stride=stride,  # downsample with first conv
            padding=calc_padding(k1),
            bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = Conv2dPrune(
            out_channels,
            out_channels,
            kernel_size=k2,
            stride=1,
            padding=calc_padding(k2),
            bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module(
                'conv',
                Conv2dPrune(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,  # downsample
                    padding=0,
                    bias=False))
            self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))  # BN

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        y += self.shortcut(x)
        y = F.relu(y, inplace=True)  # apply ReLU after addition
        return y




class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()
        print("Init frequency-Damped pruned CP-ResNet...")
        input_shape = config['input_shape']
        n_classes = config['n_classes']

        base_channels = config['base_channels']
        block_type = config['block_type']
        depth = config['depth']
        self.pooling_padding = config.get("pooling_padding", 0) or 0
        self.use_raw_spectograms = config.get("use_raw_spectograms") or False
        self.apply_softmax = config.get("apply_softmax") or False

        assert block_type in ['basic', 'bottleneck']
        if self.use_raw_spectograms:
            mel_basis = librosa_mel_fn(
                22050, 2048, 256)
            mel_basis = torch.from_numpy(mel_basis).float()
            self.register_buffer('mel_basis', mel_basis)
        if block_type == 'basic':
            block = BasicBlock
            n_blocks_per_stage = (depth - 2) // 6
            assert n_blocks_per_stage * 6 + 2 == depth
        else:
            block = BottleneckBlock
            n_blocks_per_stage = (depth - 2) // 9
            assert n_blocks_per_stage * 9 + 2 == depth
        n_blocks_per_stage = [n_blocks_per_stage, n_blocks_per_stage, n_blocks_per_stage]

        if config.get("n_blocks_per_stage") is not None:
            shared_globals.console.warning(
                "n_blocks_per_stage is specified ignoring the depth param, nc=" + str(config.get("n_channels")))
            n_blocks_per_stage = config.get("n_blocks_per_stage")

        n_channels = config.get("n_channels")
        if n_channels is None:
            n_channels = [
                base_channels,
                base_channels * 2 * block.expansion,
                base_channels * 4 * block.expansion
            ]
        if config.get("grow_a_lot"):
            n_channels[2] = base_channels * 8 * block.expansion

        self.in_c = nn.Sequential(Conv2dPrune(
            input_shape[1],
            n_channels[0],
            kernel_size=5,
            stride=2,
            padding=1,
            bias=False),
            nn.BatchNorm2d(n_channels[0]),
            nn.ReLU(True)
        )
        self.stage1 = self._make_stage(
            n_channels[0], n_channels[0], n_blocks_per_stage[0], block, stride=1, maxpool=config['stage1']['maxpool'],
            k1s=config['stage1']['k1s'], k2s=config['stage1']['k2s'])
        if n_blocks_per_stage[1] == 0:
            self.stage2 = nn.Sequential()
            n_channels[1] = n_channels[0]
            print("WARNING: stage2 removed")
        else:
            self.stage2 = self._make_stage(
                n_channels[0], n_channels[1], n_blocks_per_stage[1], block, stride=1,
                maxpool=config['stage2']['maxpool'],
                k1s=config['stage2']['k1s'], k2s=config['stage2']['k2s'])
        if n_blocks_per_stage[2] == 0:
            self.stage3 = nn.Sequential()
            n_channels[2] = n_channels[1]
            print("WARNING: stage3 removed")
        else:
            self.stage3 = self._make_stage(
                n_channels[1], n_channels[2], n_blocks_per_stage[2], block, stride=1,
                maxpool=config['stage3']['maxpool'],
                k1s=config['stage3']['k1s'], k2s=config['stage3']['k2s'])

        ff_list = []
        if config.get("attention_avg"):
            if config.get("attention_avg") == "sum_all":
                ff_list.append(AttentionAvg(n_channels[2], n_classes, sum_all=True))
            else:
                ff_list.append(AttentionAvg(n_channels[2], n_classes, sum_all=False))
        else:
            ff_list += [Conv2dPrune(
                n_channels[2],
                n_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False),
                nn.BatchNorm2d(n_classes),
            ]

        self.stop_before_global_avg_pooling = False
        if config.get("stop_before_global_avg_pooling"):
            self.stop_before_global_avg_pooling = True
        else:
            ff_list.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.feed_forward = nn.Sequential(
            *ff_list
        )
        # # compute conv feature size
        # with torch.no_grad():
        #     self.feature_size = self._forward_conv(
        #         torch.zeros(*input_shape)).view(-1).shape[0]
        #
        # self.fc = nn.Linear(self.feature_size, n_classes)

        # initialize weights
        if config.get("weight_init") == "fixup":
            self.apply(initialize_weights)
            if isinstance(self.feed_forward[0], Conv2dPrune):
                self.feed_forward[0].weight.data.zero_()
            self.apply(initialize_weights_fixup)
        else:
            self.apply(initialize_weights)
        self.use_check_point = config.get("use_check_point") or False

    def _make_stage(self, in_channels, out_channels, n_blocks, block, stride, maxpool=set(), k1s=[3, 3, 3, 3, 3, 3],
                    k2s=[3, 3, 3, 3, 3, 3]):
        stage = nn.Sequential()
        if 0 in maxpool:
            stage.add_module("maxpool{}_{}".format(0, 0)
                             , nn.MaxPool2d(2, 2, padding=self.pooling_padding))
        for index in range(n_blocks):
            stage.add_module('block{}'.format(index + 1),
                             block(in_channels,
                                   out_channels,
                                   stride=stride, k1=k1s[index], k2=k2s[index]))

            in_channels = out_channels
            stride = 1
            # if index + 1 in maxpool:
            for m_i, mp_pos in enumerate(maxpool):
                if index + 1 == mp_pos:
                    stage.add_module("maxpool{}_{}".format(index + 1, m_i)
                                     , nn.MaxPool2d(2, 2, padding=self.pooling_padding))
        return stage

    def update_prune_weights(self, percentage, mode):
        global prune_percentage, prune_threshold, total_params, total_pruned_params
        if mode == "layer":
            prune_percentage = percentage
            self.apply(apply_prune_percentage)
        elif mode == "all":
            all_params = []
            for p in self.parameters():
                if len(p.size()) != 1:
                    all_params.append(p.abs().view(-1).cpu())
            all_params = torch.cat(all_params).view(-1)
            k = 1 + round(float(percentage) * (all_params.numel() - 1))
            threshold = all_params.kthvalue(k).values.item()
            prune_threshold = threshold
            self.apply(apply_prune_per_threshold)
        else:
            raise RuntimeError("not implemented mode")
        total_params = 0
        total_pruned_params = 0
        self.apply(apply_sum_prunes)
        print(mode, ":  params ", total_params, " pruned ", total_pruned_params, " remaining ",
              total_params - total_pruned_params)
        return total_params - total_pruned_params

    def set_prune_flag(self, flag):
        global apply_prune
        apply_prune = flag

    def _forward_conv(self, x):
        global first_RUN

        if first_RUN: print("x:", x.size())
        x = self.in_c(x)
        if first_RUN: print("in_c:", x.size())

        if self.use_check_point:
            if first_RUN: print("use_check_point:", x.size())
            return checkpoint_sequential([self.stage1, self.stage2, self.stage3], 3,
                                         (x))
        x = self.stage1(x)

        if first_RUN: print("stage1:", x.size())
        x = self.stage2(x)
        if first_RUN: print("stage2:", x.size())
        x = self.stage3(x)
        if first_RUN: print("stage3:", x.size())
        return x

    def get_num_true_params(self):
        sum_params = 0
        sum_non_zero = 0
        desc = ""

        def calc_params(model):
            nonlocal desc, sum_params, sum_non_zero
            skip = "count"
            # if "batchnorm" in type(model).__name__.lower():
            #     skip = "skip"
            for k, p in model.named_parameters(recurse=False):
                if p.requires_grad:
                    nonzero = p[p != 0].numel()
                    total = p.numel()
                    desc += f"type {type(model).__name__}, {k},  {total}, {nonzero}, {p.dtype}, {skip} " + "\n"
                    if skip != "skip":
                        sum_params += total
                        sum_non_zero += nonzero

        self.apply(calc_params)
        return sum_params

    def get_num_prunable_params(self):
        sum_params = 0
        sum_non_zero = 0
        desc = ""
        print("get_num_prunable_params")

        def calc_params(model):
            nonlocal desc, sum_params, sum_non_zero

            if "Conv2dPrune".lower() in type(model).__name__.lower():
                sum_params += model.weight.numel()

        self.apply(calc_params)
        return sum_params

    def forward(self, x):
        global first_RUN
        if self.use_raw_spectograms:
            if first_RUN: print("raw_x:", x.size())
            x = torch.log10(torch.sqrt((x * x).sum(dim=3)))
            if first_RUN: print("log10_x:", x.size())
            x = torch.matmul(self.mel_basis, x)
            if first_RUN: print("mel_basis_x:", x.size())
            x = x.unsqueeze(1)
        x = self._forward_conv(x)
        x = self.feed_forward(x)
        if first_RUN: print("feed_forward:", x.size())
        if self.stop_before_global_avg_pooling:
            first_RUN = False
            return x
        logit = x.squeeze(2).squeeze(2)

        if first_RUN: print("logit:", logit.size())
        if self.apply_softmax:
            logit = torch.softmax(logit, 1)
        first_RUN = False
        return logit




def get_model_based_on_rho(rho, arch, config_only=False, model_config_overrides={}):
    # extra receptive checking
    extra_kernal_rf = rho - 7
    model_config = {
        "arch": arch,
        "base_channels": 128,
        "block_type": "basic",
        "depth": 26,
        "input_shape": [
            10,
            2,
            -1,
            -1
        ],
        "multi_label": False,
        "n_classes": 10,
        "prediction_threshold": 0.4,
        "stage1": {"maxpool": [1, 2, 4],
                   "k1s": [3,
                           3 - (-extra_kernal_rf > 6) * 2,
                           3 - (-extra_kernal_rf > 4) * 2,
                           3 - (-extra_kernal_rf > 2) * 2],
                   "k2s": [1,
                           3 - (-extra_kernal_rf > 5) * 2,
                           3 - (-extra_kernal_rf > 3) * 2,
                           3 - (-extra_kernal_rf > 1) * 2]},

        "stage2": {"maxpool": [], "k1s": [3 - (-extra_kernal_rf > 0) * 2,
                                          1 + (extra_kernal_rf > 1) * 2,
                                          1 + (extra_kernal_rf > 3) * 2,
                                          1 + (extra_kernal_rf > 5) * 2],
                   "k2s": [1 + (extra_kernal_rf > 0) * 2,
                           1 + (extra_kernal_rf > 2) * 2,
                           1 + (extra_kernal_rf > 4) * 2,
                           1 + (extra_kernal_rf > 6) * 2]},
        "stage3": {"maxpool": [],
                   "k1s": [1 + (extra_kernal_rf > 7) * 2,
                           1 + (extra_kernal_rf > 9) * 2,
                           1 + (extra_kernal_rf > 11) * 2,
                           1 + (extra_kernal_rf > 13) * 2],
                   "k2s": [1 + (extra_kernal_rf > 8) * 2,
                           1 + (extra_kernal_rf > 10) * 2,
                           1 + (extra_kernal_rf > 12) * 2,
                           1 + (extra_kernal_rf > 14) * 2]},
        "block_type": "basic",
        "use_bn": True,
        "weight_init": "fixup"
    }
    # override model_config 
    model_config=update_dict(model_config, model_config_overrides)
    if config_only:
        return model_config
    return Network(model_config)

