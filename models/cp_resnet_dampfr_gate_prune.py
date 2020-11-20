# coding: utf-8

# this source is useful for loading pre-trained models that was submitted to the challenge






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


class PruneGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, training=True):
        ctx.save_for_backward(input)
        # if training:
        #     return  1.0/(1.0 + torch.exp(-input))
        out = torch.ones_like(input)
        out[input < 0] = 0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output * torch.exp(-input) / ((1.0 + torch.exp(-input)).pow(2))
        return grad_input


class Conv2dPrune(_ConvNd):
    r"""Applies a 2D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\text{in}}, H, W)` and output :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`
    can be precisely described as:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)


    where :math:`\star` is the valid 2D `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`H` is a height of input planes in pixels, and :math:`W` is
    width in pixels.

    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a tuple.

    * :attr:`padding` controls the amount of implicit zero-paddings on both
      sides for :attr:`padding` number of points for each dimension.

    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters, of size:
          :math:`\left\lfloor\frac{out\_channels}{in\_channels}\right\rfloor`.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    .. note::

         Depending of the size of your kernel, several (of the last)
         columns of the input might be lost, because it is a valid `cross-correlation`_,
         and not a full `cross-correlation`_.
         It is up to the user to add proper padding.

    .. note::

        When `groups == in_channels` and `out_channels == K * in_channels`,
        where `K` is a positive integer, this operation is also termed in
        literature as depthwise convolution.

        In other words, for an input of size :math:`(N, C_{in}, H_{in}, W_{in})`,
        a depthwise convolution with a depthwise multiplier `K`, can be constructed by arguments
        :math:`(in\_channels=C_{in}, out\_channels=C_{in} \times K, ..., groups=C_{in})`.

    .. include:: cudnn_deterministic.rst

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        padding_mode (string, optional). Accepted values `zeros` and `circular` Default: `zeros`
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},`
                         :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]})`.
                         The values of these weights are sampled from
                         :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{1}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape (out_channels). If :attr:`bias` is ``True``,
                         then the values of these weights are
                         sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{1}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.Conv2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        try:
            super(Conv2dPrune, self).__init__(
                in_channels, out_channels, kernel_size, stride, padding, dilation,
                False, _pair(0), groups, bias)
        except:
            super(Conv2dPrune, self).__init__(
                in_channels, out_channels, kernel_size, stride, padding, dilation,
                False, _pair(0), groups, bias, padding_mode='zeros')

        self.register_parameter('prune_mask', torch.nn.parameter.Parameter(torch.ones_like(self.weight)))

    def update_prune_by_threshold(self, threshold):
        if threshold < 0:
            print("\n\n\n\n threshold<0 : ", threshold, "\n\n\n\n")
        with torch.no_grad():
            mask = self.prune_mask < threshold
            self.prune_mask[mask] = -1
            self.prune_mask[~mask] = 1

    def update_prune_by_percentage(self, pecentage):
        k = 1 + round(float(pecentage) * (self.weight.numel() - 1))
        threshold = self.prune_mask.view(-1).kthvalue(k).values.item()
        self.update_prune_by_threshold(threshold)

    def copy_prune_to_weight(self):
        with torch.no_grad():
            self.weight.mul_(PruneGate.apply(self.prune_mask))

    def conv2d_forward(self, input, weight):

        # print(damper)
        if apply_prune:
            return F.conv2d(input, weight *
                            PruneGate.apply(self.prune_mask), self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(input, weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

    def forward(self, input):
        return self.conv2d_forward(input, self.weight)



class Conv2dDamped(_ConvNd):
    r"""Applies a 2D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\text{in}}, H, W)` and output :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`
    can be precisely described as:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)


    where :math:`\star` is the valid 2D `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`H` is a height of input planes in pixels, and :math:`W` is
    width in pixels.

    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a tuple.

    * :attr:`padding` controls the amount of implicit zero-paddings on both
      sides for :attr:`padding` number of points for each dimension.

    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters, of size:
          :math:`\left\lfloor\frac{out\_channels}{in\_channels}\right\rfloor`.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    .. note::

         Depending of the size of your kernel, several (of the last)
         columns of the input might be lost, because it is a valid `cross-correlation`_,
         and not a full `cross-correlation`_.
         It is up to the user to add proper padding.

    .. note::

        When `groups == in_channels` and `out_channels == K * in_channels`,
        where `K` is a positive integer, this operation is also termed in
        literature as depthwise convolution.

        In other words, for an input of size :math:`(N, C_{in}, H_{in}, W_{in})`,
        a depthwise convolution with a depthwise multiplier `K`, can be constructed by arguments
        :math:`(in\_channels=C_{in}, out\_channels=C_{in} \times K, ..., groups=C_{in})`.

    .. include:: cudnn_deterministic.rst

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        padding_mode (string, optional). Accepted values `zeros` and `circular` Default: `zeros`
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},`
                         :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]})`.
                         The values of these weights are sampled from
                         :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{1}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape (out_channels). If :attr:`bias` is ``True``,
                         then the values of these weights are
                         sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{1}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.Conv2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        try:
            super(Conv2dDamped, self).__init__(
                in_channels, out_channels, kernel_size, stride, padding, dilation,
                False, _pair(0), groups, bias)
        except:
            super(Conv2dDamped, self).__init__(
                in_channels, out_channels, kernel_size, stride, padding, dilation,
                False, _pair(0), groups, bias, padding_mode='zeros')

    def conv2d_forward(self, input, weight):

        damper=get_damper(weight)
        #print(damper)
        return F.conv2d(input, weight*damper, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        return self.conv2d_forward(input, self.weight)

cach_damp={}
def get_damper(w):
    k=(w.shape[2],w.shape[3])
    if cach_damp.get(k) is None:
        t=torch.ones_like(w[0,0]).reshape(1,1,w.shape[2],w.shape[3])
        center2=(w.shape[2]-1.)/2
        center3 = (w.shape[3]-1.) / 2
        minscale=0.1
        if center2 >=1 :
            for i in range(w.shape[2]):
                distance = np.abs(i - center2)
                sacale = -(1 - minscale) * distance/center2 + 1.
                t[:, :, i, :] *= sacale
        # if center2 >= 1:
        #     for i in range(w.shape[3]):
        #         distance = np.abs(i - center3)
        #         sacale = -(1 - minscale) * distance/center3 + 1.
        #         t[:, :, :, i] *= sacale
        cach_damp[k]=t.detach()
    return cach_damp.get(k)

def initialize_weights(module):
    if isinstance(module, Conv2dPrune):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity="relu")
    elif isinstance(module, Conv2dDamped):
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

apply_prune = False

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
        total_params += module.weight.numel()
        total_pruned_params += (module.prune_mask < 0).sum().item()
    elif isinstance(module, nn.Conv2d):
        total_params += module.weight.numel()
    elif isinstance(module, Conv2dDamped):
        total_params += module.weight.numel()

total_prunable_params = 0


def apply_sum_prunable_params(module):
    global total_prunable_params
    if isinstance(module, Conv2dPrune):
        total_prunable_params += module.prune_mask.numel()


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
        ConvClass = Conv2dDamped
        if k1 == 1:
            ConvClass = Conv2dPrune
        self.conv1 = ConvClass(
            in_channels,
            out_channels,
            kernel_size=k1,
            stride=stride,  # downsample with first conv
            padding=calc_padding(k1),
            bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # to do damp non-pruned
        ConvClass = Conv2dDamped
        if k2 == 1:
            ConvClass = Conv2dPrune
        self.conv2 = ConvClass(
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

        self.in_c = nn.Sequential(nn.Conv2d(
            input_shape[1],
            n_channels[0],
            kernel_size=5,
            stride=2,
            padding=1,
            bias=False),
            nn.BatchNorm2d(n_channels[0]),
            nn.ReLU(True)
        )
        print("\nWarning Pruning only 1x1 CONVS\n")
        self.stage1 = self._make_stage(
            n_channels[0], n_channels[0], n_blocks_per_stage[0], block, stride=1, maxpool=config['stage1']['maxpool'],
            k1s=config['stage1']['k1s'], k2s=config['stage1']['k2s'])
        if n_blocks_per_stage[1] == 0:
            self.stage2 = nn.Sequential()
            n_channels[1] = n_channels[0]
            print("WARNING: stage2 removed")
        else:
            self.stage2 = self._make_stage(
                n_channels[0], n_channels[1], n_blocks_per_stage[1], block, stride=1, maxpool=config['stage2']['maxpool'],
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
            raise NotImplementedError()
        else:
            ff_list += [nn.Conv2d(
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
            if isinstance(self.feed_forward[0], nn.Conv2d):
                self.feed_forward[0].weight.data.zero_()
            self.apply(initialize_weights_fixup)
        else:
            self.apply(initialize_weights)
        self.use_check_point = config.get("use_check_point") or False

    def _make_stage(self, in_channels, out_channels, n_blocks, block, stride, maxpool=set(), k1s=[3, 3, 3, 3, 3, 3],
                    k2s=[3, 3, 3, 3, 3, 3],):
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

    def get_num_prunable_params(self):
        global total_prunable_params
        total_prunable_params = 0
        self.apply(apply_sum_prunable_params)
        return total_prunable_params

    def get_num_true_params(self):
        total_true_params = 0

        def apply_sum_true_params(module):
            nonlocal total_true_params
            if isinstance(module, Conv2dPrune):
                total_true_params += module.weight.numel()
            elif isinstance(module, nn.Conv2d):
                total_true_params += module.weight.numel()
            elif isinstance(module, Conv2dDamped):
                total_true_params += module.weight.numel()
        self.apply(apply_sum_true_params)
        print("total_true_params (old): ",total_true_params)
        total_gate_params=0
        def apply_sum_gate_params(module):
            nonlocal total_gate_params
            if isinstance(module, Conv2dPrune):
                total_gate_params += module.prune_mask.numel()

        self.apply(apply_sum_gate_params)
        print("total gate params: ", total_gate_params)
        sum_params = 0
        for p in self.parameters():
            if p.requires_grad:
                sum_params += p.numel()
        print("sum_params: ", sum_params,", new total=",sum_params-total_gate_params)
        return sum_params-total_gate_params

    def update_prune_weights(self, percentage, mode):
        global prune_percentage, prune_threshold, total_params, total_pruned_params
        total_params = 0
        total_pruned_params = 0
        self.apply(apply_sum_prunes)
        print("\nBefore: ", mode, ":  params ", total_params, " pruned ", total_pruned_params, " remaining ",
              total_params - total_pruned_params)
        if mode == "layer":
            prune_percentage = percentage
            self.apply(apply_prune_percentage)
        elif mode == "all":
            all_params = []

            def gather_masks(module):
                if isinstance(module, Conv2dPrune):
                    all_params.append(module.prune_mask.view(-1).cpu())

            self.apply(gather_masks)
            all_params = torch.cat(all_params).view(-1)
            k = 1 + round(float(percentage) * (all_params.numel() - 1))
            threshold = all_params.kthvalue(k).values.item()
            if threshold < 0:
                print("\n\n\n\n threshold<0 : ", threshold, "\n\n\n\n")
            prune_threshold = threshold
            self.apply(apply_prune_per_threshold)
        else:
            raise RuntimeError("not implemented mode")
        total_params = 0
        total_pruned_params = 0
        self.apply(apply_sum_prunes)
        print(mode, " (after)  params ", total_params, " pruned ", total_pruned_params, " remaining ",
              total_params - total_pruned_params, "\n")
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
