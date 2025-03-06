import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    def __init__(self, name=None):
        super().__init__()
        self.name = name

    def forward(self, x):
        return x * torch.sigmoid(x)


class Conv3dSamePadding(nn.Conv3d):
    """
    3D Convolutions with 'same' padding.
    Ensures that the output spatial size (D_out, H_out, W_out) is
    math.ceil(D_in / stride_d), math.ceil(H_in / stride_h), math.ceil(W_in / stride_w).
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
        name=None
    ):
        # We always set padding=0 here; we compute the actual padding in forward()
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        # Ensure stride is a tuple of length 3 (d, h, w)
        if isinstance(self.stride, int):
            self.stride = (self.stride,) * 3
        elif len(self.stride) == 1:
            self.stride = (self.stride[0],) * 3
        elif len(self.stride) == 2:
            self.stride = (self.stride[0], self.stride[1],self.stride[0] )
        # You can further adapt logic if you expect more flexible inputs.

        self.name = name

    def forward(self, x):
        # x is [N, C, D, H, W]
        in_d, in_h, in_w = x.shape[2], x.shape[3], x.shape[4]
        # weight is [out_channels, in_channels, kd, kh, kw]
        kd, kh, kw = self.weight.shape[2], self.weight.shape[3], self.weight.shape[4]
        sd, sh, sw = self.stride
        dd, dh, dw = self.dilation

        # Compute output dimensions (without explicit padding)
        out_d = math.ceil(in_d / sd)
        out_h = math.ceil(in_h / sh)
        out_w = math.ceil(in_w / sw)

        # Total padding along each dimension
        pad_d = max((out_d - 1) * sd + (kd - 1) * dd + 1 - in_d, 0)
        pad_h = max((out_h - 1) * sh + (kh - 1) * dh + 1 - in_h, 0)
        pad_w = max((out_w - 1) * sw + (kw - 1) * dw + 1 - in_w, 0)

        # Pad (pad_w_front, pad_w_back, pad_h_front, pad_h_back, pad_d_front, pad_d_back)
        if any(p > 0 for p in [pad_d, pad_h, pad_w]):
            x = F.pad(
                x,
                [
                    pad_w // 2, pad_w - pad_w // 2,
                    pad_h // 2, pad_h - pad_h // 2,
                    pad_d // 2, pad_d - pad_d // 2
                ]
            )

        return F.conv3d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,     # still zero in the constructor
            dilation=self.dilation,
            groups=self.groups
        )


        
class BatchNorm3d(nn.BatchNorm3d):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        name=None
    ):
        super().__init__(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats
        )
        self.name = name

def drop_connect(inputs, drop_connect_rate, training):
    if not training:
        return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1.0 - drop_connect_rate
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block
    """

    def __init__(self, block_args, global_params, idx):
        super().__init__()

        block_name = 'blocks_' + str(idx) + '_'

        self.block_args = block_args
        self.batch_norm_momentum = 1 - global_params.batch_norm_momentum
        self.batch_norm_epsilon = global_params.batch_norm_epsilon
        self.has_se = (self.block_args.se_ratio is not None) and (0 < self.block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip

        self.swish = Swish(block_name + '_swish')

        # Expansion phase
        in_channels = self.block_args.input_filters
        out_channels = self.block_args.input_filters * self.block_args.expand_ratio
        if self.block_args.expand_ratio != 1:
            self._expand_conv = Conv3dSamePadding(in_channels=in_channels,
                                                  out_channels=out_channels,
                                                  kernel_size=1,
                                                  bias=False,
                                                  name=block_name + 'expansion_conv')
            self._bn0 = BatchNorm3d(num_features=out_channels,
                                    momentum=self.batch_norm_momentum,
                                    eps=self.batch_norm_epsilon,
                                    name=block_name + 'expansion_batch_norm')

        # Depth-wise convolution phase
        kernel_size = self.block_args.kernel_size
        strides = self.block_args.strides
        self._depthwise_conv = Conv3dSamePadding(in_channels=out_channels,
                                                 out_channels=out_channels,
                                                 groups=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=strides,
                                                 bias=False,
                                                 name=block_name + 'depthwise_conv')
        self._bn1 = BatchNorm3d(num_features=out_channels,
                                momentum=self.batch_norm_momentum,
                                eps=self.batch_norm_epsilon,
                                name=block_name + 'depthwise_batch_norm')

        # Squeeze and Excitation layer
        if self.has_se:
            num_squeezed_channels = max(1, int(self.block_args.input_filters * self.block_args.se_ratio))
            self._se_reduce = Conv3dSamePadding(in_channels=out_channels,
                                                out_channels=num_squeezed_channels,
                                                kernel_size=1,
                                                name=block_name + 'se_reduce')
            self._se_expand = Conv3dSamePadding(in_channels=num_squeezed_channels,
                                                out_channels=out_channels,
                                                kernel_size=1,
                                                name=block_name + 'se_expand')

        # Output phase
        final_output_channels = self.block_args.output_filters
        self._project_conv = Conv3dSamePadding(in_channels=out_channels,
                                               out_channels=final_output_channels,
                                               kernel_size=1,
                                               bias=False,
                                               name=block_name + 'output_conv')
        self._bn2 = BatchNorm3d(num_features=final_output_channels,
                                momentum=self.batch_norm_momentum,
                                eps=self.batch_norm_epsilon,
                                name=block_name + 'output_batch_norm')

    def forward(self, x, drop_connect_rate=None):
        identity = x
        # Expansion and depth-wise convolution
        if self.block_args.expand_ratio != 1:
            x = self._expand_conv(x)
            x = self._bn0(x)
            x = self.swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self.swish(x)

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(self.swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self.block_args.input_filters, self.block_args.output_filters
        if self.id_skip and self.block_args.strides == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, drop_connect_rate=drop_connect_rate, training=self.training)
            x = x + identity
        return x



def double_conv_3d(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True)
    )



def up_conv_3d(in_channels, out_channels):
    return nn.ConvTranspose3d(
        in_channels, out_channels, kernel_size=2, stride=2
    )



def custom_head_3d(in_features, out_features):
    return nn.Sequential(
        nn.Dropout(),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(512, out_features)
    )
