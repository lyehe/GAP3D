"""
This is a modified version of the U-Net implementation by Jackson Huang
(https://github.com/jaxony/unet-pytorch/)


This is the original license text:

MIT License

Copyright (c) 2017 Jackson Huang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torch.nn import init


# 3D convolutions with kernel size 3
def conv3x3x3(
    in_channels: int,  # number of input channels
    out_channels: int,  # number of output channels
    stride: int = 1,  # stride of convolution
    padding: int = 1,  # padding
    bias: bool = True,  # use bias
    groups: int = 1,  # number of groups
) -> nn.Conv3d:
    return nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups,
    )


# 3D upconvolution with kernel size 2
def upconv2x2x2(
    in_channels: int,  # number of input channels
    out_channels: int,  # number of output channels
    mode: str = "transpose",  # type of upconvolution
) -> nn.Module:
    if mode == "transpose":
        return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
    else:
        return nn.Sequential(
            nn.Upsample(mode="bilinear", scale_factor=2),
            conv3x3x3(in_channels, out_channels),
        )


# 3D convolution with kernel size 1
def conv1x1x1(
    in_channels: int,  # number of input channels
    out_channels: int,  # number of output channels
    groups: int = 1,  # number of groups
) -> nn.Conv3d:
    return nn.Conv3d(in_channels, out_channels, kernel_size=1, groups=groups, stride=1)


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """

    def __init__(
        self,
        in_channels: int,  # number of input channels
        out_channels: int,  # number of output channels
        pooling: bool = True,  # use pooling
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.conv1 = conv3x3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3x3(self.out_channels, self.out_channels)
        self.conv3 = conv3x3x3(self.out_channels, self.out_channels)
        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(
        self,
        input: torch.Tensor,  # input tensor
        dropout: float = 0.0,  # dropout probability
    ) -> torch.Tensor:
        input_skip = self.conv1(input_skip)
        input = F.relu(self.conv2(input_skip))
        input = F.relu(self.conv3(input) + input_skip)
        if dropout > 0.0:
            input = F.dropout(input, p=dropout)
        output = input
        if self.pooling:
            output = self.pool(output)
        return output, input


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """

    def __init__(
        self,
        in_channels: int,  # number of input channels
        out_channels: int,  # number of output channels
        merge_mode: str = "concat",  # merge mode
        up_mode: str = "transpose",  # type of upconvolution
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2x2(
            self.in_channels, self.out_channels, mode=self.up_mode
        )

        if self.merge_mode == "concat":
            self.conv1 = conv3x3x3(2 * self.out_channels, self.out_channels)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3x3(self.out_channels, self.out_channels)
        self.conv2 = conv3x3x3(self.out_channels, self.out_channels)
        self.conv3 = conv3x3x3(self.out_channels, self.out_channels)

    def forward(
        self,
        from_down: torch.Tensor,  # tensor from the encoder pathway
        from_up: torch.Tensor,  # upconv'd tensor from the decoder pathway
        dropout: float = 0.0,  # dropout probability
    ) -> torch.Tensor:
        from_up = self.upconv(from_up)
        if self.merge_mode == "concat":
            input = torch.cat((from_up, from_down), 1)
        else:
            input = from_up + from_down
        input_skip = self.conv1(input)
        input = F.relu(self.conv2(input_skip))
        input = F.relu(self.conv3(input) + input_skip)
        if dropout > 0.0:
            input = F.dropout(input, p=dropout)
        return input


class UN(pl.LightningModule):
    """`UNet` class is based on https://arxiv.org/abs/1505.04597
    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).
    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')

    Update:
    This is updated to 3D convolution from 2D
    """

    def __init__(
        self,
        levels: int = 4,  # number of levels in the u-net
        channels: int = 1,  # number of input channels
        depth: int = 5,  # number of MaxPools in the U-Net
        start_filts: int = 64,  # number of convolutional filters for the first conv
        up_mode: str = "transpose",  # upconvolution type 'transpose' | 'upsample'
        merge_mode: str = "add",  # merge mode "concat" | "add
    ):
        self.save_hyperparameters()
        super().__init__()

        assert up_mode in (
            "transpose",
            "upsample",
        ), "`up_mode` must be transpose or upsample" "but got {}".format(up_mode)
        self.up_mode = up_mode

        assert merge_mode in (
            "concat",
            "add",
        ), "`merge_mode` must be concat or add" "but got {}".format(merge_mode)
        self.merge_mode = merge_mode

        assert not (
            self.up_mode == "upsample" and self.merge_mode == "add"
        ), "upsample is incompatible with add to decrease the number of channels"

        self.levels = levels
        self.channels = channels
        self.start_filts = start_filts
        self.depth = depth

        self.down_convs = []
        self.up_convs = []

        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.channels * self.levels if i == 0 else outs
            outs = self.start_filts * (2**i)
            pooling = True if i < depth - 1 else False
            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode, merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1x1(outs, self.channels)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)
        self.reset_params()

    @staticmethod
    def weight_init(m: nn.Conv2d):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(
        self,
        input: torch.Tensor,  # input tensor
        factor: float = 10.0,  # factor for scaling the input
    ) -> torch.Tensor:
        output = None
        for i in range(self.levels):
            scale = input.clone() * (factor ** (-i))
            scale = torch.sin(scale)
            if output is None:
                output = scale
            else:
                output = torch.cat((output, scale), 1)

        encoder_outs = []
        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            output, before_pool = module(output)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            output = module(before_pool, output)

        output = self.conv_final(output)
        return output

    def configure_optimizers(
        self,
        lr: float = 1e-4,  # learning rate
        mode: str = "min",  # mode for ReduceLROnPlateau
        factor: float = 0.5,  # factor for ReduceLROnPlateau
        patience: int = 10,  # patience for ReduceLROnPlateau
    ) -> dict:
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    # Photon Loss
    def photonLoss(
        self,
        result: torch.Tensor,  # result tensor
        target: torch.Tensor,  # target tensor
    ) -> torch.Tensor:
        expEnergy = torch.exp(result)
        perImage = -torch.mean(result * target, dim=(-1, -2, -3, -4), keepdims=True)
        perImage += torch.log(
            torch.mean(expEnergy, dim=(-1, -2, -3, -4), keepdims=True)
        ) * torch.mean(target, dim=(-1, -2, -3, -4), keepdims=True)
        return torch.mean(perImage)

    # Mean Square Error Loss
    def MSELoss(
        self,
        result: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        expEnergy = torch.exp(result)
        expEnergy /= torch.mean(expEnergy, dim=(-1, -2, -3, -4), keepdims=True)
        target = target / (torch.mean(target, dim=(-1, -2, -3, -4), keepdims=True))
        return torch.mean((expEnergy - target) ** 2)

    # Training step
    def training_step(
        self,
        batch: torch.Tensor,  # batch of training data
        batch_idx: int = None,  # index of batch
    ) -> torch.Tensor:
        '''
        This is the training step for the model.
        The model iterates over the batch and applies the photonLoss to 
        the output of the model and the target
        '''
        if batch_idx is None:
            loss = self.photonLoss(
                self(batch[:, self.channels :, ...]),  # apply to output of all batch
                batch[:, : self.channels, ...],
            )
        else:
            loss = self.photonLoss(
                self(batch[batch_idx, self.channels :, ...]),
                batch[batch_idx, : self.channels, ...],
            )
        self.log("train_loss", loss)
        return loss

    def validation_step(
        self,
        batch: torch.Tensor,  # batch of validation data
        batch_idx: int = None,  # index of batch
    ) -> None:
        if batch_idx is None:
            loss = self.photonLoss(
                self(batch[:, self.channels :, ...]),  # apply to all batch
                batch[:, : self.channels, ...],
            )
        else:
            loss = self.photonLoss(
                self(batch[batch_idx, self.channels :, ...]),
                batch[batch_idx, : self.channels, ...],
            )
        self.log("val_loss", loss)

    def test_step(
        self,
        batch: torch.Tensor,  # batch of test data
        batch_idx: int = None,  # index of batch
    ) -> None:
        if batch_idx is None:
            loss = self.photonLoss(
                self(batch[:, self.channels :, ...]),  # apply to all batch
                batch[:, : self.channels, ...],
            )
        else:
            loss = self.photonLoss(
                self(batch[batch_idx, self.channels :, ...]),
                batch[batch_idx, : self.channels, ...],
            )
        self.log("test_loss", loss)
