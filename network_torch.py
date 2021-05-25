import torch.nn as nn
from torch_lib import *
import torch.nn.functional as F


class dcs_si(nn.Module):
    def __init__(self, kernel_width, c, amino_dim, batch_size, stage_depth, dropout):
        super(dcs_si, self).__init__()
        self.amino_dim = amino_dim
        self.batch_size = batch_size
        self.c = c
        self.kernel_width = kernel_width
        self.stage_depth = stage_depth
        self.block_shape = [self.kernel_width, 1, self.c, self.c * 2]
        self.dropout = dropout

        # Norm and Trans
        self.conv_0 = Conv([3, self.amino_dim, 1, self.c * 2])

        # stage_1
        self.stage_1 = self.make_layer(stage = 'stage_1')

        # stage_2
        self.stage_2 = self.make_layer(stage = 'stage_2')

        # Decoder
        self.conv_1 = Conv([1, 1, self.c, self.c])
        self.drop_out = nn.Dropout(p = self.dropout)
        self.conv_2 = Conv([1, 1, self.c, 2])


    def make_layer(self, stage = None):
        layers = []
        if stage == 'stage_1':
            for i in range(self.stage_depth):
                layers.append(ResidualBlock(self.batch_size, self.kernel_width, self.c))
            layers.append(PlainBlock(self.batch_size, self.kernel_width, self.c))
            return nn.Sequential(*layers)

        elif stage == 'stage_2':
            for i in range(self.stage_depth):
                layers.append(ResidualBlock(self.batch_size, self.kernel_width, self.c))
            layers.append(NormBlock(self.batch_size, self.c))
            return nn.Sequential(*layers)

        else:
            print('please check stage!')


    def forward(self, x):
        x = x.reshape([self.batch_size, 1, -1, self.amino_dim])
        conv_x = self.conv_0(x)
        stage_1_out = self.stage_1(conv_x)
        stage_2_out = self.stage_2(stage_1_out)
        fc0 = self.conv_1(stage_2_out)
        fc0_relu = F.relu(fc0)
        fc0_dropout = self.drop_out(fc0_relu)
        fc1 = self.conv_2(fc0_dropout)
        fc1_relu = F.relu(fc1)
        return fc1_relu

