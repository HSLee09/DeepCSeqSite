import torch
import torch.nn as nn


def Conv(filter_size, bias=False):
    pad = filter_size[0] // 2
    if bias == False:
        conv = nn.Conv2d(in_channels=filter_size[2], out_channels=filter_size[3],
                         kernel_size=(filter_size[0], filter_size[1]), padding=(pad, 0), bias=False
    else:
        conv = nn.Conv2d(in_channels=filter_size[2], out_channels=filter_size[3],
                         kernel_size=(filter_size[0], filter_size[1]), padding=(pad, 0), bias=True)

    return conv  # conv_output


def GLU(input_tensor):
    tensor_a, tensor_b = torch.split(input_tensor, input_tensor.shape[1] // 2, dim=1)
    return tensor_a * torch.sigmoid(tensor_b)


class ResidualBlock(nn.Module):
    def __init__(self, batch_size, kernel_width, c):
        super(ResidualBlock, self).__init__()
        self.kernel_width = kernel_width
        self.c = c
        self.batch_size = batch_size
        self.LayerNorm = nn.LayerNorm(self.c * 2, eps = 0.001)
        self.conv = Conv([self.kernel_width, 1, self.c, self.c * 2])
        
    def forward(self, x):
        l_input = x.reshape([-1, self.c * 2])
        l_out = self.LayerNorm(l_input)
        glu_in = l_out.reshape([self.batch_size, self.c * 2, -1, 1])
        glu_out = GLU(glu_in)
        conv_out = self.conv(glu_out)
        block_out = x + conv_out
        return block_out

class PlainBlock(nn.Module):
    def __init__(self, batch_size, kernel_width, c):
        super(PlainBlock, self).__init__()
        self.kernel_width = kernel_width
        self.c = c
        self.batch_size = batch_size
        self.LayerNorm = nn.LayerNorm(self.c * 2, eps = 0.001)
        self.conv = Conv([self.kernel_width, 1, self.c, self.c * 2])
        
    def forward(self, x):
        l_input = x.reshape([-1, self.c * 2])
        l_out = self.LayerNorm(l_input)
        glu_in = l_out.reshape([self.batch_size, self.c * 2, -1, 1])
        glu_out = GLU(glu_in)
        conv_out = self.conv(glu_out)
        return conv_out


class NormBlock(nn.Module):
    def __init__(self, batch_size, c):
        super(NormBlock, self).__init__()
        self.c = c
        self.batch_size = batch_size
        self.LayerNorm = nn.LayerNorm(self.c * 2, eps = 0.001)
        
    def forward(self, x):
        l_input = x.reshape([-1, self.c * 2])
        l_out = self.LayerNorm(l_input)
        glu_in = l_out.reshape([self.batch_size, self.c * 2, -1, 1])
        glu_out = GLU(glu_in)
        return glu_out


def mask_padding(src, lens, batch_size):
    res = src[0][:lens[0]]

    for i in range(1, batch_size):
        res = torch.cat([res, src[i][:lens[i]]], 0)
    return res

def PredictionResult(y_, batch_y, softmax_thr):
    # if softmax_thr:
    #     ofs_value = 0.5 - softmax_thr
    #     ofs = torch.tensor([-ofs_value, ofs_value], dtype=torch.float32)
    #     y_ = y_ + ofs

    prediction = torch.argmax(y_, 1)
    ground_truth = torch.argmax(batch_y, 1)
    correctness = torch.eq(prediction, ground_truth)
    accuracy = torch.mean(torch.tensor(correctness, dtype=torch.float32), dim=0)
    return prediction, correctness, accuracy

def final_results(pred_y, input_y, lens_y, batch_size, softmax_thr):
    grd_y = mask_padding(input_y, lens_y, batch_size)
    pred_analy = PredictionResult(pred_y, grd_y, softmax_thr)
    return pred_analy
