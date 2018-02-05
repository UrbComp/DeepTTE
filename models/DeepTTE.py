import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import base
import numpy as np

from torch.autograd import Variable

EPS = 10

class EntireEstimator(nn.Module):
    def __init__(self, input_size, num_final_fcs, hidden_size = 128):
        super(EntireEstimator, self).__init__()

        self.input2hid = nn.Linear(input_size, hidden_size)

        self.residuals = nn.ModuleList()
        for i in range(num_final_fcs):
            self.residuals.append(nn.Linear(hidden_size, hidden_size))

        self.hid2out = nn.Linear(hidden_size, 1)

    def forward(self, attr_t, sptm_t):
        inputs = torch.cat((attr_t, sptm_t), dim = 1)

        hidden = F.leaky_relu(self.input2hid(inputs))

        for i in range(len(self.residuals)):
            residual = F.leaky_relu(self.residuals[i](hidden))
            hidden = hidden + residual

        out = self.hid2out(hidden)

        return out

    def eval_on_batch(self, pred, label, mean, std):
        label = label.view(-1, 1)

        label = label * std + mean
        pred = pred * std + mean

        loss = torch.abs(pred - label) / label

        return {'label': label, 'pred': pred}, loss.mean()

class LocalEstimator(nn.Module):
    def __init__(self, input_size):
        super(LocalEstimator, self).__init__()

        self.input2hid = nn.Linear(input_size, 64)
        self.hid2hid = nn.Linear(64, 32)
        self.hid2out = nn.Linear(32, 1)

    def forward(self, sptm_s):
        hidden = F.leaky_relu(self.input2hid(sptm_s))

        hidden = F.leaky_relu(self.hid2hid(hidden))

        out = self.hid2out(hidden)

        return out

    def eval_on_batch(self, pred, lens, label, mean, std):
        label = nn.utils.rnn.pack_padded_sequence(label, lens, batch_first = True)[0]
        label = label.view(-1, 1)

        label = label * std + mean
        pred = pred * std + mean

        loss = torch.abs(pred - label) / (label + EPS)

        return loss.mean()


class Net(nn.Module):
    def __init__(self, kernel_size = 3, num_filter = 32, pooling_method = 'attention', num_final_fcs = 3, final_fc_size = 128, alpha = 0.3):
        super(Net, self).__init__()

        # parameter of attribute / spatio-temporal component
        self.kernel_size = kernel_size
        self.num_filter = num_filter
        self.pooling_method = pooling_method

        # parameter of multi-task learning component
        self.num_final_fcs = num_final_fcs
        self.final_fc_size = final_fc_size
        self.alpha = alpha

        self.build()
        self.init_weight()

    def init_weight(self):
        for name, param in self.named_parameters():
            if name.find('.bias') != -1:
                param.data.fill_(0)
            elif name.find('.weight') != -1:
                nn.init.xavier_uniform(param.data)

    def build(self):
        # attribute component
        self.attr_net = base.Attr.Net()

        # spatio-temporal component
        self.spatio_temporal = base.SpatioTemporal.Net(attr_size = self.attr_net.out_size(), \
                                                       kernel_size = self.kernel_size, \
                                                       num_filter = self.num_filter, \
                                                       pooling_method = self.pooling_method
        )

        self.entire_estimate = EntireEstimator(input_size =  self.spatio_temporal.out_size() + self.attr_net.out_size(), num_final_fcs = self.num_final_fcs, hidden_size = self.final_fc_size)

        self.local_estimate = LocalEstimator(input_size = self.spatio_temporal.out_size())


    def forward(self, attr, traj, config):
        attr_t = self.attr_net(attr)

        # sptm_s: hidden sequence (B * T * F); sptm_l: lens (list of int); sptm_t: merged tensor after attention/mean pooling
        sptm_s, sptm_l, sptm_t = self.spatio_temporal(traj, attr_t, config)

        entire_out = self.entire_estimate(attr_t, sptm_t)

        # sptm_s is a packed sequence (see pytorch doc for details), only used during the training
        if self.training:
            local_out = self.local_estimate(sptm_s[0])
            return entire_out, (local_out, sptm_l)
        else:
            return entire_out

    def eval_on_batch(self, attr, traj, config):
        if self.training:
            entire_out, (local_out, local_length) = self(attr, traj, config)
        else:
            entire_out = self(attr, traj, config)

        pred_dict, entire_loss = self.entire_estimate.eval_on_batch(entire_out, attr['time'], config['time_mean'], config['time_std'])

        if self.training:
            # get the mean/std of each local path
            mean, std = (self.kernel_size - 1) * config['time_gap_mean'], (self.kernel_size - 1) * config['time_gap_std']

            # get ground truth of each local path
            local_label = utils.get_local_seq(traj['time_gap'], self.kernel_size, mean, std)
            local_loss = self.local_estimate.eval_on_batch(local_out, local_length, local_label, mean, std)

            return pred_dict, (1 - self.alpha) * entire_loss + self.alpha * local_loss
        else:
            return pred_dict, entire_loss
