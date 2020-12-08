#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
from copy import deepcopy
import torch.nn.functional as F

import torchvision.models as models

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        try:
            nn.init.constant_(m.bias, 0.01)
        except:
            pass
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.constant_(m.bias, 0.01)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(60 + 1, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 512)
        self.linear4 = nn.Linear(512, 512)
        self.linear5 = nn.Linear(512, 3)
        self.linear6 = nn.Linear(512, 3)
        # self.linear7 = nn.Linear(512, 3)


        self.apply(weights_init)

    def forward(self, x, t):
        x = x.view(-1,60)
        x = torch.cat([x, t], dim=1)
        # x = torch.cat([x, v0], dim=1)
        x = self.linear1(x)
        # x = F.leaky_relu(x)
        x = torch.tanh(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear2(x)
        # x = F.leaky_relu(x)
        x = torch.tanh(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear3(x)
        # x = F.leaky_relu(x)
        x = torch.tanh(x)
        # x = F.dropout(x, p=0.5, training=self.training)

        x = self.linear4(x)
        # x = F.leaky_relu(x)
        x = torch.tanh(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        pos = self.linear5(x)
        vel = self.linear6(x)
        # acc = self.linear7(x)
        return pos,vel#,acc

class LSTM_xvq2vadq_0(nn.Module):
    def __init__(self, input_size=10, hidden_layer_size=100, output_size=10):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size,batch_first=True)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        predictions = self.linear(lstm_out)
        return predictions

class LSTM_xva2xva(nn.Module):
    def __init__(self, input_size=9, hidden_layer_size=100, output_size=9):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size,batch_first=True)

        self.linear1 = nn.Linear(hidden_layer_size, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        predictions = self.linear1(lstm_out)
        predictions = torch.sigmoid(predictions)
        predictions = self.linear2(predictions)
        predictions = torch.sigmoid(predictions)
        predictions = self.linear3(predictions)
        predictions = torch.sigmoid(predictions)
        return predictions

class LSTM_xva2xva_1(nn.Module):
    def __init__(self, input_size=10, hidden_layer_size=100, output_size=10):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size,batch_first=True)

        self.linear1 = nn.Linear(hidden_layer_size, 512)
        # self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        predictions = self.linear1(lstm_out)
        predictions = torch.sigmoid(predictions)
        # predictions = self.linear2(predictions)
        # predictions = torch.sigmoid(predictions)
        predictions = self.linear3(predictions)
        predictions = torch.sigmoid(predictions)
        return predictions






# class Encoder(nn.Module):
#     def __init__(self,input_size=9,output_size=128,dropout=0.5):
#         super().__init__()
#         self.linear = nn.Linear(input_size, output_size)
#         self.drop = nn.Dropout(dropout)
#         self.init_weights()
#     def forward(self, input_seq):
#         output_seq = self.linear(input_seq)
#         output_seq = self.drop(output_seq)
#         return output_seq
#     def init_weights(self):
#         initrange = 0.1
#         self.linear.weight.data.uniform_(-initrange, initrange)
#         self.linear.bias.data.fill_(0)
#
#
# class Decoder(nn.Module):
#     def __init__(self,input_size=128,output_size=9,dropout=0.5):
#         super().__init__()
#         self.linear = nn.Linear(input_size, output_size)
#         self.drop = nn.Dropout(dropout)
#         self.init_weights()
#     def forward(self, input_seq):
#         output_seq = self.linear(input_seq)
#         output_seq = self.drop(output_seq)
#         return output_seq
#     def init_weights(self):
#         initrange = 0.1
#         self.linear.weight.data.uniform_(-initrange, initrange)
#         self.linear.bias.data.fill_(0)

class ED_LSTM(nn.Module):
    def __init__(self, input_size=9, hidden_layer_size=256,output_size=9, nlayer = 2,dropout = 0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.lstm = nn.LSTM(hidden_layer_size, hidden_layer_size,nlayer,batch_first=True)
        self.nlayer = nlayer
        self.encoder1 = nn.Linear(input_size, 128)
        self.encoder2 = nn.Linear(128, hidden_layer_size)
        self.encoder3 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.decoder3 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.decoder2 = nn.Linear(hidden_layer_size, 128)
        self.decoder1 = nn.Linear(128, output_size)
        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def forward(self, input_seq,hidden):
        input_seq = self.encoder1(input_seq)
        input_seq = torch.tanh(input_seq)
        input_seq = self.encoder2(input_seq)
        input_seq = torch.tanh(input_seq)
        input_seq = self.encoder3(input_seq)
        input_seq = torch.tanh(input_seq)
        output_seq, hidden = self.lstm(input_seq, hidden)
        output_seq = self.drop(output_seq)
        output_seq = self.decoder3(output_seq)
        output_seq = torch.tanh(output_seq)
        output_seq = self.decoder2(output_seq)
        output_seq = torch.tanh(output_seq)
        output_seq = self.decoder1(output_seq)
        output_seq = torch.tanh(output_seq)
        return output_seq,hidden

    def recon(self,input_seq):
        input_seq = self.encoder1(input_seq)
        input_seq = torch.tanh(input_seq)
        input_seq = self.encoder2(input_seq)
        input_seq = torch.tanh(input_seq)
        input_seq = self.encoder3(input_seq)
        input_seq = torch.tanh(input_seq)
        output_seq = self.decoder3(input_seq)
        output_seq = torch.tanh(output_seq)
        output_seq = self.decoder2(output_seq)
        output_seq = torch.tanh(output_seq)
        output_seq = self.decoder1(output_seq)
        output_seq = torch.tanh(output_seq)
        return output_seq

    def repackage_hidden(self,h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == tuple:
            return tuple(self.repackage_hidden(v) for v in h)
        else:
            return h.detach()

    def init_weights(self):
        initrange = 0.1
        self.encoder1.bias.data.fill_(0)
        self.encoder1.weight.data.uniform_(-initrange, initrange)
        self.decoder1.bias.data.fill_(0)
        self.decoder1.weight.data.uniform_(-initrange, initrange)
        self.encoder2.bias.data.fill_(0)
        self.encoder2.weight.data.uniform_(-initrange, initrange)
        self.decoder2.bias.data.fill_(0)
        self.decoder2.weight.data.uniform_(-initrange, initrange)


    def init_hidden(self, bsz):
        weight = next(self.parameters()).data

        return (Variable(weight.new(self.nlayer, bsz, self.hidden_layer_size).zero_()),
                Variable(weight.new(self.nlayer, bsz, self.hidden_layer_size).zero_()))

class ED_LSTM_2(nn.Module):
    def __init__(self, input_size=9, hidden_layer_size=256,output_size=9, nlayer = 2,dropout = 0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.lstm = nn.LSTM(hidden_layer_size, hidden_layer_size,nlayer,batch_first=True)
        self.nlayer = nlayer
        self.encoder1 = nn.Linear(input_size, 128)
        self.encoder2 = nn.Linear(128, hidden_layer_size)
        self.decoder2 = nn.Linear(hidden_layer_size, 128)
        self.decoder1 = nn.Linear(128, output_size)
        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def forward(self, input_seq,hidden):
        input_seq = self.encoder1(input_seq)
        input_seq = torch.tanh(input_seq)
        input_seq = self.encoder2(input_seq)
        input_seq = torch.tanh(input_seq)
        output_seq, hidden = self.lstm(input_seq, hidden)
        output_seq = self.drop(output_seq)
        output_seq = self.decoder2(output_seq)
        output_seq = torch.tanh(output_seq)
        output_seq = self.decoder1(output_seq)
        output_seq = torch.tanh(output_seq)
        return output_seq,hidden

    def recon(self,input_seq):
        input_seq = self.encoder1(input_seq)
        input_seq = torch.tanh(input_seq)
        input_seq = self.encoder2(input_seq)
        input_seq = torch.tanh(input_seq)
        output_seq = self.decoder2(input_seq)
        output_seq = torch.tanh(output_seq)
        output_seq = self.decoder1(output_seq)
        output_seq = torch.tanh(output_seq)
        return output_seq

    def repackage_hidden(self,h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == tuple:
            return tuple(self.repackage_hidden(v) for v in h)
        else:
            return h.detach()

    def init_weights(self):
        initrange = 0.1
        self.encoder1.bias.data.fill_(0)
        self.encoder1.weight.data.uniform_(-initrange, initrange)
        self.decoder1.bias.data.fill_(0)
        self.decoder1.weight.data.uniform_(-initrange, initrange)
        self.encoder2.bias.data.fill_(0)
        self.encoder2.weight.data.uniform_(-initrange, initrange)
        self.decoder2.bias.data.fill_(0)
        self.decoder2.weight.data.uniform_(-initrange, initrange)


    def init_hidden(self, bsz):
        weight = next(self.parameters()).data

        return (Variable(weight.new(self.nlayer, bsz, self.hidden_layer_size).zero_()),
                Variable(weight.new(self.nlayer, bsz, self.hidden_layer_size).zero_()))

class ED_LSTM_3(nn.Module):
    def __init__(self, input_size=9, hidden_layer_size=256,output_size=9, nlayer = 2,dropout = 0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.lstm = nn.LSTM(hidden_layer_size, hidden_layer_size,nlayer,batch_first=True)
        self.nlayer = nlayer
        self.encoder1 = nn.Linear(input_size, 128)
        self.encoder2 = nn.Linear(128, hidden_layer_size)
        self.decoder2 = nn.Linear(hidden_layer_size, 128)
        self.decoder1 = nn.Linear(128, output_size)
        self.cov = nn.Linear(128,9)
        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def forward(self, input_seq,hidden):
        input_seq = self.encoder1(input_seq)
        input_seq = torch.tanh(input_seq)
        input_seq = self.encoder2(input_seq)
        input_seq = torch.tanh(input_seq)
        output_seq, hidden = self.lstm(input_seq, hidden)
        output_seq = self.drop(output_seq)
        output_seq = self.decoder2(output_seq)
        output_seq = torch.tanh(output_seq)
        output_cov = self.cov(output_seq)
        output_cov = torch.tanh(output_cov)
        output_seq = self.decoder1(output_seq)
        output_seq = torch.tanh(output_seq)
        return output_seq,hidden,output_cov

    def recon(self,input_seq):
        input_seq = self.encoder1(input_seq)
        input_seq = torch.tanh(input_seq)
        input_seq = self.encoder2(input_seq)
        input_seq = torch.tanh(input_seq)
        output_seq = self.decoder2(input_seq)
        output_seq = torch.tanh(output_seq)
        output_seq = self.decoder1(output_seq)
        output_seq = torch.tanh(output_seq)
        return output_seq

    def repackage_hidden(self,h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == tuple:
            return tuple(self.repackage_hidden(v) for v in h)
        else:
            return h.detach()

    def init_weights(self):
        initrange = 0.1
        self.encoder1.bias.data.fill_(0)
        self.encoder1.weight.data.uniform_(-initrange, initrange)
        self.decoder1.bias.data.fill_(0)
        self.decoder1.weight.data.uniform_(-initrange, initrange)
        self.encoder2.bias.data.fill_(0)
        self.encoder2.weight.data.uniform_(-initrange, initrange)
        self.decoder2.bias.data.fill_(0)
        self.decoder2.weight.data.uniform_(-initrange, initrange)


    def init_hidden(self, bsz):
        weight = next(self.parameters()).data

        return (Variable(weight.new(self.nlayer, bsz, self.hidden_layer_size).zero_()),
                Variable(weight.new(self.nlayer, bsz, self.hidden_layer_size).zero_()))

class ED_LSTM_4(nn.Module):
    def __init__(self, input_size=9, hidden_layer_size=256,output_size=9, nlayer = 2,dropout = 0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.lstm = nn.LSTM(hidden_layer_size, hidden_layer_size,nlayer,batch_first=True)
        self.nlayer = nlayer
        self.encoder1 = nn.Linear(input_size, 128)
        self.encoder2 = nn.Linear(128, hidden_layer_size)
        self.decoder2 = nn.Linear(hidden_layer_size, 128)
        self.decoder1 = nn.Linear(128, output_size)
        self.cov = nn.Linear(128,3)
        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def forward(self, input_seq,hidden):
        input_seq = self.encoder1(input_seq)
        input_seq = torch.tanh(input_seq)
        input_seq = self.encoder2(input_seq)
        input_seq = torch.tanh(input_seq)
        output_seq, hidden = self.lstm(input_seq, hidden)
        output_seq = self.drop(output_seq)
        output_seq = self.decoder2(output_seq)
        output_seq = torch.tanh(output_seq)
        output_cov = self.cov(output_seq)
        output_cov = torch.tanh(output_cov)
        output_seq = self.decoder1(output_seq)
        output_seq = torch.tanh(output_seq)
        return output_seq,hidden,output_cov

    def recon(self,input_seq):
        input_seq = self.encoder1(input_seq)
        input_seq = torch.tanh(input_seq)
        input_seq = self.encoder2(input_seq)
        input_seq = torch.tanh(input_seq)
        output_seq = self.decoder2(input_seq)
        output_seq = torch.tanh(output_seq)
        output_seq = self.decoder1(output_seq)
        output_seq = torch.tanh(output_seq)
        return output_seq

    def repackage_hidden(self,h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == tuple:
            return tuple(self.repackage_hidden(v) for v in h)
        else:
            return h.detach()

    def init_weights(self):
        initrange = 0.1
        self.encoder1.bias.data.fill_(0)
        self.encoder1.weight.data.uniform_(-initrange, initrange)
        self.decoder1.bias.data.fill_(0)
        self.decoder1.weight.data.uniform_(-initrange, initrange)
        self.encoder2.bias.data.fill_(0)
        self.encoder2.weight.data.uniform_(-initrange, initrange)
        self.decoder2.bias.data.fill_(0)
        self.decoder2.weight.data.uniform_(-initrange, initrange)


    def init_hidden(self, bsz):
        weight = next(self.parameters()).data

        return (Variable(weight.new(self.nlayer, bsz, self.hidden_layer_size).zero_()),
                Variable(weight.new(self.nlayer, bsz, self.hidden_layer_size).zero_()))

class Kalman_filter_torch():
    def __init__(self, dim_x, dim_z, dim_u=0,bsz=1):
        if dim_x < 1:
            raise ValueError('dim_x must be 1 or greater')
        if dim_z < 1:
            raise ValueError('dim_z must be 1 or greater')
        if dim_u < 0:
            raise ValueError('dim_u must be 0 or greater')

        def init_eye(bsz,dim_x):
            m = torch.ones((bsz, dim_x, dim_x))
            for i in range(m.shape[0]):
                m[i] = torch.eye(dim_x)
            return m

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.x = torch.zeros((bsz, dim_x, 1))        # state
        self.P = init_eye(bsz,dim_x)             # uncertainty covariance
        self.Q = torch.eye(dim_x)              # process uncertainty
        self.B = None                     # control transition matrix
        self.F = torch.eye(dim_x)                # state transition matrix
        self.H = torch.zeros((dim_z, dim_x))    # Measurement function
        self.R = init_eye(bsz,dim_z)              # state uncertainty
        self._alpha_sq = 1.               # fading memory control
        self.M = torch.zeros((bsz, dim_z, dim_z)) # process-measurement cross correlation
        self.z = torch.zeros((bsz, dim_z,1))

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.K = torch.zeros((bsz, dim_x, dim_z)) # kalman gain
        self.y = torch.zeros((bsz, dim_z, 1))
        self.S = torch.zeros((bsz, dim_z, dim_z)) # system uncertainty
        self.SI = torch.zeros((bsz, dim_z, dim_z)) # inverse system uncertainty

        # identity matrix. Do not alter this.
        self.I = init_eye(bsz,dim_x)

        # # these will always be a copy of x,P after predict() is called
        # self.x_prior = self.x.copy()
        # self.P_prior = self.P.copy()
        #
        # # these will always be a copy of x,P after update() is called
        # self.x_post = self.x.copy()
        # self.P_post = self.P.copy()

    def predict(self, u=None, B=None, F=None, Q=None):
        """
        Predict next state (prior) using the Kalman filter state propagation
        equations.

        Parameters
        ----------

        u : np.array
            Optional control vector. If not `None`, it is multiplied by B
            to create the control input into the system.

        B : np.array(dim_x, dim_z), or None
            Optional control transition matrix; a value of None
            will cause the filter to use `self.B`.

        F : np.array(dim_x, dim_x), or None
            Optional state transition matrix; a value of None
            will cause the filter to use `self.F`.

        Q : np.array(dim_x, dim_x), scalar, or None
            Optional process noise matrix; a value of None will cause the
            filter to use `self.Q`.
        """

        if B is None:
            B = self.B
        if F is None:
            F = self.F
        if Q is None:
            Q = self.Q

        # x = Fx + Bu
        if B is not None and u is not None:
            self.x = torch.matmul(F, self.x) + torch.matmul(B, u)
        else:
            self.x = torch.matmul(F, self.x)

        # P = FPF' + Q
        self.P = self._alpha_sq * torch.matmul(torch.matmul(F, self.P), F.T) + Q

        # # save prior
        # self.x_prior = self.x.copy()
        # self.P_prior = self.P.copy()

    def update(self, z, R=None, H=None):
        """
        Add a new measurement (z) to the Kalman filter.

        If z is None, nothing is computed. However, x_post and P_post are
        updated with the prior (x_prior, P_prior), and self.z is set to None.

        Parameters
        ----------
        z : (dim_z, 1): array_like
            measurement for this update. z can be a scalar if dim_z is 1,
            otherwise it must be convertible to a column vector.

        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.

        H : np.array, or None
            Optionally provide H to override the measurement function for this
            one call, otherwise self.H will be used.
        """

        if R is None:
            R = self.R

        if H is None:
            H = self.H

        # y = z - Hx
        # error (residual) between measurement and prediction
        self.y = z - torch.matmul(H, self.x)

        # common subexpression for speed
        PHT = torch.matmul(self.P, H.T)

        # S = HPH' + R
        # project system uncertainty into measurement space
        self.S = torch.matmul(H, PHT) + R
        self.SI = torch.inverse(self.S)
        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        self.K = torch.matmul(PHT, self.SI)

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = self.x + torch.matmul(self.K, self.y)

        # P = (I-KH)P(I-KH)' + KRK'
        # This is more numerically stable
        # and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature.

        I_KH = self.I - torch.matmul(self.K, H)

        # I_KHP = torch.matmul(I_KH, self.P)
        # I_KHT = I_KH.T
        # I_KHPI_KHT = torch.matmul(I_KHP, I_KHT)
        # KRK =  torch.matmul(torch.matmul(self.K, R), self.K.T)
        self.P = torch.matmul(torch.matmul(I_KH, self.P), torch.transpose(I_KH,1,2)) + \
                 torch.matmul(torch.matmul(self.K, R), torch.transpose(self.K,1,2))

        # save measurement and posterior state
        # self.z = deepcopy(z)
        # self.x_post = self.x.copy()
        # self.P_post = self.P.copy()

# class single_LSTM(nn.Module):
#     def __init__(self, input_size=9, hidden_layer_size=9, nlayer = 2):
#         super().__init__()
#         self.hidden_layer_size = hidden_layer_size
#
#         self.lstm = nn.LSTM(input_size, hidden_layer_size,nlayer,batch_first=True)
#         self.nlayer = nlayer
#
#     def forward(self, input_seq,hidden):
#         lstm_out, hidden = self.lstm(input_seq, hidden)
#         return lstm_out,hidden
#
#     def repackage_hidden(self,h):
#         """Wraps hidden states in new Variables, to detach them from their history."""
#         if type(h) == tuple:
#             return tuple(self.repackage_hidden(v) for v in h)
#         else:
#             return h.detach()
#
#     def init_hidden(self, bsz):
#         weight = next(self.parameters()).data
#
#         return (Variable(weight.new(self.nlayer, bsz, self.hidden_layer_size).zero_()),
#                 Variable(weight.new(self.nlayer, bsz, self.hidden_layer_size).zero_()))
#

