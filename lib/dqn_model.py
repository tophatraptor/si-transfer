import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(Variable(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc(conv_out)


class DQN_Hidden(DQN):
    def __init__(self, input_shape, n_actions, orignal_model):
        super(DQN_Hidden, self).__init__(input_shape, n_actions)

        layers = list(orignal_model.children())

        self.conv = layers[0]

        conv_out_size = self._get_conv_out(input_shape)

        self.fc = layers[1][:-1]



class DQN_AM(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN_AM, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc1 = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU())
        self.fc2 = nn.Linear(512, n_actions)

    def _get_conv_out(self, shape):
        o = self.conv(Variable(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def forward(self, x, hidden_bool=False):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)

        hidden = self.fc1(conv_out)

        if hidden_bool:
            return self.fc2(hidden), hidden
        else:
            return self.fc2(hidden)
