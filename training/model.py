import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


class AMiL(nn.Module):

    def __init__(self):


        super(AMiL, self).__init__()

        self.L = 1000  # condense every image into self.L features (further encoding before actual MIL starts)
        self.D = 128  # hidden layer size for attention network


        # feature extractor before multiple instance learning starts

        self.ftr_proc = nn.Sequential(
#            nn.Conv2d(2048, 1024, kernel_size=1),
#           nn.ReLU(),
#            nn.Conv2d(1024, 512, kernel_size=1),
#            nn.ReLU(),
            nn.Conv2d(512, 300, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(300, 200, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(200, 150, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(150, 1000, kernel_size=2),
            nn.ReLU(),
            nn.Flatten()
        )

        # Networks for single attention approach
        ##### attention network (single attention approach)
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, 1)
        )

        ##### classifier (single attention approach)
        self.classifier = nn.Sequential(
            nn.Linear(self.L, 256),
            nn.ReLU(),
            nn.Linear(256, 6)
        )

    def forward(self, x):

        ft = self.ftr_proc(x)
#        print(ft.shape)
        # calculate attention
        att_raw = self.attention(ft)
        att_raw = torch.transpose(att_raw, 1, 0)

        att_softmax = F.softmax(att_raw, dim=1)
        bag_features = torch.mm(att_softmax, ft)

        prediction = self.classifier(bag_features)

        return prediction, att_softmax








