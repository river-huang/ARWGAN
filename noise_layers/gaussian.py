import math

import utils
import torch
import torch.nn as nn
import torch.nn.functional as F

class Gaussian_blur(nn.Module):
    def __init__(self,kernel,sigma):
        super(Gaussian_blur, self).__init__()
        self.kernel=int(kernel)
        self.sigma=float(sigma)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    def forward(self, noised_and_cover):
        # get the gaussian filter
        encode_image=noised_and_cover[0]
        batch_size,channel=encode_image.shape[0],encode_image.shape[1]
        assert encode_image.shape[1]==3|1
        gaussian_kernel=utils.gaussian_kernel(self.sigma,self.kernel)
        kernel = torch.FloatTensor(gaussian_kernel).unsqueeze(0).unsqueeze(0)
        kernel=kernel.expand(channel,1,self.kernel,self.kernel).to(self.device)
        # print(kernel)
        # print(kernel.size())
        # print(encode_image.size())
        # weight = nn.Parameter(data=kernel, requires_grad=False)
        noised_and_cover[0]=F.conv2d(encode_image,kernel,stride=1,padding=1,groups=3)
        return noised_and_cover









        