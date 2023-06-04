import numpy as np
import torch.nn as nn
import torch

class grid_crop(nn.Module):
    def __init__(self,rate):
        super(grid_crop, self).__init__()
        self.rate=float(rate)
        self.device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, noised_and_cover):
        encoded_image=noised_and_cover[0].clone()
        block_size = 8
        block_switch = np.random.uniform(0.0, high=1.0, size=[encoded_image.shape[2] // block_size, encoded_image.shape[3] // block_size])
        block_switch = block_switch < self.rate
        # new_img = img.copy()
        for i in range(block_switch.shape[0]):
            for j in range(block_switch.shape[1]):
                if block_switch[i, j] == 0:
                    continue
                encoded_image[:,:,block_size * i:block_size + block_size * i, block_size * j:block_size + block_size * j] = -1
        noised_and_cover[0]=encoded_image
        return noised_and_cover