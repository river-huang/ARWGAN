import torch
import torch.nn as nn
import numpy as np

class Salt_and_Pepper(nn.Module):
    def __init__(self,ratio):
        super(Salt_and_Pepper, self).__init__()
        self.ratio=float(ratio)
        self.device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    def forward(self, noise_and_cover):
        encoded_image=noise_and_cover[0]
        B,C,H,W=encoded_image.size()
        mask = np.random.choice((0, 1, 2), size=(B,1,H,W), p=[self.ratio, self.ratio, 1 - 2 * self.ratio])
        mask=torch.tensor(np.repeat(mask,C, axis=1),device=self.device)
        encoded_image[mask==0]=-1
        encoded_image[mask==1]=1
        noise_and_cover[0]=encoded_image
        return noise_and_cover

