import torch
import torch.nn as nn
from mmseg.ops import resize

class extend_head(nn.Module):
    def __init__(self, input_channels) -> None:
        super().__init__()

        self.expand1 = nn.Conv2d(in_channels=input_channels, 
                                 out_channels=2, 
                                 kernel_size=3,
                                 padding = 1, 
                                 padding_mode='replicate',
                                 bias=False)
    
    def forward(self,x):
        x_x4 = resize(
            input=x,
            size=torch.Size((256,256)),
            mode='bilinear',
            align_corners=False
        )
        x_x4 = self.expand1(x_x4)
        return x_x4


        
        