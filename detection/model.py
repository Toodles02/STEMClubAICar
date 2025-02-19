import torch.nn as nn 


class TrafficSignDetector(nn.Module):
    def __init__(self):
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            #TODO 
        )
        
    def forward(self, x):
        #TODO 
        return x 
    