import torch
import torch.nn as nn
import torch.nn.functional as F

class myCNN(nn.Module):
    def __init__(self, dropout_p=0.3):
        super().__init__()
        self.model = nn.Sequential([
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.Dropout2d(p=dropout_p),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),

            nn.AdaptiveAvgPool2d((1, 1)),  # GlobalAveragePooling2D
        ])
        # -> (batch,2,128,128)
        # -> (batch,32,64,64)
        # -> (batch,64,32,32)
        # -> (batch,128,16,16)
        
        # -> (batch,256,8,8)
        # -> (batch,256,1,1)

    def forward(self, x): # -> batch x 2(current,last) x 128x128(tail image grayscale)
        return self.model(x)

class myDriver(nn.Module):
    def __init__(self, dropout_p=0.3):
        super().__init__()
        self.model = nn.Sequential([
            nn.Linear(256, 128),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, 5)  # 4 commands + 1 speed( + 15 raycast)
        ])
        # -> (batch,256)
        # -> (batch,128)
        
        # -> (batch,5) raw logits

    def forward(self,x): # -> batch x 256 x 1 x 1
        return self.model(x)