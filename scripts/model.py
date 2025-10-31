import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2),  # [1,128,128] → [16,62,62]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2), # → [32,30,30]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2), # → [64,14,14]
            nn.ReLU(),
            nn.Flatten(),                               # → [64×14×14]
        )
        self.output_dim = 64 * 14 * 14

    def forward(self, x):  # x: [batch, 1, 128, 128]
        return self.conv(x)  # → [batch, output_dim]

class RNNDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=5):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):  # x: [batch, seq_len, input_dim]
        _, h = self.rnn(x)  # h: [1, batch, hidden_dim]
        return self.fc(h.squeeze(0))  # → [batch, 5]

class RallyAutopilot(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CNNEncoder()
        self.decoder = RNNDecoder(input_dim=self.encoder.output_dim)

    def forward(self, x):  # x: [batch, seq_len, 1, 128, 128]
        batch, seq_len, _, _, _ = x.shape
        x = x.view(batch * seq_len, 1, 128, 128)
        features = self.encoder(x)  # → [batch * seq_len, feat_dim]
        features = features.view(batch, seq_len, -1)
        return self.decoder(features)  # → [batch, 5]


class myCNN(nn.Module):
    def __init__(self, dropout_p=0.3):
        super().__init__()
        self.model = nn.Sequential([
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=2),
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