# model.py
import torch
import torch.nn as nn

# === GLOBAL CONFIG ===
IMAGE_SIZE = (256, 256)   # must match your preprocess_sequences
SEQ_LEN = 5               # must match your preprocessing sequence length


# ----- CNN Encoder -----
class CNNEncoder(nn.Module):
    def __init__(self, input_size=IMAGE_SIZE, channels=1):
        super().__init__()
        # Efficient CNN for top-down grayscale inputs
        self.features = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=5, stride=2, padding=2),  # /2
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),        # /4
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),        # /8
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),       # /16
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        # compute output feature dim automatically
        with torch.no_grad():
            dummy = torch.zeros(1, channels, *input_size)
            out = self.features(dummy)
            self.output_dim = out.shape[1]

    def forward(self, x):
        # x: (B*S, C, H, W)
        return self.features(x)  # -> (B*S, feat_dim)


# ----- GRU Decoder -----
class GRUDecoder(nn.Module):
    def __init__(self, feat_dim, hidden_dim=256, num_layers=1, dropout=0.1, output_dim=5):
        """
        GRU decoder for temporal aggregation.
        output_dim=5 for [forward, back, left, right, speed]
        """
        super().__init__()
        self.gru = nn.GRU(
            input_size=feat_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0)
        )
        self.head = nn.Sequential(
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, output_dim)
        )

        # initialize weights
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        # x: (B, seq_len, feat_dim)
        out, _ = self.gru(x)
        last = out[:, -1, :]       # final time step
        return self.head(last)     # -> (B, 5) continuous outputs

# ----- MLP Decoder -----
class MLPDecoder(nn.Module):
    def __init__(self, feat_dim, hidden_dim=256, output_dim=5, dropout=0.1):
        """
        Simple MLP decoder for temporal aggregation via flattening or averaging.
        - feat_dim: feature size from CNN encoder
        - output_dim: number of targets (5 for [f, b, l, r, speed])
        """
        super().__init__()
        # Process the average of frame features (temporal pooling)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        # Initialize weights
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, feats):
        """
        feats: (B, seq_len, feat_dim)
        returns: (B, output_dim)
        """
        # Temporal average pooling
        pooled = feats.mean(dim=1)   # (B, feat_dim)
        return self.mlp(pooled)      # (B, output_dim)


# ----- Combined Model -----
class CNNMLP_(nn.Module):
    """
    CNN-MLP model for no-sequence-based control/speed prediction.
    Input:  (B, seq_len, 1, H, W)
    Output: (B, 5) continuous [forward, back, left, right, speed]
    """
    def __init__(self, image_size=IMAGE_SIZE, seq_len=SEQ_LEN, hidden_dim=256, num_outputs=5):
        super().__init__()
        self.seq_len = seq_len
        self.encoder = CNNEncoder(input_size=image_size, channels=1)
        self.decoder = MLPDecoder(
            feat_dim=self.encoder.output_dim,
            hidden_dim=hidden_dim,
            output_dim=num_outputs
        )

    def forward(self, x):
        B, S, C, H, W = x.shape
        x = x[:, -1]                      # x: (B, S, C, H, W) -> out: (B, C, H, W)
        feats = self.encoder(x)           # (B, feat_dim)
        feats = feats.view(B, 1, -1)      # (B, S, feat_dim)
        return self.decoder(feats)        # (B, 5)

class CNNMLP(nn.Module):
    """
    CNN-MLP model for sequence-based control/speed prediction.
    Input:  (B, seq_len, 1, H, W)
    Output: (B, 5) continuous [forward, back, left, right, speed]
    """
    def __init__(self, image_size=IMAGE_SIZE, seq_len=SEQ_LEN, hidden_dim=256, num_outputs=5):
        super().__init__()
        self.seq_len = seq_len
        self.encoder = CNNEncoder(input_size=image_size, channels=1)
        self.decoder = MLPDecoder(
            feat_dim=self.encoder.output_dim,
            hidden_dim=hidden_dim,
            output_dim=num_outputs
        )

    def forward(self, x):
        B, S, C, H, W = x.shape
        assert S == self.seq_len, f"Expected seq_len={self.seq_len}, got {S}"
        x = x.view(B * S, C, H, W)
        feats = self.encoder(x)           # (B*S, feat_dim)
        feats = feats.view(B, S, -1)      # (B, S, feat_dim)
        return self.decoder(feats)        # (B, 5)

class CNNGRU(nn.Module):
    """
    CNN-GRU model for sequence-based control/speed prediction.
    Input:  (B, seq_len, 1, H, W)
    Output: (B, 5) continuous [forward, back, left, right, speed]
    """
    def __init__(self, image_size=IMAGE_SIZE, seq_len=SEQ_LEN, hidden_dim=256, num_outputs=5):
        super().__init__()
        self.seq_len = seq_len
        self.encoder = CNNEncoder(input_size=image_size, channels=1)
        self.decoder = GRUDecoder(
            feat_dim=self.encoder.output_dim,
            hidden_dim=hidden_dim,
            output_dim=num_outputs
        )

    def forward(self, x):
        B, S, C, H, W = x.shape
        assert S == self.seq_len, f"Expected seq_len={self.seq_len}, got {S}"

        x = x.view(B * S, C, H, W)      # (B*S, C, H, W)
        feats = self.encoder(x)         # (B*S, feat_dim)
        feats = feats.view(B, S, -1)    # (B, S, feat_dim)
        return self.decoder(feats)      # (B, 5)
