import torch
import torch.nn as nn

class MultiScaleConvBlock(nn.Module):
    """
    "Lite" Inception Block for Small Datasets.
    Uses 1x1 Bottlenecks to reduce parameters and prevent overfitting.
    """
    def __init__(self, in_channels, out_channels, dropout=0.2): # Increased dropout default
        super().__init__()
        
        # Calculate bottleneck dimension (usually 1/4 of out_channels)
        inter_channels = out_channels // 2
        
        # Branch 1: Fine details (Kernel 3)
        # 1x1 Conv (Bottleneck) -> 3x3 Conv
        self.branch_small = nn.Sequential(
            nn.Conv1d(in_channels, inter_channels, kernel_size=1), # Squeeze
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(),
            nn.Conv1d(inter_channels, inter_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Branch 2: Coarse context (Kernel 7)
        # 1x1 Conv (Bottleneck) -> 7x7 Conv
        self.branch_large = nn.Sequential(
            nn.Conv1d(in_channels, inter_channels, kernel_size=1), # Squeeze
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(),
            # Groups=inter_channels makes this a Depthwise Conv (Very low params!)
            nn.Conv1d(inter_channels, inter_channels, kernel_size=7, padding=3, groups=inter_channels),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Residual connection
        if in_channels != out_channels:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_proj = nn.Identity()
            
        self.act = nn.ReLU()

    def forward(self, x):
        small = self.branch_small(x)
        large = self.branch_large(x)
        out = torch.cat([small, large], dim=1)
        res = self.residual_proj(x)
        return self.act(out + res)

class PoseStream(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Input Projection: Map 154 dims to first hidden dim
        self.input_projection = nn.Sequential(
            nn.Conv1d(config.LANDMARK_DIMS, config.POSE_HIDDEN_DIMS[0], kernel_size=1),
            nn.BatchNorm1d(config.POSE_HIDDEN_DIMS[0]),
            nn.ReLU()
        )
        
        # Stacking Multi-Scale Blocks dynamically based on Config
        layers = []
        in_dim = config.POSE_HIDDEN_DIMS[0]
        
        # If you have multiple hidden dims in config [256, 512], this builds layers for them
        for hidden_dim in config.POSE_HIDDEN_DIMS:
            layers.append(MultiScaleConvBlock(in_dim, hidden_dim, config.DROPOUT))
            in_dim = hidden_dim
            
        self.feature_extractor = nn.Sequential(*layers)
        
        # Final projection to Transformer dimension (512)
        self.to_transformer = nn.Linear(config.POSE_HIDDEN_DIMS[-1], config.POSE_FEATURE_DIM)
        
        # ========== Transformer Section ==========
        self.positional_encoding = PositionalEncoding(
            config.POSE_FEATURE_DIM, 
            config.NUM_FRAMES, 
            config.DROPOUT
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.POSE_FEATURE_DIM,
            nhead=config.NUM_ATTENTION_HEADS,
            dim_feedforward=config.FEEDFORWARD_DIM,
            dropout=config.DROPOUT,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config.NUM_TRANSFORMER_LAYERS
        )

        print(f"✓ Pose Stream initialized (Multi-Scale)")

    def forward(self, landmarks):
        """
        Args:
            landmarks: (B, T, 154)
        Returns:
            features: (B, T, D)
        """
        # 1. Transpose for Conv1D: (B, 154, T)
        x = landmarks.transpose(1, 2)
        
        # 2. Input Projection
        x = self.input_projection(x)
        
        # 3. Multi-Scale Feature Extraction
        x = self.feature_extractor(x)  # (B, Last_Hidden_Dim, T)
        
        # 4. Transpose back for Transformer: (B, T, Last_Hidden_Dim)
        x = x.transpose(1, 2)
        
        # 5. Project to Embedding Dim
        x = self.to_transformer(x)
        
        # 6. Transformer
        x = self.positional_encoding(x)
        features = self.transformer(x)
        
        return features

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)









