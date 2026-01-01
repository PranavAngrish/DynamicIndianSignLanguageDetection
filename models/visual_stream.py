import torch
import torch.nn as nn
import timm
from einops import rearrange

class VisualStream(nn.Module):
    """
    Visual stream: Processes video frames
    Architecture: Per-frame CNN → Temporal Transformer
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # ========== CNN Feature Extractor ==========
        # Use efficient pretrained CNN
        self.cnn = timm.create_model(
            config.CNN_BACKBONE,
            pretrained=config.CNN_PRETRAINED,
            num_classes=0,  # Remove classification head
            global_pool=''  # Remove global pooling
        )
        
        # Get CNN output dimension
        # === CRITICAL CHANGE: FREEZE CNN WEIGHTS ===
        for param in self.cnn.parameters():
            param.requires_grad = False
        print(f"✓ CNN Backbone Frozen")
        
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, config.FRAME_SIZE[0], config.FRAME_SIZE[1])
            cnn_features = self.cnn(dummy_input)
            self.cnn_out_dim = cnn_features.shape[1]  # Channel dimension
        
        # Project CNN features to embedding dimension
        self.feature_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(self.cnn_out_dim, config.VISUAL_FEATURE_DIM),
            nn.LayerNorm(config.VISUAL_FEATURE_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT)
        )
        
        # ========== Positional Encoding ==========
        self.positional_encoding = PositionalEncoding(
            d_model=config.VISUAL_FEATURE_DIM,
            max_len=config.NUM_FRAMES,
            dropout=config.DROPOUT
        )
        
        # ========== Transformer Encoder ==========
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.VISUAL_FEATURE_DIM,
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
        
        print(f"✓ Visual Stream initialized")
        print(f"  CNN: {config.CNN_BACKBONE} (output dim: {self.cnn_out_dim})")
        print(f"  Feature dim: {config.VISUAL_FEATURE_DIM}")
    
    def forward(self, frames):
        """
        Args:
            frames: (B, T, C, H, W) - Batch of video frames
        Returns:
            features: (B, T, D) - Temporal features
        """
        B, T, C, H, W = frames.shape
        
        # Reshape for CNN: (B*T, C, H, W)
        frames = rearrange(frames, 'b t c h w -> (b t) c h w')
        
        # Extract CNN features
        cnn_features = self.cnn(frames)  # (B*T, C', H', W')
        
        # Project to embedding dimension
        frame_features = self.feature_projection(cnn_features)  # (B*T, D)
        
        # Reshape back to temporal sequence
        frame_features = rearrange(frame_features, '(b t) d -> b t d', b=B, t=T)
        
        # Add positional encoding
        frame_features = self.positional_encoding(frame_features)
        
        # Apply transformer
        visual_features = self.transformer(frame_features)  # (B, T, D)
        
        return visual_features


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences"""
    
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, D)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)






