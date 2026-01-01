import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFusion(nn.Module):
    """
    Cross-modal attention fusion
    Allows visual and pose streams to attend to each other
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        d_model = config.FUSION_DIM
        
        # Cross-attention: Visual attends to Pose
        self.visual_to_pose_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=config.NUM_ATTENTION_HEADS,
            dropout=config.DROPOUT,
            batch_first=True
        )
        
        # Cross-attention: Pose attends to Visual
        self.pose_to_visual_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=config.NUM_ATTENTION_HEADS,
            dropout=config.DROPOUT,
            batch_first=True
        )
        
        # Fusion layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )
        
        print(f"✓ Attention Fusion initialized")
    
    def forward(self, visual_features, pose_features):
        """
        Args:
            visual_features: (B, T, D)
            pose_features: (B, T, D)
        Returns:
            fused_features: (B, D)
        """
        # Cross-attention
        attended_visual, _ = self.visual_to_pose_attention(
            query=visual_features,
            key=pose_features,
            value=pose_features
        )  # (B, T, D)
        
        attended_pose, _ = self.pose_to_visual_attention(
            query=pose_features,
            key=visual_features,
            value=visual_features
        )  # (B, T, D)
        
        # Global pooling (mean over time)
        visual_pooled = visual_features.mean(dim=1)  # (B, D)
        pose_pooled = pose_features.mean(dim=1)  # (B, D)
        attended_visual_pooled = attended_visual.mean(dim=1)  # (B, D)
        attended_pose_pooled = attended_pose.mean(dim=1)  # (B, D)
        
        # Concatenate all representations
        combined = torch.cat([
            visual_pooled,
            pose_pooled,
            attended_visual_pooled,
            attended_pose_pooled
        ], dim=1)  # (B, 4*D)
        
        # Fuse
        fused = self.fusion_layer(combined)  # (B, D)
        
        return fused


class ConcatFusion(nn.Module):
    """Simple concatenation fusion"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.FUSION_DIM * 2, config.FUSION_DIM),
            nn.LayerNorm(config.FUSION_DIM),
            nn.GELU(),
            nn.Dropout(config.DROPOUT)
        )
        
        print(f"✓ Concat Fusion initialized")
    
    def forward(self, visual_features, pose_features):
        """
        Args:
            visual_features: (B, T, D)
            pose_features: (B, T, D)
        Returns:
            fused_features: (B, D)
        """
        # Global average pooling
        visual_pooled = visual_features.mean(dim=1)  # (B, D)
        pose_pooled = pose_features.mean(dim=1)  # (B, D)
        
        # Concatenate
        combined = torch.cat([visual_pooled, pose_pooled], dim=1)  # (B, 2*D)
        
        # Fuse
        fused = self.fusion_layer(combined)  # (B, D)
        
        return fused


class GatedFusion(nn.Module):
    """Gated fusion with learnable weights"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        d_model = config.FUSION_DIM
        
        # Gate networks
        self.visual_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        self.pose_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(config.DROPOUT)
        )
        
        print(f"✓ Gated Fusion initialized")
    
    def forward(self, visual_features, pose_features):
        """
        Args:
            visual_features: (B, T, D)
            pose_features: (B, T, D)
        Returns:
            fused_features: (B, D)
        """
        # Global average pooling
        visual_pooled = visual_features.mean(dim=1)  # (B, D)
        pose_pooled = pose_features.mean(dim=1)  # (B, D)
        
        # Compute gates
        visual_gate = self.visual_gate(visual_pooled)
        pose_gate = self.pose_gate(pose_pooled)
        
        # Gated fusion
        fused = visual_gate * visual_pooled + pose_gate * pose_pooled  # (B, D)
        
        # Final fusion layer
        fused = self.fusion_layer(fused)  # (B, D)
        
        return fused


def create_fusion_module(config):
    """Factory function to create fusion module"""
    fusion_type = config.FUSION_TYPE.lower()
    
    if fusion_type == 'attention':
        return AttentionFusion(config)
    elif fusion_type == 'concat':
        return ConcatFusion(config)
    elif fusion_type == 'gated':
        return GatedFusion(config)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")