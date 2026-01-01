import torch
import torch.nn as nn
from models.visual_stream import VisualStream
from models.pose_stream import PoseStream
from models.fusion_module import create_fusion_module

class TwoStreamTransformer(nn.Module):
    """
    Two-Stream Transformer for Sign Language Recognition
    
    Architecture:
        Visual Stream: Frames → CNN → Transformer → Visual Features
        Pose Stream: Landmarks → Conv1D → Transformer → Pose Features
        Fusion: Cross-Modal Attention → Combined Features
        Classifier: MLP → Class Predictions
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # ========== Two Streams ==========
        self.visual_stream = VisualStream(config)
        self.pose_stream = PoseStream(config)
        
        # ========== Fusion Module ==========
        self.fusion = create_fusion_module(config)
        
        # ========== Classification Head ==========
        self.classifier = nn.Sequential(
            nn.Linear(config.FUSION_DIM, config.FUSION_DIM // 2),
            nn.LayerNorm(config.FUSION_DIM // 2),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.FUSION_DIM // 2, config.NUM_CLASSES)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        print(f"\n{'='*70}")
        print(f"TWO-STREAM TRANSFORMER MODEL")
        print(f"{'='*70}")
        print(f"✓ Complete model initialized")
        print(f"  Number of classes: {config.NUM_CLASSES}")
        print(f"  Total parameters: {self.count_parameters():,}")
        print(f"{'='*70}\n")
    
    def forward(self, frames, landmarks):
        """
        Args:
            frames: (B, T, C, H, W) - Video frames
            landmarks: (B, T, 154) - Landmark sequences
        Returns:
            logits: (B, num_classes) - Class logits
        """
        # Extract features from both streams
        visual_features = self.visual_stream(frames)  # (B, T, D)
        pose_features = self.pose_stream(landmarks)  # (B, T, D)
        
        # Fuse features
        fused_features = self.fusion(visual_features, pose_features)  # (B, D)
        
        # Classify
        logits = self.classifier(fused_features)  # (B, num_classes)
        
        return logits
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_feature_representations(self, frames, landmarks):
        """
        Extract intermediate feature representations (for analysis)
        
        Returns:
            dict with visual_features, pose_features, fused_features
        """
        with torch.no_grad():
            visual_features = self.visual_stream(frames)
            pose_features = self.pose_stream(landmarks)
            fused_features = self.fusion(visual_features, pose_features)
        
        return {
            'visual': visual_features,
            'pose': pose_features,
            'fused': fused_features
        }


def create_model(config):
    """Factory function to create the model"""
    model = TwoStreamTransformer(config)
    model = model.to(config.DEVICE)
    
    # Print model summary
    print(f"\nModel Device: {config.DEVICE}")
    print(f"Mixed Precision: {config.MIXED_PRECISION}")
    
    return model


if __name__ == '__main__':
    # Test model creation
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from configs.config import Config
    
    config = Config()
    config.NUM_CLASSES = 10  # Example
    
    model = create_model(config)
    
    # Test forward pass
    batch_size = 2
    frames = torch.randn(batch_size, config.NUM_FRAMES, 3, *config.FRAME_SIZE).to(config.DEVICE)
    landmarks = torch.randn(batch_size, config.NUM_FRAMES, config.LANDMARK_DIMS).to(config.DEVICE)
    
    with torch.no_grad():
        logits = model(frames, landmarks)
    
    print(f"\nTest forward pass:")
    print(f"  Input frames: {frames.shape}")
    print(f"  Input landmarks: {landmarks.shape}")
    print(f"  Output logits: {logits.shape}")
    print(f"\n✓ Model test passed!")





