"""
configs/gating_config.py
Configuration for the lightweight gating model
"""

import torch
from pathlib import Path

class GatingConfig:
    """Lightweight config for the gating model that classifies groups"""
    
    def __init__(self):
        # ========== Paths ==========

        self.PROJECT_ROOT = Path(__file__).parent.parent
        
        
        # Combined dataset paths for gating model training
        self.DATA_DIRS = [
            Path('/workspace/ISL-Pronouns/datasets_parallel_data_10_1'),
            Path('/workspace/ISL-Pronouns/datasets_parallel_data_10_3'),
            Path('/workspace/ISL-Pronouns/datasets_parallel_data_10_4')
        ]
        
        self.CHECKPOINT_DIR = self.PROJECT_ROOT / 'checkpoints' / 'gating_model'
        self.LOG_DIR = self.PROJECT_ROOT / 'logs' / 'gating_model'
        # self.LOG_DIR = self.PROJECT_ROOT / 'logs' / 'gating_model_1_4'
        # self.LOG_DIR = self.PROJECT_ROOT / 'logs' / 'gating_model_1_8'
        # self.LOG_DIR = self.PROJECT_ROOT / 'logs' / 'gating_model_1_16'
        
        # Create directories
        self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        # ========== Model Architecture (Lightweight) ==========
        self.NUM_CLASSES = 3  # Only 3 groups to classify
        self.NUM_FRAMES = 30
        self.FRAME_SIZE = (128, 128)
        self.LANDMARK_DIMS = 154
        
        # Smaller dimensions for faster training
        self.VISUAL_FEATURE_DIM = 256  # Reduced from 512
        self.POSE_FEATURE_DIM = 256    # Reduced from 512
        self.FUSION_DIM = 256          # Reduced from 512
        
        # Lighter transformer
        self.NUM_TRANSFORMER_LAYERS = 2  # Reduced from 4
        self.NUM_ATTENTION_HEADS = 4     # Reduced from 8  # 2 -> 4 # 3 -> 8 
        self.FEEDFORWARD_DIM = 512       # Reduced from 1024
        
        # CNN Backbone
        self.CNN_BACKBONE = 'mobilenetv3_large_100'  # Lighter backbone
        self.CNN_PRETRAINED = True
        
        # Pose stream
        self.POSE_HIDDEN_DIMS = [128, 256]  # Reduced complexity
        
        # Fusion
        self.FUSION_TYPE = 'gated'  # Simplest fusion

        # ========== Class Imbalance Handling ==========

        self.FOCAL_ALPHA = 0.25

        # Method 1: Focal Loss (RECOMMENDED) ⭐
        self.USE_FOCAL_LOSS = True
        self.FOCAL_GAMMA = 2.0  # Higher = more focus on hard examples
        # Focal loss will use class weights as alpha
        
        # Method 2: Weighted Cross Entropy (Alternative)
        # Set USE_FOCAL_LOSS=False to use this
        self.USE_CLASS_WEIGHTS = True  # Used by both Focal and Weighted CE
        
        # Method 3: Balanced Sampling (Can combine with above)
        self.USE_BALANCED_SAMPLING = False  # Set True to oversample minority class
        






        
        
        # ========== Training Hyperparameters ==========
        self.BATCH_SIZE = 24
        self.EVAL_BATCH_SIZE = 16
        self.NUM_EPOCHS = 30  # Shorter training
        self.LEARNING_RATE = 3e-4
        self.MIN_LR = 1e-6
        self.WEIGHT_DECAY = 0.01
        self.DROPOUT = 0.3
        
        self.OPTIMIZER = 'adamw'
        self.SCHEDULER = 'cosine'
        self.LABEL_SMOOTHING = 0.1
        self.GRADIENT_CLIP = 1.0
        
        # Regularization - No MixUp needed for gating
        self.USE_MIXUP = False
        self.MIXUP_ALPHA = 0.0
        
        # ========== Early Stopping ==========
        self.PATIENCE = 10
        self.SAVE_INTERVAL = 5
        
        # ========== Hardware ==========
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.NUM_WORKERS = 2
        self.PIN_MEMORY = True
        self.MIXED_PRECISION = torch.cuda.is_available()
        
        # ========== Logging ==========
        self.LOG_INTERVAL = 10
        self.TENSORBOARD = True
        
        print(f"✓ Gating Config initialized")
        print(f"  Device: {self.DEVICE}")
        print(f"  Model: Lightweight 3-class gating classifier")
    
    def validate(self):
        """Validate configuration"""
        for data_dir in self.DATA_DIRS:
            if not data_dir.exists():
                raise ValueError(f"Data directory not found: {data_dir}")
        
        print(f"✓ Config validated")