import torch
from pathlib import Path

class Config:
    # ============= PATHS =============
    DATA_DIR = Path("/workspace/ISL-Pronouns/datasets_parallel_data_10_3/npy_files")
    TRAIN_DIR = DATA_DIR / "train"
    VAL_DIR = DATA_DIR / "validation"
    CHECKPOINT_DIR = Path("checkpoints")
    LOG_DIR = Path("logs")
    
    # ============= DATA =============
    FRAME_SIZE = (128, 128)  # Resize frames to this
    NUM_FRAMES = 30
    LANDMARK_DIMS = 154
    NUM_CLASSES = None  # Will be set automatically from data
    TARGET_COUNT = 30
    
    # ============= MODEL ARCHITECTURE =============
    # Visual Stream (CNN + Transformer)
    CNN_BACKBONE = 'mobilenetv3_large_100'  # Options: efficientnet_b0, mobilenetv3_small
    CNN_PRETRAINED = True
    VISUAL_FEATURE_DIM = 256
    
    # Pose Stream (Conv1D + Transformer)
    POSE_HIDDEN_DIMS = [128, 256]
    POSE_FEATURE_DIM = 256
    
    # Transformer
    EMBEDDING_DIM = 256
    NUM_TRANSFORMER_LAYERS = 1
    NUM_ATTENTION_HEADS = 4
    FEEDFORWARD_DIM = 512
    DROPOUT = 0.5
    
    # Fusion
    FUSION_TYPE = 'gated'  # Options: 'concat', 'attention', 'gated'
    FUSION_DIM = 256
    
    # ============= TRAINING =============
    BATCH_SIZE = 32
    NUM_EPOCHS = 150
    LEARNING_RATE = 1.5e-3
    WEIGHT_DECAY = 1e-3
    LABEL_SMOOTHING = 0.2

    USE_MIXUP = True
    MIXUP_ALPHA = 0.4
    
    # Learning Rate Scheduler
    SCHEDULER = 'cosine'  # Options: 'cosine', 'step', 'plateau'
    WARMUP_EPOCHS = 5
    MIN_LR = 1e-6
    
    # Early Stopping
    PATIENCE = 20
    MIN_DELTA = 0.001
    
    # ============= OPTIMIZATION =============
    OPTIMIZER = 'adamw'  # Options: 'adam', 'adamw', 'sgd'
    GRADIENT_CLIP = 1.0
    MIXED_PRECISION = True  # FP16 training
    
    
    # ============= HARDWARE =============
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 2
    PIN_MEMORY = True
    
    # ============= LOGGING =============
    LOG_INTERVAL = 10  # Log every N batches
    SAVE_INTERVAL = 5  # Save checkpoint every N epochs
    TENSORBOARD = True
    
    # ============= EVALUATION =============
    EVAL_BATCH_SIZE = 32
    SAVE_CONFUSION_MATRIX = True
    COMPUTE_PER_CLASS_METRICS = True
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        assert cls.EMBEDDING_DIM == cls.VISUAL_FEATURE_DIM == cls.POSE_FEATURE_DIM, \
            "All feature dimensions must match for fusion"
        assert cls.NUM_ATTENTION_HEADS > 0 and cls.EMBEDDING_DIM % cls.NUM_ATTENTION_HEADS == 0, \
            "Embedding dim must be divisible by num attention heads"
        
        # Create directories
        cls.CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
        cls.LOG_DIR.mkdir(exist_ok=True, parents=True)
        
        print(f"✓ Configuration validated")
        print(f"  Device: {cls.DEVICE}")
        print(f"  Mixed Precision: {cls.MIXED_PRECISION}")
        print(f"  Batch Size: {cls.BATCH_SIZE}")















# import torch
# from pathlib import Path

# class Config:
#     # ============= PATHS =============
    
#     self.DATA_DIRS = [
#             '/workspace/ISL-Pronouns/datasets_parallel_data_10_4/datasets_parallel_10_4',
#             '/workspace/ISL-Pronouns/datasets_parallel_data_10_4/datasets_parallel_10_3',
#             '/workspace/ISL-Pronouns/datasets_parallel_data_10_4/datasets_parallel_10_1'
#         ]
    
#     TRAIN_DIR = DATA_DIR / "train"
#     VAL_DIR = DATA_DIR / "validation"
#     CHECKPOINT_DIR = Path("checkpoints")
#     LOG_DIR = Path("logs")

#     # Create directories
#     self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
#     self.LOG_DIR.mkdir(parents=True, exist_ok=True)
    
#     # ============= DATA =============
#     FRAME_SIZE = (128, 128)  # Resize frames to this
#     NUM_FRAMES = 30
#     LANDMARK_DIMS = 154
#     self.NUM_CLASSES = 3  # Will be set automatically from data
#     TARGET_COUNT = 30
    
#     # ============= MODEL ARCHITECTURE =============
#     # Visual Stream (CNN + Transformer)
#     CNN_BACKBONE = 'mobilenetv3_large_100'  # Options: efficientnet_b0, mobilenetv3_small
#     CNN_PRETRAINED = True
#     VISUAL_FEATURE_DIM = 256
    
#     # Pose Stream (Conv1D + Transformer)
#     POSE_HIDDEN_DIMS = [128, 256]
#     POSE_FEATURE_DIM = 256
    
#     # Transformer
#     EMBEDDING_DIM = 256
#     NUM_TRANSFORMER_LAYERS = 1
#     NUM_ATTENTION_HEADS = 4
#     FEEDFORWARD_DIM = 512
#     DROPOUT = 0.5
    
#     # Fusion
#     FUSION_TYPE = 'gated'  # Options: 'concat', 'attention', 'gated'
#     FUSION_DIM = 256
    
#     # ============= TRAINING =============
#     BATCH_SIZE = 32
#     NUM_EPOCHS = 150
#     LEARNING_RATE = 1.5e-3
#     WEIGHT_DECAY = 1e-3
#     LABEL_SMOOTHING = 0.2

#     USE_MIXUP = True
#     MIXUP_ALPHA = 0.4
    
#     # Learning Rate Scheduler
#     SCHEDULER = 'cosine'  # Options: 'cosine', 'step', 'plateau'
#     WARMUP_EPOCHS = 5
#     MIN_LR = 1e-6
    
#     # Early Stopping
#     PATIENCE = 20
#     MIN_DELTA = 0.001
    
#     # ============= OPTIMIZATION =============
#     OPTIMIZER = 'adamw'  # Options: 'adam', 'adamw', 'sgd'
#     GRADIENT_CLIP = 1.0
#     MIXED_PRECISION = True  # FP16 training
    
    
#     # ============= HARDWARE =============
#     DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     NUM_WORKERS = 4
#     PIN_MEMORY = True
    
#     # ============= LOGGING =============
#     LOG_INTERVAL = 10  # Log every N batches
#     SAVE_INTERVAL = 5  # Save checkpoint every N epochs
#     TENSORBOARD = True
    
#     # ============= EVALUATION =============
#     EVAL_BATCH_SIZE = 32
#     SAVE_CONFUSION_MATRIX = True
#     COMPUTE_PER_CLASS_METRICS = True
    
#     @classmethod
#     def validate(cls):
#         """Validate configuration"""
#         assert cls.EMBEDDING_DIM == cls.VISUAL_FEATURE_DIM == cls.POSE_FEATURE_DIM, \
#             "All feature dimensions must match for fusion"
#         assert cls.NUM_ATTENTION_HEADS > 0 and cls.EMBEDDING_DIM % cls.NUM_ATTENTION_HEADS == 0, \
#             "Embedding dim must be divisible by num attention heads"
        
#         # Create directories
#         cls.CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
#         cls.LOG_DIR.mkdir(exist_ok=True, parents=True)
        
#         print(f"✓ Configuration validated")
#         print(f"  Device: {cls.DEVICE}")
#         print(f"  Mixed Precision: {cls.MIXED_PRECISION}")
#         print(f"  Batch Size: {cls.BATCH_SIZE}")