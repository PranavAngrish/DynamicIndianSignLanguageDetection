import torch
import numpy as np
import cv2
import json
from pathlib import Path
import mediapipe as mp
import warnings
import sys
import os

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'

from preprocessing import HandSignDatasetPreprocessor, suppress_stderr
from models.two_stream_transformer import create_model
from configs.config import Config


class SignLanguageInference:
    """
    Single video inference pipeline for sign language recognition
    """
    
    def __init__(self, model_path, metadata_path, config=None, device=None):
        """
        Args:
            model_path: Path to trained model checkpoint (.pth)
            metadata_path: Path to metadata.json (contains class names)
            config: Config object (optional, will create default if None)
            device: torch device (optional, will auto-detect if None)
        """
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        
        # Setup config
        if config is None:
            self.config = Config()
        else:
            self.config = config
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"{'='*70}")
        print(f"SIGN LANGUAGE INFERENCE PIPELINE")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        
        # Load metadata (class names)
        self.load_metadata()
        
        # Initialize preprocessor (reusing your code)
        self.preprocessor = self.setup_preprocessor()
        
        # Load model
        self.model = self.load_model()
        
        print(f"{'='*70}\n")
    
    def load_metadata(self):
        """Load class names from metadata"""
        with open(self.metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.class_names = metadata['class_names']
        self.num_classes = metadata['num_classes']
        
        print(f"✓ Loaded {self.num_classes} classes:")
        for i, name in enumerate(self.class_names):
            print(f"  {i}: {name}")
    
    def setup_preprocessor(self):
        """Setup preprocessor using your existing code"""
        # # Create a dummy preprocessor instance to reuse methods
        # preprocessor = HandSignDatasetPreprocessor.__new__(HandSignDatasetPreprocessor)
        # preprocessor.config = self.config
        # preprocessor.augment = False  # No augmentation for inference
        # preprocessor.verbose = False
        
        
        # print(f"✓ Preprocessor initialized")
        # return preprocessor
        preprocessor = HandSignDatasetPreprocessor(
        config=self.config,
        input_dir=".",       # not used in inference
        output_dir=".",      # not used in inference
        augment=False,
        verbose=False
        )
        print("✓ Preprocessor initialized")
        return preprocessor
    
    def load_model(self):
        """Load trained model"""
        # Update config with correct number of classes
        self.config.NUM_CLASSES = self.num_classes
        
        # Create model
        model = create_model(self.config)
        
        # Load checkpoint (weights_only=False for PyTorch 2.6+)
        checkpoint = torch.load(
            self.model_path, 
            map_location=self.device,
            weights_only=False
        )
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"  Training accuracy: {checkpoint.get('train_acc', 'N/A')}")
            print(f"  Validation accuracy: {checkpoint.get('val_acc', 'N/A')}")
        else:
            model.load_state_dict(checkpoint)
            print(f"✓ Loaded model weights")
        
        model.eval()
        return model
    
    def preprocess_video(self, video_path):
        """
        Preprocess a single video using your existing pipeline
        
        Args:
            video_path: Path to video file
        Returns:
            frames: np.array of shape (T, H, W, 3)
            landmarks: np.array of shape (T, 154)
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        print(f"\nPreprocessing: {video_path.name}")
        
        # Initialize MediaPipe models (reusing your setup)
        with suppress_stderr():
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            mp_hands = mp.solutions.hands
            hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=1
            )
        
        try:
            # Use your existing preprocess_video method
            frames, landmarks = self.preprocessor.preprocess_video(
                video_path,
                pose,
                hands,
                self.config.TARGET_COUNT
            )
            
            print(f"  ✓ Extracted {len(frames)} frames")
            print(f"  ✓ Landmarks shape: {landmarks.shape}")
            
            # Check if valid landmarks exist
            valid_frames = np.sum(np.any(landmarks != 0, axis=1))
            print(f"  ✓ Valid landmark frames: {valid_frames}/{len(landmarks)}")
            
            if valid_frames < 5:
                print(f"  ⚠️  Warning: Very few valid landmarks detected!")
            
        finally:
            # Cleanup MediaPipe
            hands.close()
            pose.close()
        
        return frames, landmarks
    
    def prepare_model_input(self, frames, landmarks):
        """
        Convert preprocessed data to model input tensors
        MATCHES training dataloader preprocessing exactly
        
        Args:
            frames: np.array of shape (T, H, W, 3) - BGR frames from preprocessing
            landmarks: np.array of shape (T, 154)
        Returns:
            frames_tensor: torch.Tensor of shape (1, T, 3, H, W)
            landmarks_tensor: torch.Tensor of shape (1, T, 154)
        """
        T = len(frames)
        H, W = self.config.FRAME_SIZE
        
        # Process frames exactly like the dataloader
        processed_frames = np.zeros((T, 3, H, W), dtype=np.float32)
        
        for i, frame in enumerate(frames):
            # Force conversion to uint8 numpy array (matching dataloader)
            frame = np.asarray(frame, dtype=np.uint8)
            
            # Safety check for empty frames
            if frame.size == 0:
                continue
            
            # Resize to expected input size
            frame = cv2.resize(frame, (W, H))
            
            # BGR to RGB (matching dataloader)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1] (matching dataloader)
            frame = frame.astype(np.float32) / 255.0
            
            # Transpose to (C, H, W) (matching dataloader)
            frame = frame.transpose(2, 0, 1)
            
            processed_frames[i] = frame
        
        # Process landmarks (matching dataloader)
        landmarks = landmarks.astype(np.float32)
        
        # Convert to tensors
        frames_tensor = torch.from_numpy(processed_frames).float()  # (T, 3, H, W)
        landmarks_tensor = torch.from_numpy(landmarks).float()  # (T, 154)
        
        # Add batch dimension
        frames_tensor = frames_tensor.unsqueeze(0)  # (1, T, 3, H, W)
        landmarks_tensor = landmarks_tensor.unsqueeze(0)  # (1, T, 154)
        
        # Move to device
        frames_tensor = frames_tensor.to(self.device)
        landmarks_tensor = landmarks_tensor.to(self.device)
        
        return frames_tensor, landmarks_tensor
    
    @torch.no_grad()
    def predict(self, video_path, top_k=5):
        """
        Perform inference on a single video
        
        Args:
            video_path: Path to video file
            top_k: Number of top predictions to return
        Returns:
            dict with predictions and probabilities
        """
        # Preprocess video
        frames, landmarks = self.preprocess_video(video_path)
        
        # Prepare model input
        frames_tensor, landmarks_tensor = self.prepare_model_input(frames, landmarks)
        
        # Run inference
        print(f"\nRunning inference...")
        logits = self.model(frames_tensor, landmarks_tensor)  # (1, num_classes)
        
        # Get probabilities
        probabilities = torch.softmax(logits, dim=1)  # (1, num_classes)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities[0], k=min(top_k, self.num_classes))
        
        # Convert to numpy
        top_probs = top_probs.cpu().numpy()
        top_indices = top_indices.cpu().numpy()
        
        # Format results
        predictions = []
        for idx, prob in zip(top_indices, top_probs):
            predictions.append({
                'class': self.class_names[idx],
                'class_id': int(idx),
                'probability': float(prob),
                'confidence_percent': float(prob * 100)
            })
        
        # Print results
        print(f"\n{'='*70}")
        print(f"PREDICTION RESULTS")
        print(f"{'='*70}")
        print(f"Video: {Path(video_path).name}\n")
        
        for i, pred in enumerate(predictions, 1):
            print(f"{i}. {pred['class']:<20} - {pred['confidence_percent']:6.2f}%")
        
        print(f"{'='*70}\n")
        
        return {
            'video_path': str(video_path),
            'top_prediction': predictions[0],
            'all_predictions': predictions,
            'raw_logits': logits[0].cpu().numpy().tolist()
        }
    
    def predict_batch(self, video_paths, top_k=5):
        """
        Perform inference on multiple videos
        
        Args:
            video_paths: List of video file paths
            top_k: Number of top predictions to return
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for video_path in video_paths:
            try:
                result = self.predict(video_path, top_k=top_k)
                results.append(result)
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                results.append({
                    'video_path': str(video_path),
                    'error': str(e)
                })
        
        return results


def main():
    """Example usage"""
    
    # ========== Configuration ==========
    MODEL_PATH = 'best_model_new.pth'

    METADATA_PATH = 'metadata_new.json'
    VIDEO_PATH = 'video.mp4'

    
    # ========== Initialize Inference Pipeline ==========
    inference = SignLanguageInference(
        model_path=MODEL_PATH,
        metadata_path=METADATA_PATH
    )
    
    # ========== Single Video Inference ==========
    result = inference.predict(VIDEO_PATH, top_k=5)
    
    # # ========== Save Results ==========
    # output_path = Path(VIDEO_PATH).parent / f"{Path(VIDEO_PATH).stem}_predictions.json"
    # with open(output_path, 'w') as f:
    #     json.dump(result, f, indent=2)
    
    # print(f"Results saved to: {output_path}")
    
    # ========== Batch Inference Example ==========
    video_paths = [
         '/workspace/ISL-Pronouns/INFERENCE/Bag/20251211_001527.mp4',
        '/workspace/ISL-Pronouns/INFERENCE/Blue/20251211_000704.mp4',
        '/workspace/ISL-Pronouns/INFERENCE/Brown/20251211_001747.mp4',
        '/workspace/ISL-Pronouns/INFERENCE/City/20251211_001843.mp4',
        '/workspace/ISL-Pronouns/INFERENCE/Dog/20251211_001157.mp4',   
        '/workspace/ISL-Pronouns/INFERENCE/Evening/20251211_001341.mp4',
        '/workspace/ISL-Pronouns/INFERENCE/Hat/20251211_001822.mp4',     
        '/workspace/ISL-Pronouns/INFERENCE/He/20251211_001011.mp4',     #wrong -> red
        '/workspace/ISL-Pronouns/INFERENCE/I/20251211_000504.mp4',         #wrong -> today
        '/workspace/ISL-Pronouns/INFERENCE/Market/20251211_001549.mp4', 
        '/workspace/ISL-Pronouns/INFERENCE/Morning/20251211_001037.mp4',  
        '/workspace/ISL-Pronouns/INFERENCE/Park/20251211_001221.mp4',       #wrong -> city
        '/workspace/ISL-Pronouns/INFERENCE/Red/20251211_001501.mp4',
         '/workspace/ISL-Pronouns/INFERENCE/School/20251211_000835.mp4',
         '/workspace/ISL-Pronouns/INFERENCE/She/20251211_001247.mp4',        #wrong -> evening
         '/workspace/ISL-Pronouns/INFERENCE/Shirt/20251211_000746.mp4',
         '/workspace/ISL-Pronouns/INFERENCE/Small/20251211_001103.mp4', #  market    
        '/workspace/ISL-Pronouns/INFERENCE/Sunday/20251211_001712.mp4',
        '/workspace/ISL-Pronouns/INFERENCE/They/20251211_001632.mp4',      #wrong -> blue
        '/workspace/ISL-Pronouns/INFERENCE/Today/20251211_000614.mp4'      #


        
        
        
        # '/workspace/ISL-Pronouns/INFERENCE/Black/20251211_002351.mp4',
        # '/workspace/ISL-Pronouns/INFERENCE/Box/20251211_003911.mp4',        
        # '/workspace/ISL-Pronouns/INFERENCE/Child/20251211_003357.mp4',
        # '/workspace/ISL-Pronouns/INFERENCE/Doctor/20251211_002239.mp4',
        # '/workspace/ISL-Pronouns/INFERENCE/Dress/20251211_004249.mp4',
        # '/workspace/ISL-Pronouns/INFERENCE/Father/20251211_003616.mp4',
        # '/workspace/ISL-Pronouns/INFERENCE/Girl/20251211_004118.mp4',
        #  '/workspace/ISL-Pronouns/INFERENCE/Happy/20251211_003306.mp4',
        # '/workspace/ISL-Pronouns/INFERENCE/Heavy/20251211_003827.mp4',
        # '/workspace/ISL-Pronouns/INFERENCE/Home/20251211_003455.mp4',
        # '/workspace/ISL-Pronouns/INFERENCE/Hospital/20251211_003112.mp4',
        # '/workspace/ISL-Pronouns/INFERENCE/Howareyou/20251211_004447.mp4',
        # '/workspace/ISL-Pronouns/INFERENCE/Monday/20251211_003712.mp4',
        #  '/workspace/ISL-Pronouns/INFERENCE/Night/20251211_002323.mp4',
        #  '/workspace/ISL-Pronouns/INFERENCE/Shoe/20251211_002431.mp4',
        # '/workspace/ISL-Pronouns/INFERENCE/Street/20251211_004005.mp4',
        # '/workspace/ISL-Pronouns/INFERENCE/ThankYou/20251211_004525.mp4',
        # '/workspace/ISL-Pronouns/INFERENCE/Yellow/20251211_004209.mp4',
        # '/workspace/ISL-Pronouns/INFERENCE/You/20251211_003210.mp4'
    ]
    # batch_results = inference.predict_batch(video_paths)



if __name__ == '__main__':
    main()



