import cv2
import numpy as np
import torch
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


class ModelInputVisualizer:
    """
    Visualizes model inputs by saving frames with landmarks overlaid.
    Captures the exact data being fed to the model for debugging and analysis.
    """
    
    def __init__(self, output_dir="visualization_outputs", config=None):
        """
        Args:
            output_dir: Directory to save visualizations
            config: Config object with FRAME_SIZE and other parameters
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        
        # Landmark connection definitions for drawing
        self.HAND_CONNECTIONS = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20),
        ]
        
        self.POSE_CONNECTIONS = [
            (0, 1), (0, 2),  # Nose to shoulders
            (1, 3), (2, 4),  # Shoulders to elbows
            (3, 5), (4, 6),  # Elbows to wrists
        ]
        
        print(f"✓ Visualizer initialized. Outputs will be saved to: {self.output_dir}")
    
    def unnormalize_landmarks(self, normalized_landmarks, original_frames):
        """
        Reverse the normalization process to get pixel coordinates.
        
        Args:
            normalized_landmarks: (T, 154) normalized landmarks
            original_frames: List of original frames to get dimensions
            
        Returns:
            unnormalized_landmarks: (T, 154) in pixel coordinates
        """
        T = normalized_landmarks.shape[0]
        H, W = original_frames[0].shape[:2]
        
        # Extract components
        left_hand = normalized_landmarks[:, :63].reshape(T, 21, 3)
        right_hand = normalized_landmarks[:, 63:126].reshape(T, 21, 3)
        pose = normalized_landmarks[:, 126:].reshape(T, 7, 4)  # xyz + visibility
        
        # Collect all normalized points
        all_xyz = np.concatenate([
            left_hand.reshape(T * 21, 3),
            right_hand.reshape(T * 21, 3),
            pose[:, :, :3].reshape(T * 7, 3)
        ], axis=0)
        
        # Find valid points to calculate statistics
        valid_mask = np.any(all_xyz != 0, axis=1)
        
        if np.any(valid_mask):
            valid_coords = all_xyz[valid_mask]
            
            # Reverse normalization
            # 1. Reverse z-score normalization for z-coordinate
            if len(valid_coords[:, 2]) > 1:
                z_std = np.std(valid_coords[:, 2])
                z_mean = np.mean(valid_coords[:, 2])
                if z_std > 0:
                    all_xyz[:, 2] = all_xyz[:, 2] * z_std + z_mean
            
            # 2. Reverse scale normalization for x, y
            max_dist = np.max(np.linalg.norm(valid_coords[:, :2], axis=1))
            if max_dist > 0:
                all_xyz[:, :2] *= max_dist
            
            # 3. Reverse centering
            center = np.mean(valid_coords[:, :2], axis=0)
            all_xyz[:, :2] += center
            
            # 4. Convert to pixel coordinates (assuming normalized coords are in [0, 1] range)
            all_xyz[:, 0] *= W
            all_xyz[:, 1] *= H
        
        # Reconstruct landmarks
        left_hand = all_xyz[:T * 21].reshape(T, 21, 3)
        right_hand = all_xyz[T * 21:T * 42].reshape(T, 21, 3)
        pose_xyz = all_xyz[T * 42:].reshape(T, 7, 3)
        
        unnormalized = np.zeros((T, 154))
        for t in range(T):
            unnormalized[t] = np.concatenate([
                left_hand[t].flatten(),
                right_hand[t].flatten(),
                pose_xyz[t].flatten(),
                pose[t, :, 3]  # Keep visibility values
            ])
        
        return unnormalized
    
    def draw_hand_landmarks(self, frame, hand_landmarks, color=(0, 255, 0), is_left=True):
        """
        Draw hand landmarks and connections on frame.
        
        Args:
            frame: Image to draw on
            hand_landmarks: (21, 3) array of x, y, z coordinates
            color: BGR color tuple
            is_left: Whether this is left hand (for labeling)
        """
        H, W = frame.shape[:2]
        
        # Draw connections
        for connection in self.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            start_point = hand_landmarks[start_idx]
            end_point = hand_landmarks[end_idx]
            
            # Check if both points are valid
            if np.any(start_point[:2] != 0) and np.any(end_point[:2] != 0):
                start_px = (int(start_point[0]), int(start_point[1]))
                end_px = (int(end_point[0]), int(end_point[1]))
                
                # Only draw if within frame bounds
                if (0 <= start_px[0] < W and 0 <= start_px[1] < H and
                    0 <= end_px[0] < W and 0 <= end_px[1] < H):
                    cv2.line(frame, start_px, end_px, color, 2)
        
        # Draw landmarks
        for i, point in enumerate(hand_landmarks):
            if np.any(point[:2] != 0):
                px = (int(point[0]), int(point[1]))
                if 0 <= px[0] < W and 0 <= px[1] < H:
                    # Special color for wrist (index 0)
                    pt_color = (255, 0, 0) if i == 0 else color
                    cv2.circle(frame, px, 4, pt_color, -1)
                    cv2.circle(frame, px, 5, (255, 255, 255), 1)
        
        # Add label
        label = "Left Hand" if is_left else "Right Hand"
        wrist = hand_landmarks[0]
        if np.any(wrist[:2] != 0):
            label_pos = (int(wrist[0]), int(wrist[1]) - 15)
            if 0 <= label_pos[0] < W and 0 <= label_pos[1] < H:
                cv2.putText(frame, label, label_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def draw_pose_landmarks(self, frame, pose_landmarks, color=(255, 0, 255)):
        """
        Draw pose landmarks and connections on frame.
        
        Args:
            frame: Image to draw on
            pose_landmarks: (7, 4) array [nose, l_shoulder, r_shoulder, l_elbow, 
                           r_elbow, l_wrist, r_wrist] with x, y, z, visibility
            color: BGR color tuple
        """
        H, W = frame.shape[:2]
        
        # Draw connections
        for connection in self.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            start_point = pose_landmarks[start_idx]
            end_point = pose_landmarks[end_idx]
            
            # Check visibility and validity
            if (start_point[3] > 0.5 and end_point[3] > 0.5 and
                np.any(start_point[:2] != 0) and np.any(end_point[:2] != 0)):
                start_px = (int(start_point[0]), int(start_point[1]))
                end_px = (int(end_point[0]), int(end_point[1]))
                
                if (0 <= start_px[0] < W and 0 <= start_px[1] < H and
                    0 <= end_px[0] < W and 0 <= end_px[1] < H):
                    cv2.line(frame, start_px, end_px, color, 2)

        landmark_names = ['Nose', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 
                         'R_Elbow', 'L_Wrist', 'R_Wrist']
        
        for i, (point, name) in enumerate(zip(pose_landmarks, landmark_names)):
            if point[3] > 0.5 and np.any(point[:2] != 0): 
                px = (int(point[0]), int(point[1]))
                if 0 <= px[0] < W and 0 <= px[1] < H:
                    cv2.circle(frame, px, 5, color, -1)
                    cv2.circle(frame, px, 6, (255, 255, 255), 1)
    
    def visualize_single_frame(self, frame, landmarks, frame_idx, 
                               save_path=None, show_normalized=False):
        """
        Visualize a single frame with landmarks overlaid.
        
        Args:
            frame: (H, W, 3) BGR image (numpy array or RGB values 0-1)
            landmarks: (154,) landmark array
            frame_idx: Frame number
            save_path: Where to save the image
            show_normalized: If True, show normalized landmarks without unnormalization
        """

        if frame.dtype == np.float32 or frame.dtype == np.float64:
            frame = (frame * 255).astype(np.uint8)
        
        if frame.shape[2] == 3 and frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        
        vis_frame = frame.copy()
        H, W = vis_frame.shape[:2]
        
        left_hand = landmarks[:63].reshape(21, 3)
        right_hand = landmarks[63:126].reshape(21, 3)
        pose = landmarks[126:].reshape(7, 4)
        
        if not show_normalized:
            left_hand[:, 0] *= W
            left_hand[:, 1] *= H
            right_hand[:, 0] *= W
            right_hand[:, 1] *= H
            pose[:, 0] *= W
            pose[:, 1] *= H

        self.draw_hand_landmarks(vis_frame, left_hand, color=(0, 255, 0), is_left=True)
        self.draw_hand_landmarks(vis_frame, right_hand, color=(0, 255, 255), is_left=False)
        self.draw_pose_landmarks(vis_frame, pose, color=(255, 0, 255))
        
        info_text = f"Frame {frame_idx}"
        cv2.putText(vis_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        valid_left = np.any(left_hand != 0, axis=1).sum()
        valid_right = np.any(right_hand != 0, axis=1).sum()
        valid_pose = (pose[:, 3] > 0.5).sum()
        
        validity_text = f"L:{valid_left}/21 R:{valid_right}/21 P:{valid_pose}/7"
        cv2.putText(vis_frame, validity_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if save_path:
            cv2.imwrite(str(save_path), vis_frame)
        
        return vis_frame
    
    def save_model_input_visualization(self, frames, landmarks, video_name, 
                                      prediction_info=None):
        """
        Save complete visualization of model input.
        
        Args:
            frames: (T, C, H, W) tensor or (T, H, W, C) numpy array - model input frames
            landmarks: (T, 154) tensor or numpy array - model input landmarks
            video_name: Name of the video being processed
            prediction_info: Optional dict with prediction results
            
        Returns:
            output_dir: Path to directory containing all visualizations
        """

        if torch.is_tensor(frames):
            frames = frames.cpu().numpy()
        if torch.is_tensor(landmarks):
            landmarks = landmarks.cpu().numpy()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_dir = self.output_dir / f"{video_name}_{timestamp}"
        video_dir.mkdir(parents=True, exist_ok=True)
        
        frames_dir = video_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"SAVING MODEL INPUT VISUALIZATION")
        print(f"{'='*70}")
        print(f"Output directory: {video_dir}")
        print(f"Number of frames: {len(frames)}")
        print(f"Landmarks shape: {landmarks.shape}")
        
        T = len(frames)
        if len(frames.shape) == 4:
            if frames.shape[1] == 3: 
                frames = frames.transpose(0, 2, 3, 1) 
        
        print(f"\nSaving {T} frames with landmarks overlaid...")
        for i in range(T):
            frame = frames[i]
            lm = landmarks[i]

            if frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = (frame * 255).astype(np.uint8)
            
            save_path = frames_dir / f"frame_{i:03d}.png"
            self.visualize_single_frame(frame_bgr, lm, i, save_path=save_path)
        
        print(f"✓ Saved {T} individual frames")

        print("\nCreating grid visualization...")
        grid_path = video_dir / "frames_grid.png"
        self.create_grid_visualization(frames, landmarks, grid_path)
        print(f"✓ Saved grid visualization")

        print("\nCreating landmark sequence plot...")
        seq_path = video_dir / "landmark_sequence.png"
        self.plot_landmark_sequence(landmarks, seq_path)
        print(f"✓ Saved landmark sequence plot")
        
        print("\nSaving raw data...")
        np.save(video_dir / "frames.npy", frames)
        np.save(video_dir / "landmarks.npy", landmarks)
        print(f"✓ Saved raw numpy arrays")
        
        metadata = {
            'video_name': video_name,
            'timestamp': timestamp,
            'num_frames': T,
            'frame_shape': list(frames.shape),
            'landmarks_shape': list(landmarks.shape),
            'frame_size': list(frames[0].shape[:2]) if len(frames) > 0 else None,
        }
        
        if prediction_info:
            metadata['prediction'] = prediction_info
        
        with open(video_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"✓ VISUALIZATION COMPLETE")
        print(f"{'='*70}\n")
        
        return video_dir
    
    def create_grid_visualization(self, frames, landmarks, save_path, grid_size=(6, 5)):
        """
        Create a grid of frames with landmarks.
        
        Args:
            frames: (T, H, W, C) array
            landmarks: (T, 154) array
            save_path: Where to save the grid
            grid_size: (rows, cols) for grid
        """
        T = len(frames)
        rows, cols = grid_size
        
        indices = np.linspace(0, T-1, rows*cols, dtype=int)
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
        axes = axes.flatten()
        
        for idx, ax in enumerate(axes):
            if idx < len(indices):
                frame_idx = indices[idx]
                frame = frames[frame_idx]
                lm = landmarks[frame_idx]

                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.shape[2] == 3 else frame
                vis_frame = self.visualize_single_frame(frame_bgr, lm, frame_idx)

                vis_frame_rgb = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
                
                ax.imshow(vis_frame_rgb)
                ax.set_title(f"Frame {frame_idx}", fontsize=8)
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_landmark_sequence(self, landmarks, save_path):
        """
        Plot landmark movement over time.
        
        Args:
            landmarks: (T, 154) array
            save_path: Where to save the plot
        """
        T = landmarks.shape[0]
        
        left_hand = landmarks[:, :63].reshape(T, 21, 3)
        right_hand = landmarks[:, 63:126].reshape(T, 21, 3)
        pose = landmarks[:, 126:].reshape(T, 7, 4)
        
        valid_left = np.any(left_hand != 0, axis=2).sum(axis=1)
        valid_right = np.any(right_hand != 0, axis=2).sum(axis=1)
        valid_pose = (pose[:, :, 3] > 0.5).sum(axis=1)
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        
        axes[0].plot(valid_left, label='Left Hand', color='green', marker='o')
        axes[0].plot(valid_right, label='Right Hand', color='cyan', marker='s')
        axes[0].plot(valid_pose, label='Pose', color='magenta', marker='^')
        axes[0].set_ylabel('Valid Landmarks')
        axes[0].set_xlabel('Frame')
        axes[0].set_title('Landmark Validity Over Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        left_wrist_x = left_hand[:, 0, 0]
        right_wrist_x = right_hand[:, 0, 0]
        
        axes[1].plot(left_wrist_x, label='Left Wrist X', color='green')
        axes[1].plot(right_wrist_x, label='Right Wrist X', color='cyan')
        axes[1].set_ylabel('X Coordinate (normalized)')
        axes[1].set_xlabel('Frame')
        axes[1].set_title('Wrist X-Position Over Time')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        left_wrist_y = left_hand[:, 0, 1]
        right_wrist_y = right_hand[:, 0, 1]
        
        axes[2].plot(left_wrist_y, label='Left Wrist Y', color='green')
        axes[2].plot(right_wrist_y, label='Right Wrist Y', color='cyan')
        axes[2].set_ylabel('Y Coordinate (normalized)')
        axes[2].set_xlabel('Frame')
        axes[2].set_title('Wrist Y-Position Over Time')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

def add_visualization_to_inference(inference_instance, output_dir="model_input_visualizations"):
    """
    Add visualization capabilities to an existing HierarchicalInference instance.
    
    Args:
        inference_instance: Instance of HierarchicalInference
        output_dir: Directory to save visualizations
    
    Returns:
        The same instance with visualization enabled
    """
    import types
    original_predict = inference_instance.predict
    
    inference_instance.visualizer = ModelInputVisualizer(
        output_dir=output_dir,
        config=inference_instance.config
    )
    
    def predict_with_visualization(self, video_path, top_k=5, save_visualization=True):
        """
        Enhanced predict function that saves visualizations.
        """
        if save_visualization:
            frames, landmarks = self.preprocess_video(video_path)
            frames_tensor, landmarks_tensor = self.prepare_model_input(frames, landmarks)
            
            result = original_predict(video_path, top_k)
            
            video_name = Path(video_path).stem
            viz_dir = self.visualizer.save_model_input_visualization(
                frames=frames_tensor,
                landmarks=landmarks_tensor,
                video_name=video_name,
                prediction_info=result
            )
            
            result['visualization_dir'] = str(viz_dir)
        else:
            result = original_predict(video_path, top_k)
        
        return result
    
    inference_instance.predict = types.MethodType(predict_with_visualization, inference_instance)
    
    return inference_instance

if __name__ == "__main__":
    """
    Example usage of the visualization tool.
    """
    
    print("Example 1: Standalone Visualization")
    print("=" * 70)
    
    visualizer = ModelInputVisualizer(output_dir="test_visualizations")

    T, H, W = 30, 224, 224
    dummy_frames = np.random.rand(T, 3, H, W).astype(np.float32)
    dummy_landmarks = np.random.rand(T, 154).astype(np.float32)

    dummy_landmarks[:, :21] = 0 
    
    viz_dir = visualizer.save_model_input_visualization(
        frames=dummy_frames,
        landmarks=dummy_landmarks,
        video_name="test_video",
        prediction_info={'predicted_class': 'test', 'confidence': 0.95}
    )
    
    print(f"\nVisualization saved to: {viz_dir}")

    print("\n\nExample 2: Integration with HierarchicalInference")
    print("=" * 70)
    print("""
To integrate with your existing code:

from hierarchical_inference import HierarchicalInference
from model_input_visualizer import add_visualization_to_inference

# Create inference object
dynamic_infer = HierarchicalInference(
    gating_model_path='path/to/gating_model.pth',
    specialist_paths=['path/to/specialist1.pth', 'path/to/specialist2.pth'],
    device='cuda'
)

# Add visualization capability (patches the instance, not the class)
dynamic_infer = add_visualization_to_inference(dynamic_infer)

# Now predictions will automatically save visualizations
result = dynamic_infer.predict('path/to/video.mp4', top_k=3, save_visualization=True)

# Visualization will be saved to result['visualization_dir']
print(f"Visualization saved to: {result['visualization_dir']}")

# You can also disable visualization for specific predictions
result = dynamic_infer.predict('path/to/video.mp4', top_k=3, save_visualization=False)
    """)