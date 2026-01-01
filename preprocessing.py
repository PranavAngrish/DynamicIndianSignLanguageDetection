
global_pose = None
global_hands = None
global_seg = None



import cv2
import numpy as np
import mediapipe as mp
import json
from pathlib import Path
from tqdm import tqdm
import warnings
import gc
import random
import time
import sys
import os
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from multiprocessing import Pool, cpu_count, Manager
import psutil
import threading
from contextlib import contextmanager
from configs.config import Config
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'

@contextmanager
def suppress_stderr():
    """Context manager to suppress stderr output"""
    original_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stderr.close()
        sys.stderr = original_stderr

cv2.setNumThreads(4)


class SystemMonitor:
    """Monitor CPU, memory, and detect potential issues"""
    
    def __init__(self, check_interval=5):
        self.check_interval = check_interval
        self.monitoring = False
        self.monitor_thread = None
        self.stats = {
            'cpu_percent': [],
            'memory_percent': [],
            'memory_available_gb': [],
            'timestamps': [],
            'warnings': []
        }
        self.lock = threading.Lock()
    
    def start(self):
        """Start monitoring in background thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("✓ System monitoring started")
    
    def stop(self):
        """Stop monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                with self.lock:
                    self.stats['cpu_percent'].append(cpu_percent)
                    self.stats['memory_percent'].append(memory.percent)
                    self.stats['memory_available_gb'].append(memory.available / (1024**3))
                    self.stats['timestamps'].append(time.time())
                    
                    if memory.percent > 90:
                        warning = f"⚠️  HIGH MEMORY: {memory.percent:.1f}% used"
                        if warning not in self.stats['warnings']:
                            self.stats['warnings'].append(warning)
                            print(f"\n{warning}")
                    
                    if cpu_percent < 20:
                        warning = "⚠️  LOW CPU UTILIZATION: Possible I/O bottleneck"
                        if warning not in self.stats['warnings'] and len(self.stats['cpu_percent']) > 10:
                            self.stats['warnings'].append(warning)
                            print(f"\n{warning}")
                
                time.sleep(self.check_interval)
            except Exception:
                pass
    
    def get_summary(self):
        """Get monitoring summary"""
        with self.lock:
            if not self.stats['cpu_percent']:
                return "No monitoring data collected"
            
            cpu_avg = np.mean(self.stats['cpu_percent'])
            cpu_max = np.max(self.stats['cpu_percent'])
            mem_avg = np.mean(self.stats['memory_percent'])
            mem_max = np.max(self.stats['memory_percent'])
            mem_min = np.min(self.stats['memory_available_gb'])
            
            summary = f"""
╔════════════════════════════════════════════════════════════════╗
║                    SYSTEM MONITORING SUMMARY                    ║
╠════════════════════════════════════════════════════════════════╣
║ CPU: Avg {cpu_avg:5.1f}% | Peak {cpu_max:5.1f}%                           ║
║ Memory: Avg {mem_avg:5.1f}% | Peak {mem_max:5.1f}% | Min Avail {mem_min:4.1f}GB   ║
╚════════════════════════════════════════════════════════════════╝"""
            return summary


class HandSignDatasetPreprocessor:
    """Enhanced parallel preprocessor with system monitoring"""

    def __init__(self, config, input_dir, output_dir, augment=True, verbose=True):
        self.input_dir = Path(input_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.augment = augment
        self.config = config
        self.verbose = verbose

        self.npy_dir = self.output_dir / "npy_files"
        self.npy_dir.mkdir(parents=True, exist_ok=True)

        H, W = self.config.FRAME_SIZE 
        self.frame_size = getattr(config, 'FRAME_SIZE', (W, H))  # (H, W)
        
        self.checkpoint_file = self.output_dir / "processing_checkpoint.json"
        self.processed_videos = self.load_checkpoint()

        self.monitor = SystemMonitor(check_interval=3)
        self.setup_augmentation_pipeline()

        if self.verbose:
            print(f"✓ Initialized PARALLEL Preprocessor")
            print(f"  CPUs: {cpu_count()} | Memory: {psutil.virtual_memory().available/(1024**3):.1f}GB")

    def setup_augmentation_pipeline(self):
        self.combined_augs = [
            {
                'name': 'background_blur_lighting',
                'params': {
                    'blur_background_prob': 1.0,
                    'stretch_horizontal': True,
                    'stretch_factor': (1.15, 1.25),
                    'brightness_range': (0.05, 0.20),
                    'contrast': (1.05, 1.20),
                    'hue_shift': (-5, 5),
                    'sat_shift': (-5, 10),
                    'value_shift': (-5, 8),
                    'rotate_prob': 0.5,
                    'rotate_range': (-10, 10)
                }
            },
            {
                'name': 'color_tone_stretch',
                'params': {
                    'stretch_vertical': True,
                    'stretch_factor': (1.10, 1.20),
                    'brightness_range': (0.05, 0.15),
                    'contrast': (1.00, 1.10),
                    'hue_shift': (-10, 10),
                    'sat_shift': (10, 25),
                    'value_shift': (-5, 10),
                    'rotate_prob': 0.5,
                    'rotate_range': (-10, 10)
                }
            }
        ]

    def load_checkpoint(self):
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    return set(json.load(f))
            except:
                return set()
        return set()

    def save_checkpoint(self, video_id):
        self.processed_videos.add(video_id)
        with open(self.checkpoint_file, 'w') as f:
            json.dump(list(self.processed_videos), f)

    def is_processed(self, video_id):
        return video_id in self.processed_videos



    def load_video(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames, fps


    def apply_background_blur(self, frame, segmentation, strength=51):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = segmentation.process(rgb)
        if result.segmentation_mask is None:
            return frame
        mask = result.segmentation_mask
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        blurred = cv2.GaussianBlur(frame, (strength, strength), 0)
        return (mask[..., None] * frame + (1 - mask[..., None]) * blurred).astype(np.uint8)

    def apply_color_shift(self, frame, hue=0, sat=0, val=0):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + sat, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + val, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def sample_augmentation_params(self, aug_config):
        p = aug_config['params']
        sampled = {}
        if 'brightness_range' in p:
            sampled['brightness'] = random.uniform(*p['brightness_range'])
        if 'contrast' in p:
            sampled['contrast'] = random.uniform(*p['contrast'])
        if p.get('stretch_horizontal'):
            sampled['apply_horizontal_stretch'] = True
            sampled['stretch_factor'] = random.uniform(*p['stretch_factor'])
        if p.get('stretch_vertical'):
            sampled['apply_vertical_stretch'] = True
            sampled['stretch_factor'] = random.uniform(*p['stretch_factor'])
        if 'rotate_prob' in p:
            if random.random() < p['rotate_prob']:
                sampled['apply_rotation'] = True
                sampled['rotation_angle'] = random.uniform(*p['rotate_range'])
        sampled['blur_background'] = p.get('blur_background_prob', 0) >= 1.0
        sampled['hue_shift'] = random.uniform(*p.get('hue_shift', (0, 0)))
        sampled['sat_shift'] = random.uniform(*p.get('sat_shift', (0, 0)))
        sampled['val_shift'] = random.uniform(*p.get('value_shift', (0, 0)))
        return sampled

    def apply_sampled_augmentation_to_frame(self, frame, sampled, segmentation):
        out = frame.copy()
        if sampled.get('apply_horizontal_stretch'):
            h, w = out.shape[:2]
            new_w = int(w * sampled.get('stretch_factor', 1.0))
            stretched = cv2.resize(out, (new_w, h), interpolation=cv2.INTER_LINEAR)
            start_x = (new_w - w) // 2
            out = stretched[:, start_x:start_x + w]
        if sampled.get('apply_vertical_stretch'):
            h, w = out.shape[:2]
            new_h = int(h * sampled.get('stretch_factor', 1.0))
            stretched = cv2.resize(out, (w, new_h), interpolation=cv2.INTER_LINEAR)
            start_y = (new_h - h) // 2
            out = stretched[start_y:start_y + h, :]
        if sampled.get('blur_background'):
            out = self.apply_background_blur(out, segmentation, 51)
        if any(k in sampled for k in ['hue_shift', 'sat_shift', 'val_shift']):
            out = self.apply_color_shift(out, int(sampled.get('hue_shift', 0)),
                                        int(sampled.get('sat_shift', 0)),
                                        int(sampled.get('val_shift', 0)))
        if sampled.get('apply_rotation'):
            h, w = out.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), sampled.get('rotation_angle', 0), 1.0)
            out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        if 'brightness' in sampled:
            out = np.clip(out.astype(np.float32) + sampled['brightness'] * 255, 0, 255).astype(np.uint8)
        if 'contrast' in sampled:
            out = np.clip(out.astype(np.float32) * sampled['contrast'], 0, 255).astype(np.uint8)
        return out

    def create_augmented_version(self, frames, aug_config, segmentation):
        sampled = self.sample_augmentation_params(aug_config)
        return [self.apply_sampled_augmentation_to_frame(f, sampled, segmentation) for f in frames]

    def augment_video_in_memory(self, video_path, class_name, segmentation):
        frames, fps = self.load_video(video_path)
        if not frames:
            return []
        video_refs = [{'frames': frames, 'name': f"{video_path.stem}_original", 'fps': fps, 'in_memory': True}]
        for aug in self.combined_augs:
            try:
                aug_frames = self.create_augmented_version(frames, aug, segmentation)
                video_refs.append({'frames': aug_frames, 'name': f"{video_path.stem}_{aug['name']}", 
                                 'fps': fps, 'in_memory': True})
            except:
                pass
        return video_refs

    def extract_landmarks(self, frame, pose, hands):
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_res = hands.process(rgb)
            pose_res = pose.process(rgb)
            left_hand = np.zeros(63)
            right_hand = np.zeros(63)
            pose_lm = np.zeros(28)
            if pose_res.pose_landmarks:
                vals = []
                for i in [0, 11, 12, 13, 14, 15, 16]:
                    lm = pose_res.pose_landmarks.landmark[i]
                    vals.extend([lm.x, lm.y, lm.z, lm.visibility])
                pose_lm = np.array(vals)
                nose_x = pose_res.pose_landmarks.landmark[0].x
                if hand_res.multi_hand_landmarks:
                    for h in hand_res.multi_hand_landmarks:
                        data = np.array([[l.x, l.y, l.z] for l in h.landmark]).flatten()
                        if h.landmark[0].x < nose_x:
                            right_hand = data
                        else:
                            left_hand = data
            elif hand_res.multi_hand_landmarks:
                for h, hd in zip(hand_res.multi_hand_landmarks, hand_res.multi_handedness):
                    data = np.array([[l.x, l.y, l.z] for l in h.landmark]).flatten()
                    if hd.classification[0].label == 'Left':
                        left_hand = data
                    else:
                        right_hand = data
            return np.concatenate([left_hand, right_hand, pose_lm])
        except:
            return np.zeros(154)

    def calculate_frame_quality(self, frame, landmarks):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        visibility = np.count_nonzero(landmarks) / len(landmarks)
        return 0.7 * sharpness + 0.3 * visibility * 1000

    def interpolate_landmarks(self, landmarks_seq):
        arr = np.array(landmarks_seq)
        T, F = arr.shape
        out = arr.copy()
        for i in range(F):
            seq = arr[:, i]
            valid = np.where(seq != 0)[0]
            if len(valid) >= 4:
                kind = 'cubic'
            elif len(valid) >= 2:
                kind = 'linear'
            elif len(valid) == 1:
                out[:, i] = seq[valid[0]]
                continue
            else:
                continue
            func = interp1d(valid, seq[valid], kind=kind, bounds_error=False,
                          fill_value=(seq[valid[0]], seq[valid[-1]]))
            out[:, i] = func(np.arange(T))
        out[:, :126] = np.clip(out[:, :126], 0.0, 1.0)
        return out

    def normalize_landmarks(self, landmarks):
        landmarks = landmarks.copy()
        T = landmarks.shape[0]
        hand1 = landmarks[:, :63].reshape(T, 21, 3)
        hand2 = landmarks[:, 63:126].reshape(T, 21, 3)
        pose = landmarks[:, 126:].reshape(T, 7, 4)
        all_xyz = np.concatenate([hand1.reshape(T*21, 3), hand2.reshape(T*21, 3),
                                 pose[:, :, :3].reshape(T*7, 3)], axis=0)
        valid = np.any(all_xyz != 0, axis=1)
        if np.any(valid):
            coords = all_xyz[valid]
            all_xyz[:, :2] -= np.mean(coords[:, :2], axis=0)
            maxd = np.max(np.linalg.norm(coords[:, :2], axis=1))
            if maxd > 0:
                all_xyz[:, :2] /= maxd
            zc = coords[:, 2]
            if len(zc) > 1:
                zs = np.std(zc)
                if zs > 0:
                    all_xyz[:, 2] = (all_xyz[:, 2] - np.mean(zc)) / zs
        hand1 = all_xyz[:T*21].reshape(T, 21, 3)
        hand2 = all_xyz[T*21:T*42].reshape(T, 21, 3)
        pose_xyz = all_xyz[T*42:].reshape(T, 7, 3)
        normalized = np.zeros((T, 154))
        for t in range(T):
            normalized[t] = np.concatenate([hand1[t].flatten(), hand2[t].flatten(),
                                          pose_xyz[t].flatten(), pose[t, :, 3]])
        return normalized



    def preprocess_video(self, video_ref, pose, hands, target_count):
        if isinstance(video_ref, dict) and video_ref.get('in_memory'):
            frames, fps = video_ref['frames'], video_ref['fps']
        else:
            frames, fps = self.load_video(video_ref)
        if not frames:
            return [], np.zeros((target_count, 154))
        duration = len(frames) / fps
        landmarks, qualities, timestamps = [], [], []

        for idx, f in enumerate(frames):
            lm = self.extract_landmarks(f, pose, hands)
            landmarks.append(lm)
            qualities.append(self.calculate_frame_quality(f, lm))
            timestamps.append(idx / fps)
        # frames = np.array(frames, dtype=object)
        landmarks = np.array(landmarks)
        qualities = np.array(qualities)
        timestamps = np.array(timestamps)
        edges = np.linspace(0, duration, target_count + 1)
        sel_idx = []
        for i in range(target_count):
            mask = (timestamps >= edges[i]) & (timestamps < edges[i+1])
            seg_q = qualities[mask]
            if len(seg_q) > 0:
                sel_idx.append(np.where(mask)[0][np.argmax(seg_q)])
            else:
                sel_idx.append(np.argmin(np.abs(timestamps - (edges[i] + edges[i+1])/2)))
        sel_idx = sorted(set(sel_idx))
        sel_frames = [frames[i] for i in sel_idx]
        target_w, target_h = self.frame_size[0], self.frame_size[1]
        sel_frames_resized = [
            cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_AREA)
            for f in sel_frames
        ]
        sel_landmarks = np.array([landmarks[i] for i in sel_idx])
        interpolated = self.interpolate_landmarks(sel_landmarks)
        normalized = self.normalize_landmarks(interpolated)
        return sel_frames_resized, normalized

    def get_next_video_index(self, split, class_name, counter_dict):
        key = f"{split}_{class_name}"
        if key not in counter_dict:
            counter_dict[key] = 0
        counter_dict[key] += 1
        return counter_dict[key]

    def save_sample_immediately(self, sample, split, class_name, counter_dict):
        video_idx = self.get_next_video_index(split, class_name, counter_dict)
        class_dir = self.npy_dir / split / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        video_folder = class_dir / f"video_{video_idx}"
        video_folder.mkdir(exist_ok=True)
        np.save(video_folder / "frames.npy", sample['frames'])
        np.save(video_folder / "landmarks.npy", sample['pose_sequence'])
        metadata = {
            'video_name': sample['video_name'],
            'video_folder': f"video_{video_idx}",
            'class': class_name,
            'frames_path': str((video_folder / "frames.npy").relative_to(self.output_dir)),
            'landmarks_path': str((video_folder / "landmarks.npy").relative_to(self.output_dir)),
            'num_frames': sample['num_frames']
        }
        del sample['frames'], sample['pose_sequence']
        gc.collect()
        return metadata

    def collect_videos_from_structure(self):
        split_videos = {'train': {}, 'validation': {}}
        for class_dir in self.input_dir.iterdir():
            if not class_dir.is_dir() or class_dir.name.startswith('.'):
                continue
            class_name = class_dir.name
            for split in ['train', 'validation']:
                split_dir = class_dir / split
                videos = []
                if split_dir.exists():
                    for ext in ['*.mp4', '*.avi', '*.MOV', '*.mov']:
                        videos.extend(list(split_dir.glob(ext)))
                split_videos[split][class_name] = videos
        return split_videos

    def process_dataset(self, num_workers=None):
        print("We are going to preprocess data(%^&^%*^(*^%(^$^*(")
        if num_workers is None:
            num_workers = max(1, cpu_count() - 2)
        print(f"\n{'='*70}\nPARALLEL PREPROCESSING WITH {num_workers} WORKERS\n{'='*70}")
        self.monitor.start()
        start_time = time.time()
        split_videos = self.collect_videos_from_structure()
        split_videos['train'] = {k: v for k, v in split_videos['train'].items() if v}
        split_videos['validation'] = {k: v for k, v in split_videos['validation'].items() if v}
        all_classes = set(split_videos['train'].keys()) | set(split_videos['validation'].keys())
        print("\nDataset Structure:")
        for split in ['train', 'validation']:
            total = sum(len(v) for v in split_videos[split].values())
            print(f"  {split}: {total} videos across {len(split_videos[split])} classes")
        split_metadata = {'train': [], 'validation': []}
        manager = Manager()
        counter_dict = manager.dict()
        for split_name in ['train', 'validation']:
            print(f"\n{'='*70}\nProcessing {split_name.upper()} split\n{'='*70}")
            tasks = []
            for class_name, videos in split_videos[split_name].items():
                for video_path in videos:
                    video_id = f"{split_name}_{class_name}_{video_path.name}"
                    if not self.is_processed(video_id):
                        tasks.append((video_path, class_name, split_name, self.config,
                                    self.augment, self.output_dir, self.npy_dir,
                                    counter_dict, self.combined_augs))
            if not tasks:
                print(f"No videos to process for {split_name}")
                continue
            print(f"Processing {len(tasks)} videos...")
            with Pool(processes=num_workers, initializer=init_worker) as pool:
                results = list(tqdm(pool.imap_unordered(process_video_worker, tasks),
                                  total=len(tasks), desc=f"  {split_name}"))
            for success, metadata_list, video_id in results:
                if success:
                    split_metadata[split_name].extend(metadata_list)
                    self.save_checkpoint(video_id)
            print(f"✓ {split_name}: {len(split_metadata[split_name])} samples")
        self.monitor.stop()
        print(f"\n{'='*70}\nSaving metadata...")
        for split_name in ['train', 'validation']:
            metadata_path = self.npy_dir / split_name / f"{split_name}_metadata.json"
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metadata_path, 'w') as f:
                json.dump(split_metadata[split_name], f, indent=2)
            print(f"✓ {split_name}: {len(split_metadata[split_name])} samples saved")
        metadata = {
            'num_classes': len(all_classes),
            'class_names': sorted(list(all_classes)),
            'train_samples': len(split_metadata['train']),
            'val_samples': len(split_metadata['validation']),
            'processing_time_minutes': (time.time() - start_time) / 60
        }
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        elapsed = time.time() - start_time
        print(f"\n{'='*70}\n✓ COMPLETE!\nTime: {elapsed/60:.1f} min")
        print(f"Training: {len(split_metadata['train'])} | Validation: {len(split_metadata['validation'])}")
        print(self.monitor.get_summary())
        return split_metadata

    def cleanup(self):
        self.monitor.stop()




def init_worker():
    global global_pose, global_hands, global_seg
    with suppress_stderr():
        mp_pose = mp.solutions.pose
        global_pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                                   min_detection_confidence=0.5, min_tracking_confidence=0.5)
        mp_hands = mp.solutions.hands
        global_hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                                      min_detection_confidence=0.5, min_tracking_confidence=0.5,
                                      model_complexity=1)
        mp_seg = mp.solutions.selfie_segmentation
        global_seg = mp_seg.SelfieSegmentation(model_selection=1)




def process_video_worker(args):
    """Worker function for multiprocessing"""
    (video_path, class_name, split_name, config, augment,
     output_dir, npy_dir, counter_dict, combined_augs) = args
    
    
    temp = HandSignDatasetPreprocessor.__new__(HandSignDatasetPreprocessor)
    temp.config = config
    temp.augment = augment
    temp.output_dir = output_dir
    temp.npy_dir = npy_dir
    temp.combined_augs = combined_augs
    temp.verbose = False

    H, W = config.FRAME_SIZE 
    temp.frame_size = (W, H)  
    
    
    video_id = f"{split_name}_{class_name}_{video_path.name}"
    
    global global_pose, global_hands, global_seg
    pose = global_pose
    hands = global_hands
    segmentation = global_seg
    
    try:
        metadata_list = []
        if split_name == 'train' and augment:
            video_refs = temp.augment_video_in_memory(video_path, class_name, segmentation)
        else:
            video_refs = [video_path]
        for video_ref in video_refs:
            vid_name = video_ref['name'] if isinstance(video_ref, dict) else video_ref.stem
            frames, landmarks = temp.preprocess_video(video_ref, pose, hands, config.TARGET_COUNT)
 
    
            if np.sum(np.any(landmarks != 0, axis=1)) > 5:
                sample = {
                    'video_name': vid_name,
                    'class': class_name,
                    'frames': frames,
                    'pose_sequence': landmarks,
                    'num_frames': len(landmarks)
                }
                metadata = temp.save_sample_immediately(sample, split_name, class_name, counter_dict)
                metadata_list.append(metadata)
            del frames, landmarks
            gc.collect()
        

        return (True, metadata_list, video_id)
    except Exception as e:
        print("Exception occurred:", e)

        return (False, [], video_id)


if __name__ == '__main__':
    
    
    #INPUT_DIR = '/islData/Data/Pronouns'
    #OUTPUT_DIR = '/islData/datasets_parallel'
    INPUT_DIR = '/workspace/ISL-Pronouns/data_10_2/'
    OUTPUT_DIR = '/workspace/ISL-Pronouns/datasets_parallel_data_10_2'
    
    config = Config()
    preprocessor = HandSignDatasetPreprocessor(
        config=config,
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        augment=True,
        verbose=True
    )
    
    split_metadata = preprocessor.process_dataset(num_workers=2)
    preprocessor.cleanup()
    print('\nDone!')