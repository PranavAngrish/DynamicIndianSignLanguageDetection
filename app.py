from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow import keras
import numpy as np
import tempfile, os, traceback, sys
from infer import SignLanguageInference
from inference_hierarchical import HierarchicalInference
from model_input_visualizer import add_visualization_to_inference
from sentence_model import ISLConverter   
from translation_model import translate_text  
from translation_model import IndianLanguageTranslator
from werkzeug.utils import secure_filename
import time
import torch
import subprocess 
import shutil

import string, copy, itertools, traceback, sys, os
import joblib  
from collections import deque, Counter
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

@app.before_request
def handle_options():
    if request.method == "OPTIONS":
        response = jsonify({})
        response.status_code = 200
        response.headers["Access-Control-Allow-Origin"] = request.headers.get("Origin", "*")
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
        return response


static_model = keras.models.load_model("isl_static_model.h5")
STATIC_LABELS = ['1','2','3','4','5','6','7','8','9'] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
FRAME_HISTORY_SIZE = 5 
history_queue = deque(maxlen=FRAME_HISTORY_SIZE)
script_dir = os.path.dirname(os.path.abspath(__file__))

model1_path = os.path.join(script_dir, "isl_static_model.h5")
model1 = None
try:
    print(f"Loading Model 1 (NN) from: {model1_path}...", file=sys.stdout)
    model1 = keras.models.load_model(model1_path)
    print("✅ Model 1 loaded!", file=sys.stdout)
except Exception as e:
    print(f"❌ Error loading Model 1: {e}", file=sys.stderr)

model2_path = os.path.join(script_dir, "isl_rf_model.pkl")
model2 = None
is_sklearn_model = False

try:
    if os.path.exists(model2_path):
        print(f"Loading Model 2 (Verifier) from: {model2_path}...", file=sys.stdout)
        if model2_path.endswith('.h5') or model2_path.endswith('.keras'):
            model2 = keras.models.load_model(model2_path)
            is_sklearn_model = False
        else:
            model2 = joblib.load(model2_path)
            is_sklearn_model = True
        print("✅ Model 2 loaded!", file=sys.stdout)
    else:
        print("ℹ️ No Model 2 found. Running in Single Model Mode.", file=sys.stdout)
except Exception as e:
    print(f"⚠️ Error loading Model 2: {e}", file=sys.stderr)


alphabet = ['1','2','3','4','5','6','7','8','9'] + list(string.ascii_uppercase)

def pre_process_landmark_fix(landmarks_data):
    """ Your existing preprocessing logic """
    temp = copy.deepcopy(landmarks_data)
    if not temp or len(temp) == 0: return [0.0] * 42
    
    for lm in temp:
        lm['x'] = float(lm.get('x', 0.0))
        lm['y'] = float(lm.get('y', 0.0))

    base_x, base_y = temp[0]['x'], temp[0]['y']
    for lm in temp:
        lm['x'] -= base_x
        lm['y'] -= base_y

    flat = list(itertools.chain.from_iterable([[lm['x'], lm['y']] for lm in temp]))

    max_val = max(map(abs, flat)) if len(flat) > 0 else 1.0
    if max_val == 0: max_val = 1.0
    normalized = [float(n) / float(max_val) for n in flat]

    expected_len = 42
    if len(normalized) < expected_len:
        normalized += [0.0] * (expected_len - len(normalized))
    elif len(normalized) > expected_len:
        normalized = normalized[:expected_len]

    return normalized

def get_prediction_model1(input_array):
    """ Returns (label, confidence_score) from Keras NN """
    if model1 is None: return None, 0.0
    
    preds = model1.predict(input_array, verbose=0)
    probs = preds[0] if preds.ndim == 2 else preds
    
    best_idx = int(np.argmax(probs))
    confidence = float(probs[best_idx])
    label = alphabet[best_idx] if best_idx < len(alphabet) else str(best_idx)
    return label, confidence

def get_prediction_model2(input_array):
    """ Returns (label, probability) from Second Model """
    if model2 is None: return None, 0.0

    if is_sklearn_model:
        try:
            probs = model2.predict_proba(input_array)[0]
            best_idx = int(np.argmax(probs))
            confidence = float(probs[best_idx])
            label = alphabet[best_idx] if best_idx < len(alphabet) else str(best_idx)
            return label, confidence
        except:
            pred = model2.predict(input_array)[0]
      
            return str(pred), 1.0 
    else:
  
        preds = model2.predict(input_array, verbose=0)
        probs = preds[0]
        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])
        label = alphabet[best_idx] if best_idx < len(alphabet) else str(best_idx)
        return label, confidence


GATING_MODEL_PATH = 'checkpoints/gating_model/best_model.pth'
SPECIALIST_PATHS = [
    'checkpoints/group_0/best_model.pth',
    'checkpoints/group_1/best_model.pth',
    'checkpoints/group_2/best_model.pth'
]


if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

print(f"Inference running on: {device}")
dynamic_infer = HierarchicalInference(
    gating_model_path=GATING_MODEL_PATH,
    specialist_paths=SPECIALIST_PATHS,
    device=device
)


print(f"Inference running on: {device}")


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        'status': 'healthy', 
        'model1': model1 is not None,
        'model2': model2 is not None
    })

def is_finger_extended(landmarks, finger_name):
    """
    Returns True if a finger is pointing up/out, False if curled.
    Landmarks indices: 
    Thumb=4, Index=8, Middle=12, Ring=16, Pinky=20
    Knuckles: Thumb=2, Index=5, Middle=9, Ring=13, Pinky=17
    """

    indices = {
        'thumb': (4, 2), 
        'index': (8, 5),
        'middle': (12, 9),
        'ring': (16, 13),
        'pinky': (20, 17)
    }
    tip_idx, joint_idx = indices[finger_name]

    tip = landmarks[tip_idx]
    wrist = landmarks[0]

    def get_dist(p1, p2):
        return ((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)**0.5

    dist_tip = get_dist(tip, wrist)

    joint = landmarks[joint_idx]
    dist_joint = get_dist(joint, wrist)

    return dist_tip > (dist_joint * 1.2) 

def apply_geometric_corrections(prediction, landmarks):
    """
    Overrules the model if the geometry makes no sense for that letter.
    """

    middle_up = is_finger_extended(landmarks, 'middle')
    ring_up = is_finger_extended(landmarks, 'ring')
    pinky_up = is_finger_extended(landmarks, 'pinky')
    index_up = is_finger_extended(landmarks, 'index')
    

    if prediction == 'E' or prediction == 'S':
        if middle_up and ring_up and pinky_up:
            print("LOGIC OVERRIDE: E/S -> F (Fingers are open)", file=sys.stdout)
            return 'F'

    if index_up and not middle_up and not ring_up and not pinky_up:
        if prediction in ['P', 'W', 'F']:
            print("LOGIC OVERRIDE: -> D (Only index up)", file=sys.stdout)
            return 'D'

    thumb_up = is_finger_extended(landmarks, 'thumb')
    if thumb_up and pinky_up and not index_up and not middle_up and not ring_up:
        if prediction in ['J', 'I']:
             print("LOGIC OVERRIDE: -> Y (Hang loose shape)", file=sys.stdout)
             return 'Y'

    return prediction


@app.route("/", methods=["GET"])
def index():
    return "<h1>ISL Backend Running 🚀</h1><p>Use endpoints: /health, /predict/static, /predict/dynamic</p>"

@app.route("/predict/static", methods=["POST"])
def predict_static():
    global history_queue
    try:
        data = request.get_json(force=True)
        if not data: return jsonify({'error': 'No Data', 'success': False}), 400


        landmarks = None
        if 'hands' in data and isinstance(data['hands'], list) and len(data['hands']) > 0:
            if 'landmarks' in data['hands'][0]: landmarks = data['hands'][0]['landmarks']
        elif 'landmarks' in data:
            landmarks = data['landmarks']

        if not landmarks:
            history_queue.clear()
            return jsonify({'prediction': '-', 'confidence': 0.0, 'success': True})

        vec = pre_process_landmark_fix(landmarks)
        input_array = np.asarray([vec], dtype=np.float32)


        l1, c1 = get_prediction_model1(input_array)
        l2, c2 = get_prediction_model2(input_array)


        current_prediction = l1
        current_confidence = c1
        used_model = "NN"


        if model2 is not None:

            if c2 > (c1 + 0.02): 
                current_prediction = l2
                current_confidence = c2
                used_model = "RF"

        print(f"DEBUG: NN={l1}({c1:.2f}) | RF={l2}({c2:.2f}) -> Winner: {used_model} ({current_prediction})", file=sys.stdout)
        

        if current_prediction and landmarks:
            original = current_prediction

            current_prediction = apply_geometric_corrections(current_prediction, landmarks)
            

            if current_prediction != original:
                current_confidence = 0.99


        if current_confidence < 0.50:
            current_prediction = None


        if current_prediction:
            history_queue.append(current_prediction)

        if len(history_queue) >= 3:
            most_common = Counter(history_queue).most_common(1)
            best_history_label, count = most_common[0]

            if count >= (len(history_queue) / 2):
                final_label = best_history_label
                final_conf = current_confidence
            else:
                final_label = "-"
                final_conf = 0.0
        else:
            final_label = "-"
            final_conf = 0.0

        return jsonify({'prediction': final_label, 'confidence': final_conf, 'success': True})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


UPLOAD_FOLDER = "debug_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)



@app.route("/predict/dynamic", methods=["POST"])
def predict_dynamic():
    try:
        if "video" not in request.files:
            return jsonify({"success": False, "error": "No video uploaded"}), 400

        video_file = request.files["video"]
        
        if video_file.filename == '':
            return jsonify({"success": False, "error": "No selected file"}), 400

        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_webm:
            video_file.save(temp_webm.name)
            webm_path = temp_webm.name
            

        mp4_path = webm_path.replace(".webm", ".mp4")
        
        print(f"Processing video at temp path: {webm_path}")


        final_video_path = webm_path 

        if shutil.which("ffmpeg"): 
            try:
                subprocess.run([
                    'ffmpeg', '-y',          
                    '-i', webm_path,         
                    '-c:v', 'libx264',       
                    '-preset', 'fast',       
                    '-crf', '23',            
                    '-pix_fmt', 'yuv420p',
                    '-c:a', 'aac',           
                    mp4_path                 
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                final_video_path = mp4_path
                
            except subprocess.CalledProcessError as e:
                print(f"FFmpeg conversion failed: {e}")
        else:
            print("FFmpeg not found. Skipping conversion.")

        result = dynamic_infer.predict(final_video_path, top_k=3)

        try:
            os.remove(webm_path)
            if os.path.exists(mp4_path):
                os.remove(mp4_path)
        except Exception as cleanup_error:
            print(f"Warning: Could not delete temp files: {cleanup_error}")

        return jsonify({
            "success": True,
            "prediction": result["top_prediction"]["class"],
            "confidence": result["top_prediction"]["overall_confidence"], 
            "top_k": result["all_predictions"]
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500



@app.route("/process_translate", methods=["POST"])
def process_and_translate():
    try:
        data = request.get_json(force=True)

        raw_sentence = data["sentence"]
        target_lang = data["language"]
        print(raw_sentence)
        print(target_lang)

        converter = ISLConverter()

        time.sleep(5)
        formed_sentence = converter.convert_to_english(raw_sentence)

        translator = IndianLanguageTranslator()


        translated_sentence = translate_text(formed_sentence, target_lang)


        return jsonify({
            "success": True,
            "formed_sentence": formed_sentence,
            "translated_sentence": translated_sentence
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    from waitress import serve
    print("🚀 ISL Backend running at http://localhost:5001", file=sys.stdout)
    serve(app, host="0.0.0.0", port=5001, threads=8)