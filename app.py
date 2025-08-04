# -- coding: utf-8 --


import cv2
import numpy as np
import mediapipe as mp
import os
from collections import deque
import sys
from PIL import Image, ImageDraw, ImageFont
import time  # For timing feature

# Fix protobuf error
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Import TensorFlow
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf._version_}")
except ImportError as e:
    print(f"Error importing TensorFlow: {e}")
    print("Please ensure TensorFlow is installed (e.g., 'conda install tensorflow')")
    sys.exit(1)

# Initialize MediaPipe Pose and Hands
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Paths to trained models
MODEL_PATHS = [
    r"C:\Users\Deo sagar\Downloads\best_model_0.keras",
    r"C:\Users\Deo sagar\Downloads\best_model_1.keras",
    r"C:\Users\Deo sagar\Downloads\best_model_2.keras"
]
try:
    models = [tf.keras.models.load_model(path) for path in MODEL_PATHS]
    print(f"Loaded {len(models)} models successfully: {MODEL_PATHS}")
    for i, model in enumerate(models):
        print(f"Model {i} input shape: {model.input_shape}")
except Exception as e:
    print(f"Error loading models: {e}")
    sys.exit(1)

# Class names in Hindi
CLASS_NAMES = ["नाम", "कैसे", "क्या", "नमस्ते", "आप", "आपका"]

# Parameters
SEQUENCE_LENGTH = 30
MODEL_SEQUENCE_LENGTH = 100  # Training used 100 frames
KEYPOINT_SIZE = 21 * 3 * 2  # 126 features (hands only)

# Load a font that supports Hindi
FONT_PATH = r"C:\Users\Deo sagar\Downloads\MANGAL\MANGAL.TTF"  # Mangal font
FONT_SIZE = 30
try:
    FONT = ImageFont.truetype(FONT_PATH, FONT_SIZE)
except:
    print("Hindi font not found. Please install a Devanagari font (e.g., Noto Sans Devanagari).")
    sys.exit(1)

def extract_keypoints(pose_results, hands_results):
    """
    Extract only hand keypoints (left and right, x, y, z) to match model training data.
    Returns 126 features (21 landmarks * 3 coordinates * 2 hands).
    Pose keypoints are excluded for prediction but kept for visualization.
    """
    keypoints = []
    hand_data = {'left': [0.0] * (21 * 3), 'right': [0.0] * (21 * 3)}
    if hands_results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
            if idx < len(hands_results.multi_handedness):
                handedness = hands_results.multi_handedness[idx].classification[0].label.lower()
                if handedness in hand_data:
                    hand_data[handedness] = [val for lm in hand_landmarks.landmark for val in (lm.x, lm.y, lm.z)]
    keypoints.extend(hand_data['left'] + hand_data['right'])
    keypoints = [0.0 if np.isnan(val) or np.isinf(val) else val for val in keypoints]
    
    # Standardize keypoints to match training
    keypoints = np.array(keypoints)
    mean = np.mean(keypoints)
    std = np.std(keypoints)
    if std > 0:
        keypoints = (keypoints - mean) / std
    
    return keypoints

def pad_sequence(sequence, max_length=MODEL_SEQUENCE_LENGTH):
    """
    Pad sequence to match model's expected input length (100 frames).
    """
    sequence = np.array(sequence)
    if sequence.shape[0] < max_length:
        padding = np.zeros((max_length - sequence.shape[0], sequence.shape[1]))
        sequence = np.vstack((sequence, padding))
    elif sequence.shape[0] > max_length:
        sequence = sequence[-max_length:]
    return sequence

def ensemble_predict(models, sequence_array):
    """
    Predict using ensemble hard voting.
    Returns predicted class index and average confidence of the voted class.
    """
    try:
        predictions = np.array([model.predict(sequence_array, verbose=0) for model in models])  # Shape: (n_models, 1, n_classes)
        predicted_classes = np.argmax(predictions, axis=2).flatten()  # Shape: (n_models,)
        voted_class = np.bincount(predicted_classes).argmax()  # Most common class
        confidences = predictions[:, 0, voted_class]  # Confidences for voted class
        avg_confidence = np.mean(confidences)
        return voted_class, avg_confidence
    except Exception as e:
        print(f"Prediction error in ensemble_predict: {e}")
        return -1, 0.0

def draw_hindi_text(image, text, position, font, color=(0, 255, 0)):
    # Convert OpenCV image to PIL format
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    draw.text(position, text, font=font, fill=color)
    # Convert back to OpenCV format
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def main():
    pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6, model_complexity=2)
    hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=2)
    
    # Use default webcam
    CAMERA_SOURCE = 0
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        print("Error: Could not open webcam. Ensure a webcam is connected and accessible.")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    sequence = deque(maxlen=SEQUENCE_LENGTH)
    predicted_sign = "सीक्वेंस की प्रतीक्षा..."  # "Waiting for sequence..." in Hindi
    last_predicted_class = -1
    start_time = None
    display_time = None
    is_displaying = False
    last_confidence = 0.0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        pose_results = pose.process(rgb_frame)
        hands_results = hands.process(rgb_frame)
        
        keypoints = extract_keypoints(pose_results, hands_results)
        sequence.append(keypoints)
        
        stickman_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(stickman_frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2))
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(stickman_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))
        
        current_time = time.time()
        
        if is_displaying:
            # Display phase: show the sign for 5 seconds
            if current_time - display_time < 5.0:
                frame = draw_hindi_text(frame, f"संकेत: {predicted_sign} ({last_confidence:.2f})", 
                                       (10, 30), FONT, color=(0, 255, 0))
            else:
                # End display phase
                is_displaying = False
                predicted_sign = "सीक्वेंस की प्रतीक्षा..."
                sequence.clear()  # Clear for next analysis
                start_time = None
                last_predicted_class = -1
        else:
            # Analysis phase: collect predictions for 5 seconds
            if len(sequence) == SEQUENCE_LENGTH:
                try:
                    sequence_array = pad_sequence(sequence)
                    sequence_array = np.expand_dims(sequence_array, axis=0)  # Shape: (1, 100, 126)
                    predicted_class, confidence = ensemble_predict(models, sequence_array)
                    
                    if predicted_class == -1:  # Error case
                        predicted_sign = "भविष्यवाणी विफल"
                        start_time = None
                        last_predicted_class = -1
                    else:
                        if predicted_class == last_predicted_class:
                            # Same class, continue timing
                            if start_time is None:
                                start_time = current_time
                            elif current_time - start_time >= 5.0:
                                # 5 seconds of analysis, switch to display
                                predicted_sign = CLASS_NAMES[predicted_class]
                                last_confidence = confidence
                                display_time = current_time
                                is_displaying = True
                                sequence.clear()  # Clear for next cycle
                        else:
                            # New class, reset timer
                            start_time = current_time
                            last_predicted_class = predicted_class
                            
                except Exception as e:
                    print(f"Prediction error: {e}")
                    predicted_sign = "भविष्यवाणी विफल"
                    start_time = None
                    last_predicted_class = -1
        
        frame = draw_hindi_text(frame, f"भविष्यवाणी: {predicted_sign}", 
                               (10, 60), FONT, color=(255, 255, 255))
        
        combined_frame = np.hstack((frame, stickman_frame))
        cv2.imshow("Live Sign Prediction", combined_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    hands.close()
    print("Program terminated.")

if _name_ == "_main_":
    main()