"""
Facial Feature Extraction Module for Bisaya Affect Recognition.
Processes raw video recordings using MediaPipe FaceLandmarker to extract
FACS blendshapes, 3D facial landmarks, geometric Action Unit proxies (EAR, MAR),
and head pose dynamics across time.
"""

import os
import sys
import glob

# Ensure workspace root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from tqdm import tqdm

from src.audio_features import AFFECT_TAXONOMY, parse_filename, SURVEY_METADATA



def compute_ear(landmarks):
    """
    Computes Eye Aspect Ratio (EAR) for both eyes from MediaPipe 478 landmarks.
    """
    # Left eye landmarks: 33 (outer), 133 (inner), 160, 158 (upper), 144, 153 (lower)
    p33 = np.array([landmarks[33].x, landmarks[33].y, landmarks[33].z])
    p133 = np.array([landmarks[133].x, landmarks[133].y, landmarks[133].z])
    p160 = np.array([landmarks[160].x, landmarks[160].y, landmarks[160].z])
    p144 = np.array([landmarks[144].x, landmarks[144].y, landmarks[144].z])
    p158 = np.array([landmarks[158].x, landmarks[158].y, landmarks[158].z])
    p153 = np.array([landmarks[153].x, landmarks[153].y, landmarks[153].z])
    
    d_v1_left = np.linalg.norm(p160 - p144)
    d_v2_left = np.linalg.norm(p158 - p153)
    d_h_left = np.linalg.norm(p33 - p133)
    ear_left = (d_v1_left + d_v2_left) / (2.0 * d_h_left) if d_h_left > 1e-6 else 0.0

    # Right eye landmarks: 362 (inner), 263 (outer), 385, 387 (upper), 373, 380 (lower)
    p362 = np.array([landmarks[362].x, landmarks[362].y, landmarks[362].z])
    p263 = np.array([landmarks[263].x, landmarks[263].y, landmarks[263].z])
    p385 = np.array([landmarks[385].x, landmarks[385].y, landmarks[385].z])
    p373 = np.array([landmarks[373].x, landmarks[373].y, landmarks[373].z])
    p387 = np.array([landmarks[387].x, landmarks[387].y, landmarks[387].z])
    p380 = np.array([landmarks[380].x, landmarks[380].y, landmarks[380].z])
    
    d_v1_right = np.linalg.norm(p385 - p373)
    d_v2_right = np.linalg.norm(p387 - p380)
    d_h_right = np.linalg.norm(p362 - p263)
    ear_right = (d_v1_right + d_v2_right) / (2.0 * d_h_right) if d_h_right > 1e-6 else 0.0
    
    ear_avg = (ear_left + ear_right) / 2.0
    return ear_left, ear_right, ear_avg


def compute_mar(landmarks):
    """
    Computes Mouth Aspect Ratio (MAR) from MediaPipe landmarks:
    upper lip (13), lower lip (14), left mouth corner (61), right mouth corner (291)
    """
    p13 = np.array([landmarks[13].x, landmarks[13].y, landmarks[13].z])
    p14 = np.array([landmarks[14].x, landmarks[14].y, landmarks[14].z])
    p61 = np.array([landmarks[61].x, landmarks[61].y, landmarks[61].z])
    p291 = np.array([landmarks[291].x, landmarks[291].y, landmarks[291].z])
    
    v_dist = np.linalg.norm(p13 - p14)
    h_dist = np.linalg.norm(p61 - p291)
    mar = v_dist / h_dist if h_dist > 1e-6 else 0.0
    return mar, h_dist, v_dist


def compute_head_pose(trans_matrix):
    """
    Extracts Euler angles (pitch, yaw, roll in degrees) from 4x4 transformation matrix.
    """
    R = trans_matrix[:3, :3]
    pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    yaw = np.degrees(np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2)))
    roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return pitch, yaw, roll


def extract_facial_features_from_video(video_path: str, detector: vision.FaceLandmarker) -> dict:
    """
    Processes video clip frame-by-frame and extracts facial features.
    """
    filename = os.path.basename(video_path)
    stem, _ = os.path.splitext(filename)
    parts = stem.split("_")
    participant_code = parts[0] if len(parts) >= 1 else stem
    year_level = parts[1] if len(parts) >= 2 else "Unknown"
    session = parts[2] if len(parts) >= 3 else "Unknown"
    spoken_word = parts[3] if len(parts) >= 4 else "Unknown"
    
    tax = AFFECT_TAXONOMY.get(spoken_word, {
        "category": "Unknown", "valence_group": "Unknown", "arousal_group": "Unknown", "english": spoken_word
    })
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_dur = total_frames / fps if fps > 0 else 0.0
    
    frame_blendshapes = {}
    frame_ears = []
    frame_mars = []
    frame_pitches = []
    frame_yaws = []
    frame_rolls = []
    
    detected_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = detector.detect(mp_img)
        
        if res.face_landmarks and len(res.face_landmarks) > 0:
            detected_frames += 1
            landmarks = res.face_landmarks[0]
            
            # Geometric metrics
            _, _, ear_avg = compute_ear(landmarks)
            mar, _, _ = compute_mar(landmarks)
            frame_ears.append(ear_avg)
            frame_mars.append(mar)
            
            # Blendshapes
            if res.face_blendshapes and len(res.face_blendshapes) > 0:
                for b in res.face_blendshapes[0]:
                    name = b.category_name
                    if name not in frame_blendshapes:
                        frame_blendshapes[name] = []
                    frame_blendshapes[name].append(b.score)
                    
            # Head pose
            if res.facial_transformation_matrixes and len(res.facial_transformation_matrixes) > 0:
                pitch, yaw, roll = compute_head_pose(res.facial_transformation_matrixes[0])
                frame_pitches.append(pitch)
                frame_yaws.append(yaw)
                frame_rolls.append(roll)
                
    cap.release()
    
    detection_rate = detected_frames / total_frames if total_frames > 0 else 0.0
    survey = SURVEY_METADATA.get(participant_code, {})
    
    features = {
        "participant_code": participant_code,
        "year_level": survey.get("year_level", "4th Year"),
        "class_activity": survey.get("class_activity", "Unknown"),
        "subject": survey.get("subject", "Unknown"),
        "session": session,
        "session_name": survey.get("session_name", "Morning" if session == "AM" else "Afternoon"),
        "spoken_word": spoken_word,
        "self_reported_feeling": survey.get("self_reported_feeling", spoken_word),
        "valence_score": survey.get("valence_score", 3),
        "valence_label": survey.get("valence_label", tax["valence_group"]),
        "arousal_score": survey.get("arousal_score", 3),
        "arousal_label": survey.get("arousal_label", tax["arousal_group"]),
        "affect_category": tax["category"],
        "word_english": tax["english"],
        "date_collected": survey.get("date_collected", ""),
        "video_filename": filename,
        "video_fps": round(fps, 2),
        "video_total_frames": total_frames,
        "video_detected_frames": detected_frames,
        "video_duration_sec": round(video_dur, 4),
        "face_detection_rate": round(detection_rate, 4),
    }
    
    # 1. Summary statistics for all 52 FACS blendshapes
    for name, vals in frame_blendshapes.items():
        arr = np.array(vals)
        features[f"face_{name}_mean"] = round(float(np.mean(arr)), 4)
        features[f"face_{name}_std"] = round(float(np.std(arr)), 4)
        features[f"face_{name}_max"] = round(float(np.max(arr)), 4)
        
    # 2. Key Action Unit / Affect Composites
    # Smile intensity (AU12 - Lip Corner Puller)
    smile_l = frame_blendshapes.get("mouthSmileLeft", [0.0])
    smile_r = frame_blendshapes.get("mouthSmileRight", [0.0])
    smile_comb = (np.array(smile_l) + np.array(smile_r)) / 2.0
    features["face_smile_mean"] = round(float(np.mean(smile_comb)), 4)
    features["face_smile_std"] = round(float(np.std(smile_comb)), 4)
    features["face_smile_max"] = round(float(np.max(smile_comb)), 4)
    
    # Frown intensity (AU15 - Lip Corner Depressor)
    frown_l = frame_blendshapes.get("mouthFrownLeft", [0.0])
    frown_r = frame_blendshapes.get("mouthFrownRight", [0.0])
    frown_comb = (np.array(frown_l) + np.array(frown_r)) / 2.0
    features["face_frown_mean"] = round(float(np.mean(frown_comb)), 4)
    features["face_frown_std"] = round(float(np.std(frown_comb)), 4)
    features["face_frown_max"] = round(float(np.max(frown_comb)), 4)
    
    # Brow Lowerer (AU4 - Anger / Concentration / Effort)
    brow_l = frame_blendshapes.get("browDownLeft", [0.0])
    brow_r = frame_blendshapes.get("browDownRight", [0.0])
    brow_comb = (np.array(brow_l) + np.array(brow_r)) / 2.0
    features["face_brow_lowerer_mean"] = round(float(np.mean(brow_comb)), 4)
    features["face_brow_lowerer_std"] = round(float(np.std(brow_comb)), 4)
    features["face_brow_lowerer_max"] = round(float(np.max(brow_comb)), 4)
    
    # Brow Inner Up (AU1 - Distress / Sadness / Wonder)
    brow_inner = np.array(frame_blendshapes.get("browInnerUp", [0.0]))
    features["face_brow_inner_raiser_mean"] = round(float(np.mean(brow_inner)), 4)
    features["face_brow_inner_raiser_max"] = round(float(np.max(brow_inner)), 4)
    
    # Jaw Open (AU26/27 - Mouth opening / Articulation)
    jaw_open = np.array(frame_blendshapes.get("jawOpen", [0.0]))
    features["face_jaw_open_mean"] = round(float(np.mean(jaw_open)), 4)
    features["face_jaw_open_std"] = round(float(np.std(jaw_open)), 4)
    features["face_jaw_open_max"] = round(float(np.max(jaw_open)), 4)
    
    # Eye Squint / Lid Tightener (AU7)
    squint_l = frame_blendshapes.get("eyeSquintLeft", [0.0])
    squint_r = frame_blendshapes.get("eyeSquintRight", [0.0])
    squint_comb = (np.array(squint_l) + np.array(squint_r)) / 2.0
    features["face_eye_squint_mean"] = round(float(np.mean(squint_comb)), 4)
    
    # Disgust / Nose Sneer & Upper Lip Raiser (AU9 + AU10)
    disgust_items = [
        frame_blendshapes.get("noseSneerLeft", [0.0]),
        frame_blendshapes.get("noseSneerRight", [0.0]),
        frame_blendshapes.get("mouthUpperUpLeft", [0.0]),
        frame_blendshapes.get("mouthUpperUpRight", [0.0])
    ]
    disgust_comb = np.mean([np.array(item) for item in disgust_items], axis=0)
    features["face_disgust_mean"] = round(float(np.mean(disgust_comb)), 4)
    
    # Expressiveness Score: Mean standard deviation across all blendshapes
    all_stds = [np.std(vals) for vals in frame_blendshapes.values() if len(vals) > 1]
    features["face_expressiveness_score"] = round(float(np.mean(all_stds)), 4) if all_stds else 0.0
    
    # 3. Geometric Landmark Features (EAR & MAR)
    if frame_ears:
        features["face_ear_mean"] = round(float(np.mean(frame_ears)), 4)
        features["face_ear_std"] = round(float(np.std(frame_ears)), 4)
        features["face_ear_min"] = round(float(np.min(frame_ears)), 4)
    else:
        features["face_ear_mean"] = 0.0
        features["face_ear_std"] = 0.0
        features["face_ear_min"] = 0.0
        
    if frame_mars:
        features["face_mar_mean"] = round(float(np.mean(frame_mars)), 4)
        features["face_mar_std"] = round(float(np.std(frame_mars)), 4)
        features["face_mar_max"] = round(float(np.max(frame_mars)), 4)
    else:
        features["face_mar_mean"] = 0.0
        features["face_mar_std"] = 0.0
        features["face_mar_max"] = 0.0
        
    # 4. Head Pose Features
    if frame_pitches:
        features["face_pitch_mean_deg"] = round(float(np.mean(frame_pitches)), 2)
        features["face_pitch_std_deg"] = round(float(np.std(frame_pitches)), 2)
        features["face_yaw_mean_deg"] = round(float(np.mean(frame_yaws)), 2)
        features["face_yaw_std_deg"] = round(float(np.std(frame_yaws)), 2)
        features["face_roll_mean_deg"] = round(float(np.mean(frame_rolls)), 2)
        features["face_roll_std_deg"] = round(float(np.std(frame_rolls)), 2)
    else:
        features["face_pitch_mean_deg"] = 0.0
        features["face_pitch_std_deg"] = 0.0
        features["face_yaw_mean_deg"] = 0.0
        features["face_yaw_std_deg"] = 0.0
        features["face_roll_mean_deg"] = 0.0
        features["face_roll_std_deg"] = 0.0
        
    return features


def extract_all_facial_features(raw_video_dir: str, output_csv: str = None) -> pd.DataFrame:
    """
    Extracts facial features from all raw video recordings.
    """
    raw_video_dir = os.path.abspath(raw_video_dir)
    files = sorted([
        os.path.join(raw_video_dir, f)
        for f in os.listdir(raw_video_dir)
        if f.lower().endswith((".mov", ".mp4"))
    ])
    
    print(f"Found {len(files)} raw video recordings in {raw_video_dir}")
    
    base_options = BaseOptions(model_asset_path='models/face_landmarker.task')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)
    
    records = []
    for filepath in tqdm(files, desc="Extracting Facial Features"):
        try:
            feat = extract_facial_features_from_video(filepath, detector)
            records.append(feat)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            
    detector.close()
    
    df = pd.DataFrame(records)
    
    if output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Facial feature dataset saved to {output_csv} (Shape: {df.shape})")
        
    return df


if __name__ == "__main__":
    video_dir = "DATA/3ImageRecording/Raw Cut Video"
    out_csv = "datasets/facial_feature_dataset.csv"
    extract_all_facial_features(video_dir, out_csv)
