"""Validated MediaPipe facial feature extraction for the Bisaya affect dataset."""

from __future__ import annotations

# --- Standard Library Imports ---
import os                     # Operating system utilities for path resolution
import sys                    # System runtime and path modification
from pathlib import Path       # Object-oriented filesystem path manipulation

# Add project root directory to sys.path so sibling packages (e.g., src.audio_features) can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- Computer Vision & Scientific Libraries ---
import cv2                    # OpenCV: video decoding, frame extraction, color space conversion
import mediapipe as mp        # Google MediaPipe: Face Landmarker & Blendshapes extraction
import numpy as np            # Numerical computations: vector norms, Euler angles, trigonometry
import pandas as pd           # Tabular dataset manipulation and CSV export
from mediapipe.tasks.python import BaseOptions, vision  # MediaPipe Tasks API configurations
from tqdm import tqdm         # Terminal progress bar for batch video processing

# Import metadata filename validator to guarantee 1-to-1 linkage with participant survey
from src.audio_features import parse_filename


def _point_2d(landmarks, index: int, width: int, height: int) -> np.ndarray:
    """
    Converts a MediaPipe normalized landmark (values in [0.0, 1.0]) to absolute 2D pixel coordinates.
    Formula: x_px = landmark.x * width, y_px = landmark.y * height.
    Note: Z-coordinate is intentionally excluded to ensure scale-invariant 2D geometric aspect ratios.
    """
    return np.array([landmarks[index].x * width, landmarks[index].y * height], dtype=float)


def compute_ear(landmarks, width: int, height: int):
    """
    Computes Eye Aspect Ratio (EAR) based on the Soukupová & Čech (2016) formulation.
    EAR measures eyelid openness; sharp dips signify blinks, sustained low values indicate fatigue/drowsiness.
    
    Eye Landmark Indices (MediaPipe Face Mesh 468-point topology):
      - Left Eye:  Corner markers (33, 133), Vertical eyelid markers (160, 144) and (158, 153)
      - Right Eye: Corner markers (362, 263), Vertical eyelid markers (385, 373) and (387, 380)
    
    Formula:
      EAR = (||p_upper1 - p_lower1|| + ||p_upper2 - p_lower2||) / (2 * ||p_left - p_right||)
    """
    # Lambda helper to project landmark index into 2D pixel space
    p = lambda i: _point_2d(landmarks, i, width, height)
    
    # Calculate horizontal eye corner distance (eye width) for left and right eyes
    left_h = np.linalg.norm(p(33) - p(133))
    right_h = np.linalg.norm(p(362) - p(263))
    
    # Calculate vertical eye aperture normalized by 2 * horizontal width
    left = (np.linalg.norm(p(160) - p(144)) + np.linalg.norm(p(158) - p(153))) / (2 * left_h) if left_h > 1e-6 else np.nan
    right = (np.linalg.norm(p(385) - p(373)) + np.linalg.norm(p(387) - p(380))) / (2 * right_h) if right_h > 1e-6 else np.nan
    
    # Return (left_ear, right_ear, mean_ear across both eyes)
    return left, right, float(np.nanmean([left, right]))


def compute_mar(landmarks, width: int, height: int):
    """
    Computes Mouth Aspect Ratio (MAR) to quantify speech articulation, yawning, and vocal expression.
    
    Mouth Landmark Indices:
      - Vertical Inner Lips: Upper inner lip (13) to Lower inner lip (14)
      - Horizontal Lip Corners: Left corner (61) to Right corner (291)
      
    Formula:
      MAR = ||p_upper_lip - p_lower_lip|| / ||p_left_corner - p_right_corner||
    """
    p = lambda i: _point_2d(landmarks, i, width, height)
    
    # Vertical distance between upper and lower inner lips
    vertical = np.linalg.norm(p(13) - p(14))
    # Horizontal distance between left and right mouth corners
    horizontal = np.linalg.norm(p(61) - p(291))
    
    # Compute ratio: guard against division by zero if face tracking glitches
    mar_ratio = vertical / horizontal if horizontal > 1e-6 else np.nan
    return mar_ratio, horizontal, vertical


def compute_head_pose(trans_matrix):
    """
    Decomposes the 4x4 facial transformation matrix into 3D Euler angles: Pitch, Yaw, and Roll.
    Trans_matrix is estimated by MediaPipe canonical face model fitting.
    
    Angles:
      - Pitch: Nodding up / down (degrees)
      - Yaw:   Turning left / right (degrees)
      - Roll:  Tilting head sideways towards shoulders (degrees)
    """
    # Extract the top-left 3x3 rotation submatrix
    rotation = trans_matrix[:3, :3]
    
    # Calculate Pitch (rotation around X-axis): arctan2(R[2, 1], R[2, 2])
    pitch = np.degrees(np.arctan2(rotation[2, 1], rotation[2, 2]))
    # Calculate Yaw (rotation around Y-axis): arctan2(-R[2, 0], sqrt(R[2, 1]^2 + R[2, 2]^2))
    yaw = np.degrees(np.arctan2(-rotation[2, 0], np.sqrt(rotation[2, 1] ** 2 + rotation[2, 2] ** 2)))
    # Calculate Roll (rotation around Z-axis): arctan2(R[1, 0], R[0, 0])
    roll = np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))
    
    return pitch, yaw, roll


def _summary(values, prefix: str, result: dict, digits: int = 4):
    """
    Helper function to summarize a time-series of frame-level values across a video clip.
    Computes: mean, standard deviation (variability), and maximum peak value.
    Filters out NaN and infinite values before calculating statistics.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    result[f"{prefix}_mean"] = round(float(np.mean(arr)), digits) if arr.size else np.nan
    result[f"{prefix}_std"] = round(float(np.std(arr)), digits) if arr.size else np.nan
    result[f"{prefix}_max"] = round(float(np.max(arr)), digits) if arr.size else np.nan


def _paired_mean(blendshapes: dict, left: str, right: str) -> np.ndarray:
    """
    Calculates the bilateral symmetrical average of left and right facial Action Units.
    Example: Average of mouthSmileLeft and mouthSmileRight to compute global smile activation.
    """
    if left not in blendshapes or right not in blendshapes:
        return np.array([], dtype=float)
    return (np.asarray(blendshapes[left]) + np.asarray(blendshapes[right])) / 2.0


def extract_facial_features_from_video(video_path: str, detector: vision.FaceLandmarker) -> dict:
    """
    Processes an individual participant .MP4 or .MOV video recording:
    1. Reads frames using OpenCV VideoCapture.
    2. Runs MediaPipe FaceLandmarker with timestamp tracking in VIDEO mode.
    3. Extracts 52 FACS blendshapes per frame (smile, brow furrow, jaw drop, sneer, etc.).
    4. Computes geometric biomarkers: Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR).
    5. Computes 3D head pose orientation: Pitch, Yaw, Roll.
    6. Aggregates frame dynamics into clip-level summary features (mean, std, max).
    """
    path = Path(video_path)
    
    # Reconcile video file with survey metadata using audio filename counterpart
    metadata = parse_filename(str(path.with_suffix(".WAV")))
    metadata.pop("audio_filename", None)  # Remove audio filename key to avoid confusion
    
    # Open video stream via OpenCV
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
        
    # Read hardware video capture properties
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Validate video metadata sanity
    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise ValueError(f"Invalid video properties for {path.name}")

    # Accumulator dictionaries and lists for frame-level biomarker trajectories
    blendshapes: dict[str, list[float]] = {}
    ears, mars, pitches, yaws, rolls = [], [], [], [], []
    decoded_frames = detected_frames = 0
    
    # Process video frame-by-frame
    while True:
        ok, frame = cap.read()
        if not ok:
            break  # End of video reached
            
        # Monotonically increasing timestamp in milliseconds required by MediaPipe VIDEO mode
        timestamp_ms = int(round(decoded_frames * 1000.0 / fps))
        decoded_frames += 1
        
        # Convert OpenCV BGR color format to RGB format expected by MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        # Execute face landmark and blendshape detection
        result = detector.detect_for_video(image, timestamp_ms)
        
        # Skip frame if face was obscured, out of frame, or undetected
        if not result.face_landmarks:
            continue
            
        detected_frames += 1
        landmarks = result.face_landmarks[0]
        
        # Compute geometric aspect ratios for current frame
        ears.append(compute_ear(landmarks, width, height)[2])
        mars.append(compute_mar(landmarks, width, height)[0])
        
        # Extract 52 FACS blendshape scores (range [0.0, 1.0])
        if result.face_blendshapes:
            for item in result.face_blendshapes[0]:
                blendshapes.setdefault(item.category_name, []).append(item.score)
                
        # Extract 3D head pose Euler angles
        if result.facial_transformation_matrixes:
            pitch, yaw, roll = compute_head_pose(result.facial_transformation_matrixes[0])
            pitches.append(pitch)
            yaws.append(yaw)
            rolls.append(roll)
            
    # Release hardware video file handle
    cap.release()
    
    # Audit check: ensure the video contained valid, trackable facial frames
    if decoded_frames == 0 or detected_frames == 0 or not blendshapes:
        raise ValueError(f"No usable face data in {path.name}")

    # Calculate tracking quality: fraction of video frames with successful face detection
    detection_rate = detected_frames / decoded_frames
    
    # Assemble video-level metadata and quality control flags
    features = {
        **metadata,
        "video_filename": path.name,
        "video_fps": round(fps, 4),
        "video_reported_frames": reported_frames,
        "video_decoded_frames": decoded_frames,
        "video_duration_sec": round(decoded_frames / fps, 4),
        "video_width_px": width,
        "video_height_px": height,
        "face_detected_frames": detected_frames,
        "face_detection_rate": round(detection_rate, 4),
        "face_detection_quality_pass": bool(detection_rate >= 0.8),  # Pass if >= 80% frames tracked
        "face_geometry_coordinate_system": "2D pixel coordinates",
        "face_landmarker_running_mode": "VIDEO",
    }

    # Aggregate all 52 raw FACS blendshapes (mean, std, max)
    for name, values in sorted(blendshapes.items()):
        _summary(values, f"face_{name}", features)

    # Compute domain-specific bilateral Action Unit combinations
    smile = _paired_mean(blendshapes, "mouthSmileLeft", "mouthSmileRight")       # AU12: Zygomaticus major
    frown = _paired_mean(blendshapes, "mouthFrownLeft", "mouthFrownRight")       # AU15: Depressor anguli oris
    brow_down = _paired_mean(blendshapes, "browDownLeft", "browDownRight")       # AU4:  Corrugator supercilii
    squint = _paired_mean(blendshapes, "eyeSquintLeft", "eyeSquintRight")        # AU7:  Orbicularis oculi
    
    # Summarize combined expressions
    for arr, name in [(smile, "face_smile"), (frown, "face_frown"), (brow_down, "face_brow_lowerer")]:
        _summary(arr, name, features)

    # Unilateral or isolated action units
    brow_inner = np.asarray(blendshapes.get("browInnerUp", []), dtype=float)     # AU1: Frontalis (inner raiser)
    jaw_open = np.asarray(blendshapes.get("jawOpen", []), dtype=float)           # AU26/27: Masseter relaxation
    _summary(brow_inner, "face_brow_inner_raiser", features)
    _summary(jaw_open, "face_jaw_open", features)
    features["face_eye_squint_mean"] = round(float(np.mean(squint)), 4) if squint.size else np.nan

    # Disgust proxy: combines nose wrinkling (AU9) and upper lip raising (AU10)
    disgust_names = ["noseSneerLeft", "noseSneerRight", "mouthUpperUpLeft", "mouthUpperUpRight"]
    disgust_arrays = [np.asarray(blendshapes[name], dtype=float) for name in disgust_names if name in blendshapes]
    features["face_disgust_proxy_mean"] = round(float(np.mean(np.vstack(disgust_arrays))), 4) if disgust_arrays else np.nan
    
    # Global facial expressiveness: average standard deviation across all blendshapes (facial animation dynamism)
    blendshape_stds = [np.std(values) for values in blendshapes.values() if len(values) > 1]
    features["face_expressiveness_proxy"] = round(float(np.mean(blendshape_stds)), 4) if blendshape_stds else np.nan

    # Summarize Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR)
    for values, name in [(ears, "face_ear"), (mars, "face_mar")]:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        features[f"{name}_mean"] = round(float(np.mean(arr)), 4) if arr.size else np.nan
        features[f"{name}_std"] = round(float(np.std(arr)), 4) if arr.size else np.nan
        # For EAR, track minimum (deepest blink); for MAR, track maximum (widest mouth opening)
        features[f"{name}_{'min' if name.endswith('ear') else 'max'}"] = round(float(np.min(arr) if name.endswith("ear") else np.max(arr)), 4) if arr.size else np.nan

    # Summarize 3D head pose movements (mean angle and angular variation in degrees)
    for values, name in [(pitches, "pitch"), (yaws, "yaw"), (rolls, "roll")]:
        arr = np.asarray(values, dtype=float)
        features[f"face_{name}_mean_deg"] = round(float(np.mean(arr)), 2) if arr.size else np.nan
        features[f"face_{name}_std_deg"] = round(float(np.std(arr)), 2) if arr.size else np.nan
        
    return features


def extract_all_facial_features(raw_video_dir: str, output_csv: str | None = None) -> pd.DataFrame:
    """
    Batch facial feature extraction pipeline:
    Iterates through all raw video recordings (.MOV / .MP4) in the target directory,
    initializes MediaPipe FaceLandmarker, extracts 225 FACS & geometric features,
    and saves the consolidated tabular dataset to disk.
    """
    raw_dir = Path(raw_video_dir).resolve()
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Video directory not found: {raw_dir}")
        
    # Gather all video files with .mov or .mp4 extensions
    files = sorted(p for p in raw_dir.iterdir() if p.suffix.lower() in {".mov", ".mp4"})
    print(f"Found {len(files)} raw video recordings in {raw_dir}")
    
    # Configure MediaPipe FaceLandmarker options
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="models/face_landmarker.task"), # Path to quantized TFLite model
        running_mode=vision.RunningMode.VIDEO,                                    # Video mode uses temporal filtering
        output_face_blendshapes=True,                                             # Enables 52 FACS blendshapes
        output_facial_transformation_matrixes=True,                               # Enables 4x4 pose matrix
        num_faces=1,                                                              # Track single participant
    )
    
    records, failures = [], []
    
    # Process each video with progress tracking via tqdm
    for filepath in tqdm(files, desc="Extracting Facial Features"):
        detector = vision.FaceLandmarker.create_from_options(options)
        try:
            records.append(extract_facial_features_from_video(str(filepath), detector))
        except Exception as exc:
            failures.append({"filename": filepath.name, "error": str(exc)})
        finally:
            detector.close()  # Ensure TFLite C++ graph memory is freed immediately
            
    # Assemble master DataFrame
    df = pd.DataFrame(records)
    
    # Export to CSV if output destination path provided
    if output_csv:
        output = Path(output_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)
        pd.DataFrame(failures, columns=["filename", "error"]).to_csv(output.parent / "facial_extraction_failures.csv", index=False)
        print(f"Facial feature dataset saved to {output} (Shape: {df.shape}; failures={len(failures)})")
        
    if failures:
        raise RuntimeError(f"Facial extraction failed for {len(failures)} files; see facial_extraction_failures.csv")
        
    return df


# Main script entry point
if __name__ == "__main__":
    extract_all_facial_features("DATA/3ImageRecording/Raw Cut Video", "datasets/facial/facial_feature_dataset.csv")
