"""Validated MediaPipe facial feature extraction for the Bisaya affect dataset."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks.python import BaseOptions, vision
from tqdm import tqdm

from src.audio_features import parse_filename


def _point_2d(landmarks, index: int, width: int, height: int) -> np.ndarray:
    """Convert normalized x/y to pixel coordinates; z is intentionally excluded."""
    return np.array([landmarks[index].x * width, landmarks[index].y * height], dtype=float)


def compute_ear(landmarks, width: int, height: int):
    p = lambda i: _point_2d(landmarks, i, width, height)
    left_h = np.linalg.norm(p(33) - p(133))
    right_h = np.linalg.norm(p(362) - p(263))
    left = (np.linalg.norm(p(160) - p(144)) + np.linalg.norm(p(158) - p(153))) / (2 * left_h) if left_h > 1e-6 else np.nan
    right = (np.linalg.norm(p(385) - p(373)) + np.linalg.norm(p(387) - p(380))) / (2 * right_h) if right_h > 1e-6 else np.nan
    return left, right, float(np.nanmean([left, right]))


def compute_mar(landmarks, width: int, height: int):
    p = lambda i: _point_2d(landmarks, i, width, height)
    vertical = np.linalg.norm(p(13) - p(14))
    horizontal = np.linalg.norm(p(61) - p(291))
    return (vertical / horizontal if horizontal > 1e-6 else np.nan), horizontal, vertical


def compute_head_pose(trans_matrix):
    rotation = trans_matrix[:3, :3]
    pitch = np.degrees(np.arctan2(rotation[2, 1], rotation[2, 2]))
    yaw = np.degrees(np.arctan2(-rotation[2, 0], np.sqrt(rotation[2, 1] ** 2 + rotation[2, 2] ** 2)))
    roll = np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))
    return pitch, yaw, roll


def _summary(values, prefix: str, result: dict, digits: int = 4):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    result[f"{prefix}_mean"] = round(float(np.mean(arr)), digits) if arr.size else np.nan
    result[f"{prefix}_std"] = round(float(np.std(arr)), digits) if arr.size else np.nan
    result[f"{prefix}_max"] = round(float(np.max(arr)), digits) if arr.size else np.nan


def _paired_mean(blendshapes: dict, left: str, right: str) -> np.ndarray:
    if left not in blendshapes or right not in blendshapes:
        return np.array([], dtype=float)
    return (np.asarray(blendshapes[left]) + np.asarray(blendshapes[right])) / 2.0


def extract_facial_features_from_video(video_path: str, detector: vision.FaceLandmarker) -> dict:
    path = Path(video_path)
    metadata = parse_filename(str(path.with_suffix(".WAV")))
    metadata.pop("audio_filename", None)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise ValueError(f"Invalid video properties for {path.name}")

    blendshapes: dict[str, list[float]] = {}
    ears, mars, pitches, yaws, rolls = [], [], [], [], []
    decoded_frames = detected_frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp_ms = int(round(decoded_frames * 1000.0 / fps))
        decoded_frames += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect_for_video(image, timestamp_ms)
        if not result.face_landmarks:
            continue
        detected_frames += 1
        landmarks = result.face_landmarks[0]
        ears.append(compute_ear(landmarks, width, height)[2])
        mars.append(compute_mar(landmarks, width, height)[0])
        if result.face_blendshapes:
            for item in result.face_blendshapes[0]:
                blendshapes.setdefault(item.category_name, []).append(item.score)
        if result.facial_transformation_matrixes:
            pitch, yaw, roll = compute_head_pose(result.facial_transformation_matrixes[0])
            pitches.append(pitch)
            yaws.append(yaw)
            rolls.append(roll)
    cap.release()
    if decoded_frames == 0 or detected_frames == 0 or not blendshapes:
        raise ValueError(f"No usable face data in {path.name}")

    detection_rate = detected_frames / decoded_frames
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
        "face_detection_quality_pass": bool(detection_rate >= 0.8),
        "face_geometry_coordinate_system": "2D pixel coordinates",
        "face_landmarker_running_mode": "VIDEO",
    }

    for name, values in sorted(blendshapes.items()):
        _summary(values, f"face_{name}", features)

    smile = _paired_mean(blendshapes, "mouthSmileLeft", "mouthSmileRight")
    frown = _paired_mean(blendshapes, "mouthFrownLeft", "mouthFrownRight")
    brow_down = _paired_mean(blendshapes, "browDownLeft", "browDownRight")
    squint = _paired_mean(blendshapes, "eyeSquintLeft", "eyeSquintRight")
    for arr, name in [(smile, "face_smile"), (frown, "face_frown"), (brow_down, "face_brow_lowerer")]:
        _summary(arr, name, features)

    brow_inner = np.asarray(blendshapes.get("browInnerUp", []), dtype=float)
    jaw_open = np.asarray(blendshapes.get("jawOpen", []), dtype=float)
    _summary(brow_inner, "face_brow_inner_raiser", features)
    _summary(jaw_open, "face_jaw_open", features)
    features["face_eye_squint_mean"] = round(float(np.mean(squint)), 4) if squint.size else np.nan

    disgust_names = ["noseSneerLeft", "noseSneerRight", "mouthUpperUpLeft", "mouthUpperUpRight"]
    disgust_arrays = [np.asarray(blendshapes[name], dtype=float) for name in disgust_names if name in blendshapes]
    features["face_disgust_proxy_mean"] = round(float(np.mean(np.vstack(disgust_arrays))), 4) if disgust_arrays else np.nan
    blendshape_stds = [np.std(values) for values in blendshapes.values() if len(values) > 1]
    features["face_expressiveness_proxy"] = round(float(np.mean(blendshape_stds)), 4) if blendshape_stds else np.nan

    for values, name in [(ears, "face_ear"), (mars, "face_mar")]:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        features[f"{name}_mean"] = round(float(np.mean(arr)), 4) if arr.size else np.nan
        features[f"{name}_std"] = round(float(np.std(arr)), 4) if arr.size else np.nan
        features[f"{name}_{'min' if name.endswith('ear') else 'max'}"] = round(float(np.min(arr) if name.endswith("ear") else np.max(arr)), 4) if arr.size else np.nan

    for values, name in [(pitches, "pitch"), (yaws, "yaw"), (rolls, "roll")]:
        arr = np.asarray(values, dtype=float)
        features[f"face_{name}_mean_deg"] = round(float(np.mean(arr)), 2) if arr.size else np.nan
        features[f"face_{name}_std_deg"] = round(float(np.std(arr)), 2) if arr.size else np.nan
    return features


def extract_all_facial_features(raw_video_dir: str, output_csv: str | None = None) -> pd.DataFrame:
    raw_dir = Path(raw_video_dir).resolve()
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Video directory not found: {raw_dir}")
    files = sorted(p for p in raw_dir.iterdir() if p.suffix.lower() in {".mov", ".mp4"})
    print(f"Found {len(files)} raw video recordings in {raw_dir}")
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="models/face_landmarker.task"),
        running_mode=vision.RunningMode.VIDEO,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )
    records, failures = [], []
    for filepath in tqdm(files, desc="Extracting Facial Features"):
        detector = vision.FaceLandmarker.create_from_options(options)
        try:
            records.append(extract_facial_features_from_video(str(filepath), detector))
        except Exception as exc:
            failures.append({"filename": filepath.name, "error": str(exc)})
        finally:
            detector.close()
    df = pd.DataFrame(records)
    if output_csv:
        output = Path(output_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)
        pd.DataFrame(failures, columns=["filename", "error"]).to_csv(output.parent / "facial_extraction_failures.csv", index=False)
        print(f"Facial feature dataset saved to {output} (Shape: {df.shape}; failures={len(failures)})")
    if failures:
        raise RuntimeError(f"Facial extraction failed for {len(failures)} files; see facial_extraction_failures.csv")
    return df


if __name__ == "__main__":
    extract_all_facial_features("DATA/3ImageRecording/Raw Cut Video", "datasets/facial/facial_feature_dataset.csv")
