"""
Face crop and alignment script for dataset preparation.

For each video in the dataset, detects the face in every frame using MediaPipe,
aligns it so the eyes are level and at a consistent position, and saves paired
256x256 low-quality and high-quality crops as PNG files.

Usage:
    python crop.py --crf <crf> --codec <codec>

    e.g. python crop.py --crf 44 --codec h264
"""

import cv2
import numpy as np
import dlib
import mediapipe as mp
from PIL import Image
from glob import glob
from pathlib import Path
from argparse import ArgumentParser
from torchmetrics.image import PeakSignalNoiseRatio
import torchvision.transforms as T
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration — set these before running
# ---------------------------------------------------------------------------
ROOT = 'path/to/dataset'                        # Root directory of the DFD dataset
DLIB_PREDICTOR = 'path/to/shape_predictor_68_face_landmarks.dat'
EYE = (.25, .10)                                # Desired left eye position (x, y) as fraction of face width/height
# ---------------------------------------------------------------------------

mp_face_detection = mp.solutions.face_detection
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(DLIB_PREDICTOR)


def align_face(lq, hq, face_detector, desiredLeftEye=EYE,
               desiredFaceWidth=256, desiredFaceHeight=256, prev=None):
    """
    Detect and align a face in a frame pair using MediaPipe for detection
    and dlib for landmark-based eye alignment.

    Uses eyebrow landmarks (rather than eye corners) for a more stable
    alignment across frames.

    Args:
        lq: Low-quality frame (BGR numpy array).
        hq: High-quality frame (BGR numpy array).
        face_detector: MediaPipe FaceDetection instance.
        desiredLeftEye: Target (x, y) position of the left eye as a fraction
                        of the output face dimensions.
        desiredFaceWidth: Output crop width in pixels.
        desiredFaceHeight: Output crop height in pixels.
        prev: If provided, reuse this scale from the previous frame for
              temporal consistency.

    Returns:
        Tuple of (aligned_lq, aligned_hq, transform_matrix, face_rect, scale),
        or (None, None, None, None, None) if no face is detected.
    """
    frame_height, frame_width = hq.shape[:2]
    faces = face_detector.process(cv2.cvtColor(hq, cv2.COLOR_BGR2RGB))

    if not faces.detections:
        return None, None, None, None, None

    face = faces.detections[0]
    face_rect = np.multiply(
        [
            face.location_data.relative_bounding_box.xmin,
            face.location_data.relative_bounding_box.ymin,
            face.location_data.relative_bounding_box.width,
            face.location_data.relative_bounding_box.height,
        ],
        [frame_width, frame_height, frame_width, frame_height]
    ).astype(int)
    x, y, w, h = face_rect

    # Use dlib eyebrow landmarks for alignment (more stable than eye corners)
    landmarks = predictor(hq, dlib.rectangle(x, y, x + w, y + h))
    left_brow = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(18, 21)])
    right_brow = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(23, 26)])

    left_eye_center = left_brow.mean(axis=0)
    right_eye_center = right_brow.mean(axis=0)

    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dy, dx))

    eyesCenter = (
        (left_eye_center[0] + right_eye_center[0]) // 2,
        (left_eye_center[1] + right_eye_center[1]) // 2
    )

    desiredRightEyeX = 1.0 - desiredLeftEye[0]
    dist = np.sqrt(dx ** 2 + dy ** 2)
    desiredDist = (desiredRightEyeX - desiredLeftEye[0]) * desiredFaceWidth
    scale = desiredDist / dist

    M = cv2.getRotationMatrix2D(eyesCenter, angle, prev if prev else scale)
    M[0, 2] += desiredFaceWidth * 0.5 - eyesCenter[0]
    M[1, 2] += desiredFaceHeight * desiredLeftEye[1] - eyesCenter[1]

    aligned_lq = cv2.warpAffine(lq, M, (desiredFaceWidth, desiredFaceHeight),
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    aligned_hq = cv2.warpAffine(hq, M, (desiredFaceWidth, desiredFaceHeight),
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return aligned_lq, aligned_hq, M, (x, y, w, h), scale


def make_face_crops(face_detector):
    """
    Process all videos in the dataset, saving aligned face crop pairs.

    For each video, reads paired low-quality and high-quality frames,
    aligns each face, and writes the crops to lq/ and hq/ subdirectories.
    Saves a CSV of per-video PSNR scores between the LQ and HQ crops.

    Args:
        face_detector: MediaPipe FaceDetection instance.
    """
    psnr = PeakSignalNoiseRatio(data_range=1.0)
    scores = []

    for fpath in sorted(glob(f'{ROOT}/vid_{CODEC}/{CRF}/*')):
        fname = Path(fpath).stem
        cap_lq = cv2.VideoCapture(fpath)
        cap_hq = cv2.VideoCapture(f'{ROOT}/videos/{fname}.mp4')

        out_path = f'{ROOT}/face_{CODEC}/{EYE[0]}/{CRF}/{fname}'
        Path(f'{out_path}/lq').mkdir(parents=True, exist_ok=True)
        Path(f'{out_path}/hq').mkdir(parents=True, exist_ok=True)

        frame_scores = []
        i = 0
        scale = None

        while True:
            ret_hq, frame_hq = cap_hq.read()
            ret_lq, frame_lq = cap_lq.read()
            if not ret_hq or not ret_lq:
                break

            frame_height, frame_width = frame_hq.shape[:2]
            aligned_lq, aligned_hq, M, face_rect, scale = align_face(
                frame_lq, frame_hq, face_detector, prev=scale
            )

            # Lock in the scale from the first frame for temporal consistency
            if i == 0:
                scale_final = scale
            else:
                scale = scale_final

            if aligned_lq is not None:
                x, y, w, h = face_rect
                if x < 0 or x > frame_width or y < 0 or y > frame_height:
                    continue

                cv2.imwrite(f'{out_path}/lq/{i:04d}.png', aligned_lq)
                cv2.imwrite(f'{out_path}/hq/{i:04d}.png', aligned_hq)

                lq_t = T.functional.to_tensor(aligned_lq)
                hq_t = T.functional.to_tensor(aligned_hq)
                frame_scores.append(psnr(lq_t, hq_t).item())
                i += 1

        cap_hq.release()
        cap_lq.release()

        scores.append({'vid': fname, 'PSNR': np.mean(np.array(frame_scores))})
        print(f'Video: {fname} | Avg PSNR: {scores[-1]["PSNR"]:.4f}')
        print('-' * 30)

    df = pd.DataFrame(scores).sort_values(by='vid')
    mean_row = pd.Series(['Avg'] + df.iloc[:, 1:].mean().tolist(), index=df.columns)
    df = pd.concat([df, mean_row.to_frame().T], ignore_index=True)
    Path('scores').mkdir(parents=True, exist_ok=True)
    df.to_csv(f'scores/{CODEC}_{CRF}_{EYE[0]}.csv', index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-f', '--crf', required=True, help='Compression CRF value (e.g. 36, 40, 44).')
    parser.add_argument('-c', '--codec', required=True, help='Video codec (e.g. h264, h265).')
    args = parser.parse_args()
    CRF = args.crf
    CODEC = args.codec

    with mp_face_detection.FaceDetection(model_selection=1,
                                         min_detection_confidence=0.5) as face_detector:
        make_face_crops(face_detector)
