"""
Real-time inference demo for FSFVE.

Reads a compressed video file frame by frame, detects the face using MediaPipe,
crops a 256x256 region around it, and enhances it using a trained FSFVE model.
The pipeline demonstrates the full process: crop → block decomposition →
2D DCT → MLP residual prediction → IDCT → reconstruct → display.

Set the configuration constants below before running.
"""

import time
import cv2 as cv
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import statistics
from torch.optim.swa_utils import AveragedModel
from torchvision.transforms.functional import to_tensor
from torch_dct import dct_2d, idct_2d
import kornia as K
from utils import cfg_from_log
from models.fsfve import FSFVE

# ---------------------------------------------------------------------------
# Configuration — set these before running
# ---------------------------------------------------------------------------
VIDEO_PATH = 'path/to/compressed_video.mp4'   # Input compressed video
CHECKPOINT_DIR = 'checkpoints/exp_name'        # Directory of the trained checkpoint
VIDEO_NAME = 'video_name'                      # Name of the video (used to find checkpoint)
EPOCHS = 100                                   # Number of epochs the checkpoint was trained for
DEVICE = 'cpu'
CROP_SIZE = 256                                # Face crop size in pixels
# ---------------------------------------------------------------------------

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_grad_enabled(False)

mp_face_detection = mp.solutions.face_detection

# Load config and model
cfg = cfg_from_log(f'{CHECKPOINT_DIR}/{VIDEO_NAME}.log')

unfold = nn.Unfold(kernel_size=8, stride=8)
fold = nn.Fold(output_size=CROP_SIZE, kernel_size=8, stride=8)

ema = cfg['train']['ema']
def ema_avg(averaged_model_parameter, model_parameter, num_averaged):
    return ema * averaged_model_parameter + (1 - ema) * model_parameter

model = AveragedModel(FSFVE(**cfg['model']['kwargs']), DEVICE, avg_fn=ema_avg)
model.load_state_dict(
    torch.load(f'{CHECKPOINT_DIR}/{VIDEO_NAME}/{EPOCHS}_mdl.pth',
               map_location=torch.device(DEVICE))
)
model.eval()
model = torch.jit.trace(model, torch.rand((1, 1024, 192)))

cap = cv.VideoCapture(VIDEO_PATH)

times = []
im_counter = 0

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.25) as face_detector:
    start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = face_detector.process(rgb_frame)
        frame_height, frame_width, _ = frame.shape

        if results.detections:
            face = results.detections[0]
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
            mid_x, mid_y = x + w // 2, y + h // 2
            size = CROP_SIZE // 2

            crop = frame[mid_y - size:mid_y + size, mid_x - size:mid_x + size]
            if crop.shape != (CROP_SIZE, CROP_SIZE, 3):
                im_counter += 1
                continue

            with torch.inference_mode():
                st = time.time()

                # Preprocess: BGR → RGB → tensor → blur → scale → DCT blocks
                lq = cv.cvtColor(crop, cv.COLOR_BGR2RGB)
                lq = to_tensor(lq).unsqueeze(0).to(DEVICE)
                ks = cfg['data']['use_blur']['kernel_size']
                sig = cfg['data']['use_blur']['sigma']
                lq = K.filters.gaussian_blur2d(lq, ks, (sig, sig))
                lq = lq * 2 - 1
                lq = unfold(lq).permute(0, 2, 1)
                lq = lq.reshape(1, -1, 3, 8, 8)
                lq = dct_2d(lq, norm='ortho').reshape(1, -1, 192)

                # Enhance
                pr = model(lq)

                # Postprocess: IDCT → fold blocks → rescale → uint8
                pr = pr.reshape(1, -1, 3, 8, 8)
                pr = idct_2d(pr, norm='ortho').reshape(1, -1, 192)
                pr = fold(pr.permute(0, 2, 1))
                pr = ((pr + 1) / 2).clamp(0, 1).squeeze(0)
                pr = pr.mul(255).permute(1, 2, 0).to(torch.uint8).cpu().numpy()
                pr = cv.cvtColor(pr, cv.COLOR_RGB2BGR)

                times.append(time.time() - st)

            # Replace the face region in the frame with the enhanced crop
            frame[mid_y - size:mid_y + size, mid_x - size:mid_x + size] = pr
            cv.imshow('FSFVE Enhanced', frame)

        key = cv.waitKey(1)
        if key == ord('q'):
            break

        im_counter += 1

    print(f'Total time: {time.time() - start:.2f}s')
    cap.release()
    cv.destroyAllWindows()

# Timing summary
if times:
    mean = statistics.mean(times)
    median = statistics.median(times)
    minv = min(times)
    maxv = max(times)
    stdv = statistics.pstdev(times, mean)
    print(f'\n--- Timing Summary ({len(times)} frames) ---')
    print(f'Mean:   {mean:.4f}s  ({1/mean:.1f} FPS)')
    print(f'Median: {median:.4f}s  ({1/median:.1f} FPS)')
    print(f'Min:    {minv:.4f}s  ({1/minv:.1f} FPS)')
    print(f'Max:    {maxv:.4f}s  ({1/maxv:.1f} FPS)')
    print(f'Std:    {stdv:.4f}s')
    print(f'Device: {DEVICE}')
