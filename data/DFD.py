"""
Dataset class for the DeepFakeDetection (DFD) dataset.

Loads paired low-quality (compressed) and high-quality face image crops.
Training frames are selected via k-means clustering for diversity; the
remaining frames are used for inference. Supports optional YCbCr conversion,
pixel scaling, and various blur augmentations via kornia.
"""

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
from glob import glob
import kornia as K
import pickle
from models.augmentation import flip, rotate


def rgb_to_ycbcr(rgb):
    """
    Convert an RGB image tensor to YCbCr.

    Args:
        rgb: Tensor of shape (C, H, W) in [0, 1] range.

    Returns:
        YCbCr tensor of shape (C, H, W) in [0, 1] range.
    """
    matrix = torch.tensor([
        [ 0.299,      0.587,      0.114    ],
        [-0.168736,  -0.331264,   0.5      ],
        [ 0.5,       -0.418688,  -0.081312 ]
    ], dtype=torch.float32, device=rgb.device)

    rgb_flat = rgb.permute(1, 2, 0).contiguous().view(-1, 3)
    ycbcr_flat = torch.mm(rgb_flat, matrix.t())
    ycbcr_flat[:, 1:] += 0.5
    return ycbcr_flat.view(rgb.shape[1], rgb.shape[2], 3).permute(2, 0, 1)


def ycbcr_to_rgb(ycbcr):
    """
    Convert a YCbCr image tensor to RGB.

    Args:
        ycbcr: Tensor of shape (C, H, W), (B, C, H, W), or
               (B, patches, H, W, 3) in [0, 1] range.

    Returns:
        RGB tensor in [0, 1] range, same shape as input.
    """
    matrix = torch.tensor([
        [1.0,  0.0,       1.402     ],
        [1.0, -0.344136, -0.714136  ],
        [1.0,  1.772,     0.0       ]
    ], dtype=torch.float32, device=ycbcr.device)

    if len(ycbcr.shape) == 3:
        ycbcr_flat = ycbcr.permute(1, 2, 0).contiguous().view(-1, 3)
        ycbcr_flat[:, 1:] -= 0.5
        rgb_flat = torch.mm(ycbcr_flat, matrix.t())
        return rgb_flat.view(ycbcr.shape[1], ycbcr.shape[2], 3).permute(2, 0, 1)
    elif len(ycbcr.shape) == 4:
        batch_size = ycbcr.shape[0]
        ycbcr_flat = ycbcr.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 3)
        ycbcr_flat[:, :, 1:] -= 0.5
        rgb_flat = torch.bmm(ycbcr_flat, matrix.expand(batch_size, -1, -1).permute(0, 2, 1))
        return rgb_flat.view(batch_size, ycbcr.shape[2], ycbcr.shape[3], 3).permute(0, 3, 1, 2)
    else:
        ycbcr[..., 1:] -= 0.5
        return torch.matmul(ycbcr, matrix.t())


class DFD(Dataset):
    """
    Dataset of paired low-quality and high-quality face image crops from the
    DeepFakeDetection dataset.

    Args:
        mode: One of 'train', 'valid', or 'inference'.
        vid_name: Name of the video subdirectory to load.
        **kwargs: Dataset options — see __init__ for details.
    """

    def __init__(self, mode, vid_name, **kwargs):
        super().__init__()
        self.mode = mode
        self.vid_name = vid_name
        self.path = kwargs.get('path', '')
        self.crf = kwargs.get('crf', 40)
        self.clusters = kwargs.get('clusters', 30)
        self.cluster_frames = kwargs.get('cluster_frames', 480)
        self.use_ycbcr = kwargs.get('use_ycbcr', False)
        self.use_dct = kwargs.get('use_dct', True)
        self.scale = kwargs.get('scale', True)
        self.use_blur = kwargs.get('use_blur', {})
        self.augmentation = kwargs.get('augmentation', '')
        self.augment_mul = kwargs.get('augment_mul', 0)
        self.valid_size = kwargs.get('valid_size', 1)
        self.raw = kwargs.get('raw', False)
        self.data = self._load_cluster()

    def _load_cluster(self):
        """
        Load frame indices for the current mode.

        Training frames are the cluster centers chosen by k-means. Inference
        frames are all frames not in the training set (after the first
        cluster_frames frames).
        """
        with open(f'{self.path}/clusters_{self.cluster_frames}_{self.clusters}.pck', 'rb') as f:
            clusters = pickle.load(f)

        train_data = set(clusters[self.vid_name])

        if self.mode == 'train':
            return [f'{i:04d}' for i in clusters[self.vid_name]]

        vid_count = len(glob(f'{self.path}/{self.crf}/{self.vid_name}/hq/*'))
        start = 0 if self.cluster_frames >= vid_count else self.cluster_frames

        if self.mode == 'valid':
            return [f'{i:04d}' for i in range(start, vid_count, self.valid_size)]
        else:  # inference
            return [f'{i:04d}' for i in range(start, vid_count) if i not in train_data]

    def __len__(self):
        if self.mode == 'train' and self.augmentation == 'affine':
            return int(self.clusters * self.augment_mul)
        elif self.mode == 'train' and self.augmentation == 'hflip':
            return len(self.data) * 2
        return len(self.data)

    def __getitem__(self, idx):
        i = idx % len(self.data)
        lq = Image.open(f'{self.path}/{self.crf}/{self.vid_name}/lq/{self.data[i]}.png')
        hq = Image.open(f'{self.path}/{self.crf}/{self.vid_name}/hq/{self.data[i]}.png')

        if self.mode == 'train' and self.augmentation == 'affine':
            hq, lq = flip(hq, lq)
            hq, lq = rotate(hq, lq)
            params = T.RandomAffine.get_params(
                degrees=[-1, 1], translate=[0.01, 0.01],
                scale_ranges=[0.99, 1.01], shears=None, img_size=[256, 256]
            )
            hq = T.functional.affine(hq, *params, interpolation=Image.BICUBIC)
            lq = T.functional.affine(lq, *params, interpolation=Image.BICUBIC)

        lq = T.functional.to_tensor(lq)
        hq = T.functional.to_tensor(hq)

        if self.mode == 'train' and self.augmentation == 'hflip' and idx >= len(self.data):
            lq = T.functional.hflip(lq)
            hq = T.functional.hflip(hq)

        if self.use_blur:
            lq = lq.unsqueeze(0)
            ks = self.use_blur.get('kernel_size', 9)
            sig = self.use_blur.get('sigma', 1.7)
            blur_type = self.use_blur['type']
            if blur_type == 'gaussian':
                lq = K.filters.gaussian_blur2d(lq, ks, (sig, sig))
            elif blur_type == 'median':
                lq = K.filters.median_blur(lq, ks)
            elif blur_type == 'box':
                lq = K.filters.box_blur(lq, ks)
            elif blur_type == 'unsharp':
                lq = K.filters.unsharp_mask(lq, ks, (sig, sig))
            lq = lq.squeeze(0)

        if self.use_ycbcr:
            lq = K.color.rgb_to_ycbcr(lq.unsqueeze(0)).squeeze(0)
            if not self.raw:
                hq = K.color.rgb_to_ycbcr(hq.unsqueeze(0)).squeeze(0)

        if self.scale:
            lq = lq * 2 - 1
            if not self.raw:
                hq = hq * 2 - 1

        return lq, hq

    def post_process(self, pr):
        """Undo scaling and YCbCr conversion, and clamp to [0, 1]."""
        if self.scale:
            pr = (pr + 1) / 2
        if self.use_ycbcr:
            pr = K.color.ycbcr_to_rgb(pr)
        return pr.clamp(0, 1)
