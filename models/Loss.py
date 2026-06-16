"""
Frequency-domain focal loss for FSFVE training.

Weights the L1 loss on DCT coefficients by the inverse of the JPEG quantization
matrix, emphasizing low-frequency components which have the greatest impact on
perceptual quality. This accelerates convergence by directing the model's
attention to the most visually significant coefficients first.
"""

import torch
import torch.nn as nn


class LuminanceLoss(nn.Module):
    """
    Frequency-domain focal loss based on the JPEG quantization matrix.

    For each 8x8 DCT block, the per-coefficient L1 loss is divided by the
    corresponding JPEG quantization value. Since the quantization matrix assigns
    smaller values to low-frequency coefficients, this reciprocal weighting
    naturally amplifies their contribution to the total loss.

    Supports both RGB input (where the luminance quantization table is tiled
    across all three channels) and YCbCr input (where separate luminance and
    chrominance tables are used).

    Args:
        use_ycbcr: If True, apply separate chrominance quantization weights to
                   the Cb and Cr channels. If False, apply the luminance table
                   to all three channels.
        device: torch device to place the quantization table on.
    """

    # JPEG standard luminance quantization table (8x8, flattened)
    LUMINANCE_TABLE = [
        16, 11, 10, 16,  24,  40,  51,  61,
        12, 12, 14, 19,  26,  58,  60,  55,
        14, 13, 16, 24,  40,  57,  69,  56,
        14, 17, 22, 29,  51,  87,  80,  62,
        18, 22, 37, 56,  68, 109, 103,  77,
        24, 35, 55, 64,  81, 104, 113,  92,
        49, 64, 78, 87, 103, 121, 120, 101,
        72, 92, 95, 98, 112, 100, 103,  99,
    ]

    # JPEG standard chrominance quantization table (8x8, flattened)
    CHROMINANCE_TABLE = [
        17, 18, 24, 47, 99, 99, 99, 99,
        18, 21, 26, 66, 99, 99, 99, 99,
        24, 26, 56, 99, 99, 99, 99, 99,
        47, 66, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
    ]

    def __init__(self, use_ycbcr, device):
        super().__init__()
        yt = torch.tensor(self.LUMINANCE_TABLE, dtype=torch.float32).reshape(1, 64)
        ct = torch.tensor(self.CHROMINANCE_TABLE, dtype=torch.float32).reshape(1, 64)

        if use_ycbcr:
            # Y channel uses luminance table; Cb and Cr share the chrominance table
            table = torch.hstack((yt, ct.repeat(1, 2)))
        else:
            # All three RGB channels use the luminance table
            table = yt.repeat(1, 3)

        self.register_buffer('table', table.to(device))
        self.loss_fn = nn.L1Loss(reduction='none')

    def forward(self, model_output, ground_truth):
        """
        Compute the frequency-weighted L1 loss.

        Args:
            model_output: Predicted DCT coefficients, shape (B, N, 3*64).
            ground_truth: Target DCT coefficients, shape (B, N, 3*64).

        Returns:
            Scalar loss value.
        """
        loss = self.loss_fn(model_output, ground_truth)
        return (loss / self.table).mean()
