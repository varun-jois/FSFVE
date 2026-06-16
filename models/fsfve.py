"""
FSFVE model architecture.

An MLP with sine activations (SineLayer) operating in the DCT frequency domain.
The model predicts a residual that is added to the input, effectively learning
to undo compression artifacts patch by patch.
"""

import torch
import torch.nn as nn
import numpy as np


class SineLayer(nn.Module):
    """
    A linear layer followed by a sine activation.

    For the first layer (is_first=True), weights are initialized uniformly in
    [-1/in_features, 1/in_features] and the input is scaled by omega_0 before
    the sine. For subsequent layers, weights are scaled down by omega_0 to keep
    activation magnitudes stable while preserving gradient magnitude.

    See Sitzmann et al., "Implicit Neural Representations with Periodic
    Activation Functions" (NeurIPS 2020), Sec. 3.2 and Supplement Sec. 1.5.
    """

    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class FSFVE(nn.Module):
    """
    Few-Shot Compressed Face Video Enhancement (FSFVE) model.

    A multi-layer perceptron with sine activations that operates on vectorized
    DCT blocks of a face image. The network learns instance-specific residuals
    that restore detail lost during video compression, and can be trained on as
    few as 10-30 frames in under 100 seconds on a CPU.

    Args:
        in_features: Size of each input DCT block vector (3 * block_size^2).
        hidden_features: Number of hidden units per layer.
        hidden_layers: Number of hidden sine layers.
        out_features: Size of output vector (same as in_features).
        outermost_linear: If True, use a plain linear layer as the final layer.
        first_omega_0: Frequency scaling for the first sine layer.
        hidden_omega_0: Frequency scaling for all subsequent sine layers.
        dropout: Dropout probability (0 = disabled).
        norm: Normalization type — 'bn' for BatchNorm, 'ln' for LayerNorm, None for none.
    """

    def __init__(self, in_features, hidden_features, hidden_layers, out_features,
                 outermost_linear=False, first_omega_0=30, hidden_omega_0=30.,
                 dropout=0, norm=None):
        super().__init__()

        layers = [SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0)]
        if dropout:
            layers.append(nn.Dropout(dropout))
        if norm == 'bn':
            layers.append(nn.BatchNorm1d(15625))
        elif norm == 'ln':
            layers.append(nn.LayerNorm([1024, 512]))

        for _ in range(hidden_layers):
            layers.append(SineLayer(hidden_features, hidden_features,
                                    is_first=False, omega_0=hidden_omega_0))
            if dropout:
                layers.append(nn.Dropout(dropout))
            if norm == 'bn':
                layers.append(nn.BatchNorm1d(15625))
            elif norm == 'ln':
                layers.append(nn.LayerNorm([1024, 512]))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
            layers.append(final_linear)
        else:
            layers.append(SineLayer(hidden_features, out_features,
                                    is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass. Returns input plus predicted residual."""
        return self.net(x) + x
