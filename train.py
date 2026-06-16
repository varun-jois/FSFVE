"""
Training and inference script for FSFVE.

For each video in the dataset, trains an instance-specific FSFVE model on a
small set of clustered frames, then evaluates it on the remaining frames using
PSNR, SSIM, and LPIPS metrics. Results are saved to a CSV file.

Usage:
    python train.py -c config.yaml
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.swa_utils import AveragedModel
import logging
import numpy as np
from torchmetrics.image import (PeakSignalNoiseRatio,
    LearnedPerceptualImagePatchSimilarity, StructuralSimilarityIndexMeasure)
import pandas as pd
import sys
from torch_dct import dct_2d, idct_2d
from models.Loss import LuminanceLoss
import yaml
from argparse import ArgumentParser
from pathlib import Path
from glob import glob
from data.DFD import DFD


def load_config():
    """Parse command-line arguments and load the YAML config file."""
    parser = ArgumentParser()
    parser.add_argument('-c', '--cfg', required=True, help='Path to the config YAML file.')
    args = parser.parse_args()
    with open(args.cfg, 'r') as f:
        config = yaml.safe_load(f)
    return config


class Model:
    """
    Wraps an FSFVE model with training and inference logic for a single video instance.

    Training is instance-specific: a model is fit to a small set of frames from
    one video, then evaluated on the remaining frames of that video.
    """

    def __init__(self, cfg, vid_name):
        """
        Args:
            cfg: Config dictionary loaded from YAML.
            vid_name: Name of the video to train on.
        """
        self.cfg = cfg
        self.checkpt_path = f'checkpoints/{cfg["name"]}/{vid_name}'
        self.device = torch.device(cfg['train']['device'])
        self.vid_name = vid_name
        self._init_model()
        self.optim = torch.optim.RAdam(
            lr=cfg['train']['learning_rate'],
            params=self.model.parameters()
        )
        self.sched = MultiStepLR(self.optim, cfg['train']['scheduler'], cfg['train']['gamma'])
        self._load_model()

    def _init_model(self):
        """Instantiate the FSFVE model and log its parameter count."""
        from models.fsfve import FSFVE
        model = FSFVE(**self.cfg['model']['kwargs']).to(self.device)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f'Total params: {total_params:,}')
        self.model = model

        ema = self.cfg['train']['ema']
        if ema != 0:
            def ema_avg(averaged_model_parameter, model_parameter, num_averaged):
                return ema * averaged_model_parameter + (1 - ema) * model_parameter
            self.model_ema = AveragedModel(self.model, self.device, avg_fn=ema_avg)

    def _save_model(self):
        """Save model and optimizer state dicts to the checkpoint directory."""
        num = self.cfg['train']['epoch_start'] + self.cfg['train']['epochs']
        model = self.model_ema if self.cfg['train']['ema'] != 0 else self.model
        torch.save(model.state_dict(), f'{self.checkpt_path}/{num}_mdl.pth')
        torch.save(self.optim.state_dict(), f'{self.checkpt_path}/{num}_opt.pth')

    def _load_model(self):
        """Resume from a checkpoint if epoch_start > 0."""
        num = self.cfg['train']['epoch_start']
        if num == 0:
            return
        if self.cfg['train']['ema'] != 0:
            self.model_ema.load_state_dict(torch.load(f'{self.checkpt_path}/{num}_mdl.pth'))
            self.model.load_state_dict(self.model_ema.module.state_dict())
        else:
            self.model.load_state_dict(torch.load(f'{self.checkpt_path}/{num}_mdl.pth'))
        self.optim.load_state_dict(torch.load(f'{self.checkpt_path}/{num}_opt.pth'))
        self.sched.last_epoch = num

    def train(self):
        """
        Train the model on the instance-specific training frames.

        Loads all training frames into memory, converts them to DCT blocks,
        and runs the training loop. The model is saved after training completes.
        """
        train_data = DFD('train', self.vid_name, **self.cfg['data'])
        train_dataloader = DataLoader(train_data, batch_size=len(train_data))
        train_lq, train_hq = next(iter(train_dataloader))

        kernel_size = self.cfg['model']['kernel_size']
        stride = self.cfg['model']['stride']
        unfold = nn.Unfold(kernel_size=kernel_size, stride=stride)
        batch_size = train_lq.shape[0]
        train_lq = unfold(train_lq).permute(0, 2, 1)
        train_hq = unfold(train_hq).permute(0, 2, 1)

        if train_data.use_dct:
            train_lq = train_lq.reshape(batch_size, -1, 3, kernel_size, kernel_size)
            train_hq = train_hq.reshape(batch_size, -1, 3, kernel_size, kernel_size)
            train_lq = dct_2d(train_lq, norm='ortho').reshape(batch_size, -1, 3 * kernel_size ** 2)
            train_hq = dct_2d(train_hq, norm='ortho').reshape(batch_size, -1, 3 * kernel_size ** 2)

        loss_fn = LuminanceLoss(train_data.use_ycbcr, self.cfg['train']['device'])

        thp = self.cfg['train']
        batch_size = train_lq.shape[0] if thp['batch_size'] == -1 else thp['batch_size']
        epoch_start = thp['epoch_start']
        epochs = thp['epochs']

        if thp['batch_size'] == -1:
            steps_per_epoch = 1
        else:
            if train_lq.shape[0] % batch_size != 0:
                raise ValueError('batch_size must be a multiple of the training set size.')
            steps_per_epoch = train_lq.shape[0] // batch_size

        for epoch in range(epoch_start, epoch_start + epochs):
            epoch_loss = 0
            for step in range(steps_per_epoch):
                i = step * batch_size
                lq = train_lq[i:i + batch_size, ...].to(self.device)
                hq = train_hq[i:i + batch_size, ...].to(self.device)

                pr = self.model(lq)
                loss = loss_fn(pr, hq)

                if self.cfg['model']['reg_mul'] != 0:
                    reg_loss = sum(torch.sum(p ** 2) for p in self.model.parameters() if p.requires_grad)
                    loss = loss + self.cfg['model']['reg_mul'] * reg_loss

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                if self.cfg['train']['ema'] != 0:
                    self.model_ema.update_parameters(self.model)

                epoch_loss += loss.item()

            epoch_loss /= steps_per_epoch
            logging.info(f'Epoch {epoch} loss: {epoch_loss:.9f} lr: {self.sched.get_last_lr()[0]:.9f}')
            self.sched.step()

        self._save_model()

    def inference(self, save_path=None):
        """
        Run inference on all frames (train and held-out) and return metric scores.

        Args:
            save_path: If provided, enhanced frames are saved as PNGs under
                       save_path/train/ and save_path/inference/.

        Returns:
            dict with video name, frame count, and mean PSNR/SSIM/LPIPS for
            both the training frames and held-out inference frames.
        """
        metrics = {
            'psnr': PeakSignalNoiseRatio(data_range=1.0).to(self.device),
            'ssim': StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device),
            'lpips': LearnedPerceptualImagePatchSimilarity('alex', normalize=True).to(self.device)
        }

        kernel_size = self.cfg['model']['kernel_size']
        unfold = nn.Unfold(kernel_size=kernel_size, stride=kernel_size)
        fold = nn.Fold(output_size=256, kernel_size=kernel_size, stride=kernel_size)

        data_kwargs = self.cfg['data'].copy()
        data_kwargs.update({'augmentation': '', 'raw': True})
        train_data = DFD('train', self.vid_name, **data_kwargs)
        valid_data = DFD('inference', self.vid_name, **data_kwargs)
        train_dataloader = DataLoader(train_data, batch_size=1)
        valid_dataloader = DataLoader(valid_data, batch_size=1)

        final_scores = {'vid': self.vid_name, 'frames': 0}
        model = self.model_ema if self.cfg['train']['ema'] != 0 else self.model
        model.eval()

        with torch.no_grad():
            for mode in ['train', 'inference']:
                dataset = train_data if mode == 'train' else valid_data
                dataloader = train_dataloader if mode == 'train' else valid_dataloader
                mets = {f'{mode}_{m}': [] for m in ['psnr', 'ssim', 'lpips']}

                for i, (lq, hq) in enumerate(dataloader):
                    lq = lq.to(self.device)
                    batch_size = lq.shape[0]

                    lq = unfold(lq).permute(0, 2, 1)
                    if dataset.use_dct:
                        lq = lq.reshape(batch_size, -1, 3, kernel_size, kernel_size)
                        lq = dct_2d(lq, norm='ortho').reshape(batch_size, -1, 3 * kernel_size ** 2)

                    pr = self.model(lq)

                    if dataset.use_dct:
                        pr = pr.reshape(batch_size, -1, 3, kernel_size, kernel_size)
                        pr = idct_2d(pr, norm='ortho').reshape(batch_size, -1, 3 * kernel_size ** 2)
                    pr = fold(pr.permute(0, 2, 1))

                    pr = dataset.post_process(pr)

                    if save_path is not None:
                        save_image(pr.squeeze(0), f'{save_path}/{mode}/{i:04d}.png')

                    hq = hq.to(self.device)
                    for m in metrics:
                        mets[f'{mode}_{m}'].append(metrics[m](pr, hq).item())

                for k, v in mets.items():
                    final_scores[k] = np.mean(np.array(v))

                if mode == 'inference':
                    final_scores['frames'] = len(v)

                for m in mets:
                    logging.info(f'Avg score for {mode} ({len(v)} frames): {final_scores[m]:.7f}')

        return final_scores


def main(cfg):
    """Set up logging and run training + inference for every video in the dataset."""
    checkpt_path = f'checkpoints/{cfg["name"]}'
    log = f'{checkpt_path}/{cfg["name"]}.log'
    Path(checkpt_path).mkdir(parents=True, exist_ok=True)
    Path(log).touch(exist_ok=True)

    file_handler = logging.FileHandler(filename=log)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', handlers=[file_handler, stdout_handler])
    logging.info(cfg)

    dpath = cfg['data']['path']
    crf = cfg['data']['crf']
    scores = []

    for folder in sorted(glob(f'{dpath}/{crf}/*')):
        vid_name = Path(folder).stem
        if vid_name.startswith('clusters'):
            continue

        logging.info('-' * 30)
        logging.info(f'Training {vid_name}')

        checkpt_path = f'checkpoints/{cfg["name"]}/{vid_name}'
        Path(checkpt_path).mkdir(parents=True, exist_ok=True)

        model = Model(cfg, vid_name)
        model.train()
        scores.append(model.inference())

    df = pd.DataFrame(scores).sort_values(by='vid')
    mean_row = pd.Series(['Avg'] + df.iloc[:, 1:].mean().tolist(), index=df.columns)
    df = pd.concat([df, mean_row.to_frame().T], ignore_index=True)
    Path('scores').mkdir(parents=True, exist_ok=True)
    df.to_csv(f'scores/{cfg["name"]}.csv', index=False)


if __name__ == '__main__':
    cfg = load_config()
    main(cfg)
