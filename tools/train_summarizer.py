import os
# test code --- TO BE AMENDED
import sys
sys.path.append('../')

import numpy as np
import torch

from skrt.dataset.dataset import VideoDataset, DataLoader, load_yaml
from skrt.layers.summarizer import SKRTSummarizer
from skrt.train import train
from skrt.utils.parser import get_config, lookup_split


cfg = get_config()

torch.manual_seed(420)

vals_split = []

losses = dict()
fscores_dict = dict()

split = lookup_split(cfg.dataset, cfg.mode)
splits_file = os.path.join(cfg.base_dir, split)
splits = load_yaml(splits_file)

model = SKRTSummarizer()

for split_idx, split in enumerate(splits):
    print(f'Train on split: {split_idx}')
    train_dataset = VideoDataset(split['train_keys'])
    train_loader = DataLoader(train_dataset, shuffle=True)

    val_set = VideoDataset(split['test_keys'])
    val_loader = DataLoader(val_set, shuffle=False)
    save_path = os.path.join(cfg.save_dir, f'skrt_{split_idx}.pth')
    val_score, loss, fscores = train(model, train_loader, val_loader, n_epochs=cfg.n_epochs, save_path=save_path)
    vals_split.append(val_score)
    losses[f'split{str(split_idx)}'] = loss
    fscores_dict[f'split{str(split_idx)}'] = fscores

print('Training complete.')
print(f'F-Score: {np.mean(vals_split):.4f}')