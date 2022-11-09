from os import PathLike
from typing import Union

import numpy as np
import torch
import torch.nn as nn

from skrt.eval import evaluate, get_keyshot_summ, downsample_summ


def xavier_init(module):
    cls_name = module.__class__.__name__
    if 'Linear' in cls_name or 'Conv' in cls_name:
        nn.init.xavier_uniform_(module.weight, gain=np.sqrt(2.0))
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.1)


def train(model: nn.Module,
          train_loader,
          val_loader,
          save_path: Union[str, PathLike],
          n_epochs: int = 300):

    model = model.cuda()

    model.apply(xavier_init)

    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(params, lr=1e-4, weight_decay=1e-5)

    criterion = nn.MSELoss()
    criterion = criterion.cuda()

    per_epoch_loss = []
    per_epoch_fscore = []
    best_val_score = -1.0
    for epoch in range(n_epochs):
        model.train()

        per_step_loss = []
        for _, seq, gtscore, cps, n_frames, nfps, picks, _ in train_loader:
            keyshot_summ = get_keyshot_summ(gtscore, cps, n_frames, nfps, picks)
            target = downsample_summ(keyshot_summ)

            seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).cuda()

            pred_cls = model(seq)
            target = torch.tensor(gtscore).unsqueeze(0).cuda()
            loss = criterion(pred_cls, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            per_step_loss.append(loss.data.cpu().numpy())

        per_epoch_loss.append(np.mean(per_step_loss))

        val_score = evaluate(model=model, val_loader=val_loader, device='cuda')
        if val_score > best_val_score:
            best_val_score = val_score
            torch.save(model.state_dict(), save_path)
        per_epoch_fscore.append(val_score)
        print(f'Epoch: {epoch + 1}/{n_epochs}\tLoss: {np.mean(per_step_loss):.3f}\tF-Score: {val_score:.3f}')
    return best_val_score, per_epoch_loss, per_epoch_fscore
