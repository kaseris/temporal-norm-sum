import sys
sys.path.append('../')

import cv2
import numpy as np
import torch

import h5py

from skrt.layers.summarizer import SKRTSummarizer
from skrt.utils.parser import get_config
from skrt.utils import video_helper
from skrt.dataset.dataset import VideoDataset


def main():
    vid = 'video_50'
    with h5py.File('/media/storage/skrt/datasets/eccv16_dataset_tvsum_google_pool5.h5', 'r') as f:
        features = np.asarray(f[vid]['features']).astype(np.float)
        gtscore = np.asarray(f[vid]['gtscore']).astype(np.float)
        cps = np.asarray(f[vid]['change_points']).astype(np.int32)
        n_frames = np.asarray(f[vid]['n_frames']).astype(np.int32)
        nfps = np.asarray(f[vid]['n_frame_per_seg']).astype(np.int32)
        picks = np.asarray(f[vid]['picks']).astype(np.int32)

    # Load the model and its parameters
    print('Loading model')
    cfg = get_config()
    model = SKRTSummarizer()
    model = model.eval().cuda()
    state_dict = torch.load(cfg.model_ckpt)
    model.load_state_dict(state_dict)

    # Preprocess the input video to run inference to
    print('Preprocessing video')
    # video_proc = video_helper.VideoPreprocessor(cfg.sampling_rate)
    # n_frames, features, cps, nfps, picks = video_proc.run(cfg.input_video)
    seq_len = len(features)
    print(f'features size: {seq_len}')
    # Run inference
    with torch.no_grad():
        seq_torch = torch.tensor(features, dtype=torch.float32).unsqueeze(0).cuda()
        pred_summ, preds = model.predict(seq_torch, cps=cps,
                                         n_frames=n_frames,
                                         nfps=nfps,
                                         picks=picks,
                                         proportion=0.05)
    print(f'preds size: {preds}')
    print(f'predicted summary: {np.where(pred_summ)[0].shape}')

    import matplotlib.pyplot as plt

    preds_norm = (preds - preds.min()) / (preds.max() - preds.min())
    gtscore_norm = (gtscore - gtscore.min()) / (gtscore.max() - gtscore.min())
    plt.plot(preds_norm)
    plt.plot(gtscore_norm)
    plt.show()


if __name__ == '__main__':
    main()
