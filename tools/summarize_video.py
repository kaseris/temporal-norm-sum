import sys
sys.path.append('../')

import cv2
import numpy as np
import torch

from skrt.layers.summarizer import SKRTSummarizer
from skrt.utils.parser import get_config
from skrt.utils import video_helper


def main():
    # Load the model and its parameters
    print('Loading model')
    cfg = get_config()
    model = SKRTSummarizer()
    model = model.eval().cuda()
    state_dict = torch.load(cfg.model_ckpt)
    model.load_state_dict(state_dict)

    # Preprocess the input video to run inference to
    print('Preprocessing video')
    video_proc = video_helper.VideoPreprocessor(cfg.sampling_rate)
    n_frames, features, cps, nfps, picks = video_proc.run(cfg.input_video)
    seq_len = len(features)

    # Run inference
    with torch.no_grad():
        seq_torch = torch.from_numpy(features).unsqueeze(0).cuda()
        pred_summ = model.predict(seq_torch, cps=cps,
                                  n_frames=n_frames,
                                  nfps=nfps,
                                  picks=picks,
                                  proportion=0.15)

    print('Writing summarized video')
    cap = cv2.VideoCapture(cfg.input_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(cfg.save_video_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if pred_summ[frame_idx]:
            out.write(frame)

        frame_idx += 1

    out.release()
    cap.release()


if __name__ == '__main__':
    main()
