import numpy as np
import torch
import torch.nn as nn

from skrt.layers.norm import VideoTemporalBatchNorm
from skrt.layers.attn import BertSelfAttention
from skrt.eval import get_keyshot_summ

import logging

logging.basicConfig(level=logging.DEBUG)

class SKRTSummarizer(nn.Module):
    def __init__(self):
        super().__init__()

        self.bn1 = VideoTemporalBatchNorm(num_channels=1024)

        self.attn1 = BertSelfAttention({'hidden_size': 1024,
                                        'num_of_attention_heads': 16})
        self.attn1_2 = BertSelfAttention({'hidden_size': 1024,
                                          'num_of_attention_heads': 16})
        self.proj1 = nn.Sequential(nn.Linear(in_features=1024, out_features=512),
                                   nn.Dropout(0.3),
                                   VideoTemporalBatchNorm(num_channels=512),
                                   nn.Linear(in_features=512, out_features=512),
                                   nn.Dropout(0.3),
                                   nn.ReLU())

        self.bn2 = VideoTemporalBatchNorm(num_channels=512)
        self.attn2 = BertSelfAttention({'hidden_size': 512,
                                        'num_of_attention_heads': 16})
        self.attn2_2 = BertSelfAttention({'hidden_size': 512,
                                          'num_of_attention_heads': 16})
        self.proj2 = nn.Sequential(nn.Linear(in_features=512, out_features=256),
                                   nn.Dropout(0.3),  # !
                                   VideoTemporalBatchNorm(num_channels=256),
                                   nn.Linear(in_features=256, out_features=256),
                                   nn.Dropout(0.3),  # !
                                   nn.ReLU())  # !

        self.bn3 = VideoTemporalBatchNorm(num_channels=256)
        self.attn3 = BertSelfAttention({'hidden_size': 256,
                                        'num_of_attention_heads': 16})
        self.attn3_2 = BertSelfAttention({'hidden_size': 256,
                                          'num_of_attention_heads': 16})
        self.proj3 = nn.Sequential(nn.Linear(in_features=256, out_features=128),
                                   nn.Dropout(0.3),
                                   VideoTemporalBatchNorm(num_channels=128),
                                   nn.Linear(in_features=128, out_features=128),
                                   nn.Dropout(0.3),
                                   nn.ReLU())

        self.bn4 = VideoTemporalBatchNorm(num_channels=128)
        self.attn4 = BertSelfAttention({'hidden_size': 128,
                                        'num_of_attention_heads': 16})
        self.proj4 = nn.Sequential(nn.Linear(in_features=128, out_features=64),
                                   nn.Dropout(0.3),
                                   VideoTemporalBatchNorm(num_channels=64),
                                   nn.Linear(in_features=64, out_features=64),
                                   nn.Dropout(0.3),
                                   nn.ReLU())

        self.lin1 = nn.Linear(in_features=64, out_features=64)
        self.dropout1 = nn.Dropout(0.5)
        self.act1 = nn.ReLU()
        self.lin2 = nn.Linear(in_features=64, out_features=1)
        self.dropout2 = nn.Dropout(0.5)
        self.act2 = nn.Sigmoid()
        self.pool = nn.MaxPool1d(kernel_size=5,
                                 stride=1,
                                 padding=2)

    def forward(self, x):
        identity = x
        o = self.bn1(x)
        o = self.attn1(x)
        o = self.attn1_2(x)
        o = identity + o
        o = self.proj1(o)

        identity = o
        o = self.bn2(o)
        o = self.attn2(o)
        o = self.attn2_2(o)
        o = identity + o
        o = self.proj2(o)

        identity = o
        o = self.bn3(o)
        o = self.attn3(o)
        o = self.attn3_2(o)
        o = identity + o
        o = self.proj3(o)

        identity = o
        o = self.bn4(o)
        o = self.attn4(o)
        o = identity + o
        o = self.proj4(o)

        o = self.lin1(o)
        o = self.act1(o)
        o = self.dropout1(o)

        o = self.lin2(o)
        o = self.act2(o)
        o = self.dropout2(o)
        o = self.pool(o)
        return o.view(1, -1)

    def predict(self,
                x: torch.Tensor,
                cps: np.ndarray,
                n_frames: int,
                nfps: np.ndarray,
                picks: np.ndarray,
                proportion: float = 0.15
                ) -> np.ndarray:
        logging.debug(f'x size: {x.size()}')
        preds = self(x).squeeze(0)
        preds = preds.cpu().numpy()
        logging.debug(f'preds size: {preds.shape}')
        pred_summ = get_keyshot_summ(pred=preds,
                                     cps=cps,
                                     n_frames=n_frames,
                                     nfps=nfps,
                                     picks=picks,
                                     proportion=proportion)
        return pred_summ, preds
