from os import PathLike
from typing import Union
import argparse


def get_parser():
    parser = argparse.ArgumentParser()

    # Dataset-related arguments
    parser.add_argument('--dataset', type=str, default='tvsum', choices=['tvsum', 'summe'])
    parser.add_argument('--mode', type=str, default='c', choices=['c', 'a', 't'])
    parser.add_argument('--base_dir', type=str, default='../data/splits')

    # Training related arguments
    parser.add_argument('--n_epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=420)
    parser.add_argument('--save_dir', default='../data/model')

    # Inference related arguments
    parser.add_argument('--input_video', type=str, default='')
    parser.add_argument('--model_ckpt', type=str, default='../model/data/tvsum_canonical/skrt_0.pth')
    parser.add_argument('--sampling_rate', type=int, default=15)
    parser.add_argument('--save_video_path', type=str, default='out.mp4')

    return parser


def get_config():
    parser = get_parser()
    cfg = parser.parse_args()
    return cfg


def lookup_split(split_name: str, mode: str) -> Union[str, PathLike]:
    key = split_name + mode
    splits = {
        'summec': 'summe.yml',
        'summea': 'summe_aug.yml',
        'summet': 'summe_trans.yml',
        'tvsumc': 'tvsum.yml',
        'tvsuma': 'tvsum_aug.yml',
        'tvsumt': 'tvsum_trans.yml'
    }

    return splits[key]
