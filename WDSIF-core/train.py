import argparse
import os
import yaml
import sys
import torch
import numpy as np
import datasets
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from models import DenoisingDiffusion


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        setattr(
            namespace, key, dict2namespace(value) if isinstance(value, dict) else value
        )
    return namespace


def parse_args_and_config():
    parser = argparse.ArgumentParser(
        description="Training Wavelet-Based Diffusion Model"
    )

    parser.add_argument("--config", default="UIEB.yml", type=str)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--sampling_timesteps", type=int, default=10)
    parser.add_argument("--image_folder", default="results/", type=str)
    parser.add_argument("--seed", default=230, type=int)

    parser.add_argument("--exp_name", type=str, default="")

    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        cfg = yaml.safe_load(f)
    config = dict2namespace(cfg)
    return args, config


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield


def main():
    args, config = parse_args_and_config()

    config.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    if args.exp_name:
        args.image_folder = os.path.join(args.image_folder, args.exp_name)
        config.data.ckpt_dir = os.path.join(config.data.ckpt_dir, args.exp_name)
    os.makedirs(args.image_folder, exist_ok=True)
    os.makedirs(config.data.ckpt_dir, exist_ok=True)
    with suppress_output():
        DATASET = datasets.__dict__[config.data.type](config)
        diffusion = DenoisingDiffusion(args, config)
    diffusion.train(DATASET)


if __name__ == "__main__":
    main()
