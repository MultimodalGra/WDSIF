import argparse
import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
from contextlib import contextmanager, redirect_stdout, redirect_stderr

import datasets
import utils
from models import DenoisingDiffusion, DiffusiveRestoration

from metrics.psnr_ssim import calculate_lpips
from metrics.SSIMMetric import SSIMMetric
from metrics.PSNRMetric import PSNRMetric
from metrics.fidmetric import FIDMetric


def dict2namespace(config):
    ns = argparse.Namespace()
    for k, v in config.items():
        setattr(ns, k, dict2namespace(v) if isinstance(v, dict) else v)
    return ns


def parse_args():
    p = argparse.ArgumentParser("Evaluate Diffusion Model (minimal)")
    p.add_argument("--config", default="UIEB.yml", type=str)
    p.add_argument("--resume", required=True, type=str)
    p.add_argument("--sampling_timesteps", type=int, default=10)
    p.add_argument("--image_folder", default="results/test", type=str)
    p.add_argument("--seed", type=int, default=230)

    p.add_argument("--save_images", action="store_true")

    return p.parse_args()


@contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield


def to_uint8_hwc(x01: torch.Tensor) -> np.ndarray:
    if x01.dim() == 4:
        x01 = x01[0]
    x = (x01.detach().clamp(0, 1).cpu() * 255.0).to(torch.uint8)
    return x.permute(1, 2, 0).numpy()


class Restorer(DiffusiveRestoration):

    pass


def main():
    args = parse_args()
    with open(os.path.join("configs", args.config), "r") as f:
        config = dict2namespace(yaml.safe_load(f))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.image_folder, exist_ok=True)
    save_dir = os.path.join(args.image_folder, "final_test_results")
    if args.save_images:
        os.makedirs(save_dir, exist_ok=True)

    with suppress_output():
        DATASET = datasets.__dict__[config.data.type](config)
        test_loader = DATASET.get_test_loader(parse_patches=False)

        diffusion = DenoisingDiffusion(args, config)
        diffusion.load_ddm_ckpt(args.resume, ema=True)
        diffusion.model.eval()

        restorer = Restorer(diffusion, args, config)

    psnr_metric = PSNRMetric()
    ssim_metric = SSIMMetric()
    fid_metric = FIDMetric(device="cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        psnr_metric = psnr_metric.cuda()
        ssim_metric = ssim_metric.cuda()

    psnr_list, ssim_list, lpips_list = [], [], []

    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(test_loader, desc="Eval")):
            x = x.to(device)
            x_cond = x[:, :3]
            x_gt = x[:, 3:]

            b, c, h, w = x_cond.shape

            hh = int(32 * np.ceil(h / 32.0))
            ww = int(32 * np.ceil(w / 32.0))
            x_pad = torch.nn.functional.pad(x_cond, (0, ww - w, 0, hh - h), "reflect")

            pred = restorer.diffusive_restoration(x_pad)[:, :, :h, :w].clamp(0, 1)

            if args.save_images:
                name = y[0] if isinstance(y[0], str) else f"img_{i:05d}"
                utils.logging.save_image(pred, os.path.join(save_dir, f"{name}.png"))

            psnr_list.append(psnr_metric(pred, x_gt).item())
            ssim_list.append(ssim_metric(pred, x_gt).item())

            pred_np = to_uint8_hwc(pred)
            gt_np = to_uint8_hwc(x_gt)
            try:
                lpips_list.append(
                    calculate_lpips(pred_np, gt_np, crop_border=0, input_order="HWC")
                )
            except:
                lpips_list.append(0.0)

            fid_metric.update_fake(pred[0].detach().cpu())
            fid_metric.update_real(x_gt[0].detach().cpu())

    fid = fid_metric.compute_fid()

    print(
        f"PSNR={np.mean(psnr_list):.2f}, "
        f"SSIM={np.mean(ssim_list):.4f}, "
        f"LPIPS={np.mean(lpips_list):.4f}, "
        f"FID={fid:.2f}"
    )


if __name__ == "__main__":
    main()
