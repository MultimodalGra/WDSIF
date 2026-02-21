import os
import time
from typing import final
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import utils
from models.unet import DiffusionUNet
from models.wavelet import DWT, IWT
from pytorch_msssim import ssim
from models.HFB import HighFrequencyEnhancementBranch
from models.RGBTrans import I2IM
from models.FM import FM


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, : h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, : w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                    1.0 - self.mu
                ) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(
                inner_module.config.device
            )
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Net(nn.Module):
    def __init__(self, args, config):
        super(Net, self).__init__()

        self.args = args
        self.config = config
        self.device = config.device

        self.high_enhance0 = HighFrequencyEnhancementBranch(
            input_channels=3,
            dim=32,
            num_blocks=[2, 3, 3, 4],
            heads=[1, 2, 4, 8],
            ffn_factor=2.0,
            bias=False,
            LayerNorm_type="WithBias",
        )

        self.high_enhance1 = HighFrequencyEnhancementBranch(
            input_channels=3,
            dim=32,
            num_blocks=[2, 3, 3, 4],
            heads=[1, 2, 4, 8],
            ffn_factor=2.0,
            bias=False,
            LayerNorm_type="WithBias",
        )

        self.Unet = DiffusionUNet(config)

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        self.betas = torch.from_numpy(betas).float()
        self.num_timesteps = self.betas.shape[0]

        self.I2IM = I2IM()
        self.FM = FM()

    @staticmethod
    def compute_alpha(beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def sample_training(self, x_cond, b, eta=0.0):
        skip = (
            self.config.diffusion.num_diffusion_timesteps
            // self.args.sampling_timesteps
        )
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        n, c, h, w = x_cond.shape
        seq_next = [-1] + list(seq[:-1])
        x = torch.randn(n, c, h, w, device=self.device)
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = self.compute_alpha(b, t.long())
            at_next = self.compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)

            et = self.Unet(torch.cat([x_cond, xt], dim=1), t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1**2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to(x.device))

        return xs[-1]

    def forward(self, x):
        data_dict = {}
        dwt, idwt = DWT(), IWT()

        input_img = x[:, :3, :, :]
        n, c, h, w = input_img.shape
        input_img_norm = data_transform(input_img)
        input_dwt = dwt(input_img_norm)

        input_LL, input_high0 = input_dwt[:n, ...], input_dwt[n:, ...]

        input_high0 = self.high_enhance0(input_high0)

        input_LL_dwt = dwt(input_LL)
        input_LL_LL, input_high1 = input_LL_dwt[:n, ...], input_LL_dwt[n:, ...]

        input_high1 = self.high_enhance1(input_high1)

        b = self.betas.to(input_img.device)

        t = torch.randint(
            low=0, high=self.num_timesteps, size=(input_LL_LL.shape[0] // 2 + 1,)
        ).to(self.device)
        t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[
            : input_LL_LL.shape[0]
        ].to(x.device)
        a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)

        e = torch.randn_like(input_LL_LL)

        I2IM_output = self.I2IM(input_img)

        if self.training:
            gt_img_norm = data_transform(x[:, 3:, :, :])
            gt_dwt = dwt(gt_img_norm)
            gt_LL, gt_high0 = gt_dwt[:n, ...], gt_dwt[n:, ...]

            gt_LL_dwt = dwt(gt_LL)
            gt_LL_LL, gt_high1 = gt_LL_dwt[:n, ...], gt_LL_dwt[n:, ...]

            x_noise = gt_LL_LL * a.sqrt() + e * (1.0 - a).sqrt()
            noise_output = self.Unet(
                torch.cat([input_LL_LL, x_noise], dim=1), t.float()
            )
            denoise_LL_LL = self.sample_training(input_LL_LL, b)

            pred_LL = idwt(torch.cat((denoise_LL_LL, input_high1), dim=0))
            pred_x_raw = idwt(torch.cat((pred_LL, input_high0), dim=0))
            diffusion_output = inverse_data_transform(pred_x_raw)

            final_output = self.FM(input_img, I2IM_output, diffusion_output)

            data_dict["input_high0"] = input_high0
            data_dict["input_high1"] = input_high1
            data_dict["gt_high0"] = gt_high0
            data_dict["gt_high1"] = gt_high1
            data_dict["pred_LL"] = pred_LL
            data_dict["gt_LL"] = gt_LL
            data_dict["noise_output"] = noise_output
            data_dict["I2IM_output"] = I2IM_output
            data_dict["diffusion_output"] = diffusion_output
            data_dict["pred_x"] = final_output
            data_dict["e"] = e

        else:
            denoise_LL_LL = self.sample_training(input_LL_LL, b)
            pred_LL = idwt(torch.cat((denoise_LL_LL, input_high1), dim=0))
            pred_x_raw = idwt(torch.cat((pred_LL, input_high0), dim=0))
            diffusion_output = inverse_data_transform(pred_x_raw)

            final_output = self.FM(input_img, I2IM_output, diffusion_output)

            data_dict["pred_x"] = final_output
            data_dict["I2IM_output"] = I2IM_output
            data_dict["diffusion_output"] = diffusion_output

        return data_dict


class DenoisingDiffusion(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        self.model = Net(args, config)
        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model)

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.l2_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
        self.TV_loss = TVLoss()

        self.optimizer, self.scheduler = utils.optimize.get_optimizer(
            self.config, self.model.parameters()
        )
        self.start_epoch, self.step = 0, 0
        self.best_metric = 0.0

    def calculate_val_metrics(self, val_loader):
        from metrics.psnr_ssim import calculate_psnr, calculate_ssim

        self.model.eval()
        psnr_list, ssim_list = [], []

        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                x = x.to(self.device)
                b, _, img_h, img_w = x.shape
                img_h_32 = int(32 * np.ceil(img_h / 32.0))
                img_w_32 = int(32 * np.ceil(img_w / 32.0))
                x = F.pad(x, (0, img_w_32 - img_w, 0, img_h_32 - img_h), "reflect")

                out = self.model(x)
                pred_x = out["pred_x"][:, :, :img_h, :img_w]
                gt_img = x[:, 3:, :, :][:, :, :img_h, :img_w]

                pred_np = (
                    pred_x.squeeze().permute(1, 2, 0).cpu().numpy() * 255
                ).astype(np.uint8)
                gt_np = (gt_img.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(
                    np.uint8
                )

                try:
                    psnr = calculate_psnr(pred_np, gt_np, 0, "HWC")
                    ssim_val = calculate_ssim(pred_np, gt_np, 0, "HWC")
                    psnr_list.append(psnr)
                    ssim_list.append(ssim_val)
                except:
                    continue

        if psnr_list and ssim_list:
            return np.mean(psnr_list), np.mean(ssim_list)
        return 0.0, 0.0

    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.ema_helper.load_state_dict(checkpoint["ema_helper"])
        if ema:
            self.ema_helper.ema(self.model)
        print("=> loaded checkpoint {} step {}".format(load_path, self.step))

    def train(self, DATASET):
        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()

        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)

        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print("epoch: ", epoch)
            data_start = time.time()

            for i, (x, y) in enumerate(train_loader):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                self.model.train()
                self.step += 1

                x = x.to(self.device)
                output = self.model(x)
                noise_loss, photo_loss, frequency_loss = self.estimation_loss(x, output)
                loss = noise_loss + photo_loss + frequency_loss

                if self.step % 10 == 0:
                    print(
                        "step:{}, lr:{:.6f}, noise_loss:{:.4f}, photo_loss:{:.4f}, frequency_loss:{:.4f}".format(
                            self.step,
                            self.scheduler.get_last_lr()[0],
                            noise_loss.item(),
                            photo_loss.item(),
                            frequency_loss.item(),
                        )
                    )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_helper.update(self.model)
                data_start = time.time()

                if (
                    self.step % self.config.training.validation_freq == 0
                    and self.step != 0
                ):
                    avg_psnr, avg_ssim = self.calculate_val_metrics(val_loader)
                    combined_metric = avg_psnr + 100 * avg_ssim

                    print(
                        f"Validation - PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}, Combined: {combined_metric:.2f}"
                    )

                    checkpoint_data = {
                        "step": self.step,
                        "epoch": epoch + 1,
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.state_dict(),
                        "ema_helper": self.ema_helper.state_dict(),
                        "params": self.args,
                        "config": self.config,
                        "best_metric": self.best_metric,
                    }

                    utils.logging.save_checkpoint(
                        checkpoint_data,
                        filename=os.path.join(
                            self.config.data.ckpt_dir, "model_latest"
                        ),
                    )

            self.scheduler.step()

    def estimation_loss(self, x, output):
        input_high0, input_high1, gt_high0, gt_high1 = (
            output["input_high0"],
            output["input_high1"],
            output["gt_high0"],
            output["gt_high1"],
        )

        pred_LL, gt_LL, pred_x, noise_output, e = (
            output["pred_LL"],
            output["gt_LL"],
            output["pred_x"],
            output["noise_output"],
            output["e"],
        )

        I2IM_output = output["I2IM_output"]
        diffusion_output = output["diffusion_output"]
        gt_img = x[:, 3:, :, :].to(self.device)

        noise_loss = self.l2_loss(noise_output, e)

        frequency_loss = 0.1 * (
            self.l2_loss(input_high0, gt_high0)
            + self.l2_loss(input_high1, gt_high1)
            + self.l2_loss(pred_LL, gt_LL)
        ) + 0.018 * (
            self.TV_loss(input_high0)
            + self.TV_loss(input_high1)
            + self.TV_loss(pred_LL)
        )

        final_content_loss = self.l1_loss(pred_x, gt_img)
        final_ssim_loss = 1 - ssim(pred_x, gt_img, data_range=1.0).to(self.device)

        diffusion_content_loss = self.l1_loss(diffusion_output, gt_img)
        diffusion_ssim_loss = 1 - ssim(diffusion_output, gt_img, data_range=1.0).to(
            self.device
        )

        photo_loss = 0.7 * (final_content_loss + final_ssim_loss) + 0.5 * (
            diffusion_content_loss + diffusion_ssim_loss
        )

        return noise_loss, photo_loss, frequency_loss

    def sample_validation_patches(self, val_loader, step):
        image_folder = os.path.join(
            self.args.image_folder,
            self.config.data.type + str(self.config.data.patch_size),
        )
        self.model.eval()
        with torch.no_grad():
            print(f"Processing a single batch of validation images at step: {step}")
            for i, (x, y) in enumerate(val_loader):
                b, _, img_h, img_w = x.shape
                img_h_32 = int(32 * np.ceil(img_h / 32.0))
                img_w_32 = int(32 * np.ceil(img_w / 32.0))
                x = F.pad(x, (0, img_w_32 - img_w, 0, img_h_32 - img_h), "reflect")

                out = self.model(x.to(self.device))
                pred_x = out["pred_x"]
                pred_x = pred_x[:, :, :img_h, :img_w]
                utils.logging.save_image(
                    pred_x, os.path.join(image_folder, str(step), f"{y[0]}.png")
                )
