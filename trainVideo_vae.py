import os
import time
import random
import argparse

from typing import Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.videoAudioClip_dataset import VideoAudioClipDataset
from configs.video_vae_config import TrainVideoVAEConfig
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module=r"torchaudio(\..*)?")

try:
    import imageio
    _HAS_IMAGEIO = True
except ImportError:
    _HAS_IMAGEIO = False

# IMPORTANT: use the repo's VAE implementation
# from wan.modules.vae2_2 import WanVAE_   # if your PYTHONPATH is set
from models.VAE.vae2_2 import WanVAE_  # assumes you run from repo root: Wan2.2/

import logging
from logging.handlers import RotatingFileHandler

def setup_logging(out_dir: str | None, name: str = "train") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # 防止重复打印

    # 避免在 notebook 多次运行时重复加 handler
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (可选)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        fh = RotatingFileHandler(
            filename=os.path.join(out_dir, "train.log"),
            maxBytes=20 * 1024 * 1024,  # 20MB
            backupCount=3,
            encoding="utf-8",
        )
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger




def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_model(cfg: TrainVideoVAEConfig, device: torch.device) -> nn.Module:
    # WanVAE_ forward(x, scale=[0,1]) returns (x_recon, mu). :contentReference[oaicite:2]{index=2}
    model = WanVAE_(
        dim=cfg.dim,
        dec_dim=cfg.dec_dim,
        z_dim=cfg.z_dim,
        dim_mult=list(cfg.dim_mult),
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=list(cfg.temperal_downsample),
        dropout=0.0,
    )
    model.to(device)
    return model


def save_ckpt(path: str, model: nn.Module, optim: torch.optim.Optimizer, step: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optim": optim.state_dict(),
        },
        path,
    )


def save_video(video: np.ndarray, save_path: str, fps: int = 5, value_range: Tuple[float, float] = (-1, 1)):
    """
    Save video from numpy array to mp4 file.
    
    Args:
        video: numpy array with shape (B, C, T, H, W) or (C, T, H, W) or (T, H, W, C)
               Values should be in value_range
        save_path: path to save the video file
        fps: frames per second for the output video
        value_range: (min, max) range of input values, default (-1, 1)
    """
    if not _HAS_IMAGEIO:
        raise ImportError("imageio is required for save_video. Install with: pip install imageio imageio-ffmpeg")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    # Convert to numpy if it's a tensor
    if isinstance(video, torch.Tensor):
        video = video.cpu().numpy()
    
    # Handle different input shapes
    if video.ndim == 5:
        # (B, C, T, H, W) -> take first batch and convert to (T, H, W, C)
        video = video[0]  # Take first batch
        video = np.transpose(video, (1, 2, 3, 0))  # (C, T, H, W) -> (T, H, W, C)
    elif video.ndim == 4:
        if video.shape[0] == 3 or video.shape[0] == 1:
            # (C, T, H, W) -> (T, H, W, C)
            video = np.transpose(video, (1, 2, 3, 0))
        # else assume (T, H, W, C) already
    else:
        raise ValueError(f"Unsupported video shape: {video.shape}. Expected (B,C,T,H,W), (C,T,H,W), or (T,H,W,C)")
    
    # Normalize from value_range to [0, 1]
    min_val, max_val = value_range
    video = np.clip(video, min_val, max_val)
    video = (video - min_val) / (max_val - min_val)  # [min_val, max_val] -> [0, 1]
    
    # Convert to [0, 255] uint8
    video = (video * 255).astype(np.uint8)
    
    # Handle grayscale (single channel)
    if video.shape[-1] == 1:
        video = np.repeat(video, 3, axis=-1)  # Convert to RGB
    
    # Write video using imageio
    try:
        writer = imageio.get_writer(
            save_path, fps=fps, codec="libx264", quality=8, format="FFMPEG", macro_block_size=None
        )

        for frame in video:
            writer.append_data(frame)
        writer.close()
    except Exception as e:
        raise RuntimeError(f"Failed to save video to {save_path}: {str(e)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir_video", type=str, default=None)
    parser.add_argument("--out_dir_audio", type=str, default=None)
    args = parser.parse_args()

    config_kwargs = {}

    if args.out_dir_video is not None:
        config_kwargs['out_dir_video'] = args.out_dir_video
    if args.out_dir_audio is not None:
        config_kwargs['out_dir_audio'] = args.out_dir_audio 
    cfg = TrainVideoVAEConfig(**config_kwargs)

    logger = setup_logging(cfg.out_dir_video, name="video_vae")
    logger.info("===== Wan2.2 VAE Training =====")
    logger.info(f"device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    logger.info(f"out_dir_video: {cfg.out_dir_video}")
    logger.info(f"out_dir_audio: {cfg.out_dir_audio}")
    logger.info(f"video root: {getattr(cfg, 'video_root', None)}")
    logger.info(f"audio root: {getattr(cfg, 'audio_root', None)}")
    logger.info(f"batch_size={cfg.batch_size}, num_workers={cfg.num_workers}, amp={cfg.amp_dtype}")

    seed_everything(cfg.seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = VideoAudioClipDataset(cfg)
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    it = iter(dl)

    model = make_model(cfg, device)
    model.train()

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    start_step = 0
    if cfg.out_dir_video:
        os.makedirs(cfg.out_dir_video, exist_ok=True)

    if cfg.resume_video:
        ckpt = torch.load(cfg.resume_ckpt_path_video, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        optim.load_state_dict(ckpt["optim"])
        start_step = int(ckpt.get("step", 0))
        logger.info(f"[resume video vae] loaded {cfg.resume_ckpt_path_video} @ step={start_step}")
    else:
        logger.info("train video vae from scratch")

    # AMP setup
    if cfg.amp_dtype == "bf16":
        amp_dtype = torch.bfloat16
    elif cfg.amp_dtype == "fp16":
        amp_dtype = torch.float16
    else:
        amp_dtype = None

    # scale used by forward/encode/decode in this file is optional; simplest: [0,1] (no latent normalization)
    scale = [0, 1]

    t_start = time.time()

    pbar = tqdm(range(start_step, cfg.max_steps), dynamic_ncols=True)

    for step in pbar:
        step_t0 = time.time()

        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)

        x = batch["video"].to(device, non_blocking=True)

        device_type = "cuda" if device.type == "cuda" else "cpu"
        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=(amp_dtype is not None)):
            x_rec, mu = model(x, scale=scale)

            # ---- align temporal length ----
            T = min(x_rec.shape[2], x.shape[2])  # time dimension
            x_rec_aligned = x_rec[:, :, :T]
            x_aligned = x[:, :, :T]

            l1 = F.l1_loss(x_rec_aligned, x_aligned)
            kl = torch.zeros((), device=device)
            loss = cfg.w_l1 * l1 + cfg.w_kl * kl

        optim.zero_grad(set_to_none=True)
        loss.backward()

        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        optim.step()

        # ---- tqdm display: 简洁即可 ----
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            l1=f"{l1.item():.4f}",
            lr=f"{optim.param_groups[0]['lr']:.2e}",
            T=f"{T}",
        )

        # ---- logger: 每 log_every step 记录一次详细信息 ----
        if step % cfg.log_every == 0:
            now = time.time()
            iter_time = now - step_t0
            elapsed = now - t_start
            it_per_sec = (step - start_step + 1) / max(elapsed, 1e-6)
            avg_it_time = elapsed / max(step - start_step + 1, 1)

            logger.info(
                f"step={step} "
                f"loss={loss.item():.6f} l1={l1.item():.6f} "
                f"lr={optim.param_groups[0]['lr']:.3e} "
                f"T_in={x.shape[2]} T_rec={x_rec.shape[2]} T_used={T} "
                f"iter_time={iter_time:.3f}s avg={avg_it_time:.3f}s it/s={it_per_sec:.2f}"
            )


        # ---- save ----
        if (step + 1) % cfg.save_every == 0 and cfg.out_dir_video:
            ckpt_path = os.path.join(cfg.out_dir_video, f"ckpt_step{step + 1}.pth")
            save_ckpt(ckpt_path, model, optim, step + 1)
            logger.info(f"[save] checkpoint -> {ckpt_path}")

            with torch.no_grad():
                x_rec_vis, _ = model(x, scale=scale)

                # 也对齐后再存，避免 recon 视频短帧看着怪
                T_vis = min(x_rec_vis.shape[2], x.shape[2])
                x_rec_vis = x_rec_vis[:, :, :T_vis]
                x_vis = x[:, :, :T_vis]
                save_path = os.path.join(cfg.out_dir_video + f"/step_{step + 1}")
                os.makedirs(save_path, exist_ok=True)
                save_video(x_rec_vis, os.path.join(save_path, f"x_rec_{step + 1}.mp4"))
                save_video(x_vis, os.path.join(save_path, f"x_{step + 1}.mp4"))
                logger.info(f"[save] videos -> x_rec_{step + 1}.mp4, x_{step + 1}.mp4 (T={T_vis})")

    # final save
    if cfg.out_dir_video:
        save_ckpt(os.path.join(cfg.out_dir_video, "ckpt_final.pth"), model, optim, cfg.max_steps)


if __name__ == "__main__":
    main()
