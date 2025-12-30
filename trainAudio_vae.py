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
from configs.audio_vae_config import TrainAudioVAEConfig
from tqdm import tqdm
import warnings
import torchaudio
warnings.filterwarnings("ignore", category=UserWarning, module=r"torchaudio(\..*)?")

from models.VAE.dac import DAC
from models.VAE.discriminator import Discriminator

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


def make_model(cfg: TrainAudioVAEConfig, device: torch.device) -> nn.Module:
    generator = DAC(
        encoder_dim=cfg.audio_encoder_dim,
        encoder_rates=cfg.audio_encoder_rates,
        decoder_dim=cfg.audio_decoder_dim,
        decoder_rates=cfg.audio_decoder_rates,
        n_codebooks=cfg.n_codebooks,
        codebook_size=cfg.codebook_size,
        codebook_dim=cfg.codebook_dim,
        quantizer_dropout=cfg.quantizer_dropout,
        sample_rate=cfg.audio_sample_rate,
    )
    generator.to(device)
    # discriminator = Discriminator(
    #     rates=cfg.audio_rates,
    #     periods=cfg.audio_periods,
    #     fft_sizes=cfg.audio_fft_sizes,
    #     sample_rate=cfg.audio_sample_rate,
    #     bands=cfg.audio_bands,
    # )
    # discriminator.to(device)
    
    discriminator = None # Stage A no discriminator
    
    return generator, discriminator


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


def save_wav(wav: torch.Tensor, path: str, sr: int = 44100):
    # wav: (B,1,T) or (1,T)
    wav = wav.detach().cpu()
    if wav.dim() == 3:
        wav = wav[0]
    elif wav.dim() == 1:
        wav = wav.unsqueeze(0)
    wav = torch.clamp(wav, -1.0, 1.0)
    torchaudio.save(path, wav, sr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir_audio", type=str, default=None)
    args = parser.parse_args()

    config_kwargs = {}

    if args.out_dir_audio is not None:
        config_kwargs['out_dir_audio'] = args.out_dir_audio
    cfg = TrainAudioVAEConfig(**config_kwargs)

    logger = setup_logging(cfg.out_dir_audio, name="audio_vae")
    logger.info("===== Audio VAE Training =====")
    logger.info(f"device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
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

    generator, discriminator = make_model(cfg, device)
    generator.train()

    # discriminator.train()


    ######### stop here##########
    optim_g = torch.optim.AdamW(generator.parameters(), lr=cfg.lr, weight_decay=cfg.wd,betas=(0.8, 0.99))

    start_step = 0
    if cfg.out_dir_audio:
        os.makedirs(cfg.out_dir_audio, exist_ok=True)

    if cfg.resume_audio and os.path.exists(cfg.resume_ckpt_path_audio):
        ckpt = torch.load(cfg.resume_ckpt_path_audio, map_location="cpu")
        generator.load_state_dict(ckpt["model"], strict=True)
        optim_g.load_state_dict(ckpt["optim"])
        start_step = int(ckpt.get("step", 0))
        logger.info(f"[resume audio] loaded {cfg.resume_ckpt_path_audio} @ step={start_step}")
    else:
        logger.info("train audio vae from scratch")

    # AMP setup
    if cfg.amp_dtype == "bf16":
        amp_dtype = torch.bfloat16
    elif cfg.amp_dtype == "fp16":
        amp_dtype = torch.float16
    else:
        amp_dtype = None



    t_start = time.time()

    pbar = tqdm(range(start_step, cfg.max_steps), dynamic_ncols=True)

    from contextlib import nullcontext
    for step in pbar:
        step_t0 = time.time()

        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)

        # x = batch["audio"].repeat(2,1,1).to(device, non_blocking=True)
        x = batch["audio"].to(device, non_blocking=True)

        if x.dim()==2:
            x = x[:,None,:]
        elif x.dim()==1:
            x = x[None,None,:]
        if x.shape[-1] == 0:
            continue    # skip empty audio
        use_amp = (amp_dtype is not None) and (device.type == "cuda")
        ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()  
        with ctx:
            out = generator(x)
            x_rec = out["audio"]
            loss_wav =F.l1_loss(x_rec, x)
            loss_vq = 0.25*out["vq/commitment_loss"].mean() + 1.0*out["vq/codebook_loss"].mean()
            loss = loss_wav + loss_vq

            

        optim_g.zero_grad(set_to_none=True)
        loss.backward()

        
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1e3)

        optim_g.step()

        # ---- tqdm display: 简洁即可 ----
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            wav=f"{loss_wav.item():.4f}",
            vq=f"{loss_vq.item():.4f}",
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
                f"loss={loss.item():.6f} wav={loss_wav.item():.6f} vq={loss_vq.item():.6f} "
                f"lr={optim_g.param_groups[0]['lr']:.3e} "
                f"iter_time={iter_time:.3f}s avg={avg_it_time:.3f}s it/s={it_per_sec:.2f}"
            )


        # ---- save ----
        if (step + 1) % cfg.save_every == 0 and cfg.out_dir_audio:
            ckpt_path = os.path.join(cfg.out_dir_audio, f"ckpt_step{step + 1}.pth")
            save_ckpt(ckpt_path, generator, optim_g, step + 1)
            logger.info(f"[save] checkpoint -> {ckpt_path}")

            with torch.no_grad():
                out = generator(x)
                x_rec_vis = out["audio"]
                save_path = os.path.join(cfg.out_dir_audio + f"/step_{step + 1}")
                os.makedirs(save_path, exist_ok=True)

                
                save_wav(x, os.path.join(save_path, f"gt_{step+1}.wav"), cfg.audio_sample_rate)
                save_wav(x_rec_vis, os.path.join(save_path, f"rec_{step+1}.wav"), cfg.audio_sample_rate)
                logger.info(f"[save] audio -> x_rec_{step + 1}.wav")

    # final save
    if cfg.out_dir_audio:
        save_ckpt(os.path.join(cfg.out_dir_audio, "ckpt_final.pth"), generator, optim_g, cfg.max_steps)


if __name__ == "__main__":
    main()
