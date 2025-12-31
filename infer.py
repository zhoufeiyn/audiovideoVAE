import os
import time

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
import imageio
from models.VAE.dac import DAC
from models.VAE.vae2_2 import WanVAE_
from torchvision.transforms import functional as TF

# Video decode (fast)
try:
    from decord import VideoReader, cpu
    _HAS_DECORD = True
except Exception:
    _HAS_DECORD = False

def make_model(cfg: TrainAudioVAEConfig, device: torch.device) -> nn.Module:
    dac = DAC(
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
    wan = WanVAE_(
        dim=cfg.dim,
        dec_dim=cfg.dec_dim,
        z_dim=cfg.z_dim,
        dim_mult=list(cfg.dim_mult),
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=list(cfg.temperal_downsample),
        dropout=0.0,
    )
    return dac, wan

def save_video(video: np.ndarray, save_path: str, fps: int = 30, value_range: Tuple[float, float] = (-1, 1)):
    """
    Save video from numpy array to mp4 file.
    
    Args:
        video: numpy array with shape (B, C, T, H, W) or (C, T, H, W) or (T, H, W, C)
               Values should be in value_range
        save_path: path to save the video file
        fps: frames per second for the output video
        value_range: (min, max) range of input values, default (-1, 1)
    """
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

def save_wav(wav: torch.Tensor, path: str, sr: int = 44100):
    # wav: (B,1,T) or (1,T)
    wav = wav.detach().cpu()
    if wav.dim() == 3:
        wav = wav[0]
    elif wav.dim() == 1:
        wav = wav.unsqueeze(0)
    wav = torch.clamp(wav, -1.0, 1.0)
    torchaudio.save(path, wav, sr)

def preprocess_video(frames_thwc_u8: torch.Tensor, size: int, keep_aspect_resize: bool = True) -> torch.Tensor:
    """
    预处理视频帧，与 dataset 中的 _preprocess_video 保持一致
    frames_thwc_u8: (T,H,W,C) uint8
    returns: (C,T,H,W) float32 in [-1,1]
    """
    x = frames_thwc_u8.float() / 255.0  # [0,1]
    x = x.permute(0, 3, 1, 2)           # (T,C,H,W)

    # Resize
    if keep_aspect_resize:
        # resize shorter side to size, keep aspect ratio
        Tt, C, H, W = x.shape
        if H < W:
            new_h = size
            new_w = int(round(W * (size / float(H))))
        else:
            new_w = size
            new_h = int(round(H * (size / float(W))))
        x = TF.resize(x, [new_h, new_w], antialias=True)
    else:
        x = TF.resize(x, [size, size], antialias=True)

    # Crop to square size
    Tt, C, H, W = x.shape
    if H < size or W < size:
        # pad if somehow smaller
        pad_h = max(0, size - H)
        pad_w = max(0, size - W)
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        Tt, C, H, W = x.shape

    # Center crop
    top = (H - size) // 2 if H > size else 0
    left = (W - size) // 2 if W > size else 0
    x = x[:, :, top:top + size, left:left + size]  # (T,C,S,S)

    # Normalize to [-1,1]
    x = x * 2.0 - 1.0

    # to (C,T,H,W)
    x = x.permute(1, 0, 2, 3).contiguous()
    return x

def read_audio_segment(audio_path: str, start_sec: float, duration_sec: float, 
                       target_sr: int = 44100, mono: bool = True, pad_to_clip: bool = True) -> torch.Tensor:
    """
    读取音频片段，与 dataset 中的 _read_audio_segment 保持一致
    audio_path: 音频文件路径
    start_sec: 开始时间（秒）
    duration_sec: 持续时间（秒）
    returns: (1, num_samples) float32 in [-1,1]
    """
    if duration_sec <= 0:
        return torch.zeros((1, 0), dtype=torch.float32)
    
    try:
        # 加载音频
        wav, sr = torchaudio.load(audio_path)  # wav: (channels, num_samples)
        wav = wav.to(torch.float32)
        
        # mono
        if mono and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        
        # resample if needed
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
            sr = target_sr
        
        # crop by time
        start_samp = max(0, int(round(start_sec * sr)))
        n_samp = max(0, int(round(duration_sec * sr)))
        end_samp = start_samp + n_samp
        
        if n_samp == 0:
            return torch.zeros((1, 0), dtype=torch.float32)
        
        if start_samp >= wav.shape[1]:
            # start beyond end
            seg = torch.zeros((wav.shape[0], 0), dtype=torch.float32)
        else:
            seg = wav[:, start_samp:min(end_samp, wav.shape[1])]
        
        # pad / truncate to exact n_samp
        if pad_to_clip:
            if seg.shape[1] < n_samp:
                seg = F.pad(seg, (0, n_samp - seg.shape[1]))
            else:
                seg = seg[:, :n_samp]
        
        seg = torch.clamp(seg, -1.0, 1.0)
        return seg
    
    except Exception as e:
        import warnings
        warnings.warn(f"Audio decode failed for {audio_path}: {type(e).__name__}: {str(e)}")
        return torch.zeros((1, 0), dtype=torch.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_video_root", type=str, default=None)
    parser.add_argument("--test_audio_root", type=str, default=None)
    parser.add_argument("--ckpt_path_video", type=str, default=None)
    parser.add_argument("--ckpt_path_audio", type=str, default=None)
    args = parser.parse_args()
    config_kwargs = {}
    if args.test_video_root is not None:
        config_kwargs['test_video_root'] = args.test_video_root
    if args.test_audio_root is not None:
        config_kwargs['test_audio_root'] = args.test_audio_root
    if args.ckpt_path_video is not None:
        config_kwargs['ckpt_path_video'] = args.ckpt_path_video
    if args.ckpt_path_audio is not None:
        config_kwargs['ckpt_path_audio'] = args.ckpt_path_audio
    cfg = TrainAudioVAEConfig(**config_kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dac, wan = make_model(cfg, device)
    dac.load_state_dict(torch.load(cfg.ckpt_path_audio, map_location="cpu")["model"], strict=True)
    wan.load_state_dict(torch.load(cfg.ckpt_path_video, map_location="cpu")["model"], strict=True)
    dac.eval()
    wan.eval()
    dac.to(device)
    wan.to(device)
    with torch.no_grad():
        for video_filename in tqdm(os.listdir(cfg.test_video_root)):
            if not video_filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v')):
                continue
            video_path = os.path.join(cfg.test_video_root, video_filename)
            audio_path = os.path.join(cfg.test_audio_root, video_filename.replace(".mp4", ".wav").replace(".mov", ".wav").replace(".avi", ".wav"))
            
            # 读取视频
            if _HAS_DECORD:
                # 使用 decord 读取视频（与 dataset 一致）
                vr = VideoReader(video_path, ctx=cpu(0))
                total_frames = len(vr)
                # 获取视频fps
                fps = float(getattr(vr, "get_avg_fps", lambda: 0.0)() or 0.0)
                
                # 读取所有帧或前 clip_len 帧
                start_indice = 120
                clip_len = 80
                indices = list(range(start_indice, start_indice + clip_len))
                frames = vr.get_batch(indices)  # (T,H,W,C) uint8 (NDArray)
                frames = torch.from_numpy(frames.asnumpy())  # torch uint8
            else:
                raise RuntimeError("Decord is required for video reading. Please install: pip install decord")
            
            # 预处理视频：转换为 (C,T,H,W) float32 in [-1,1]
            video = preprocess_video(frames, size=cfg.size, keep_aspect_resize=cfg.keep_aspect_resize)
            video = video.unsqueeze(0).to(device)  # 添加 batch 维度: (1,C,T,H,W)
            
            # 读取和预处理音频（与视频帧对齐）
            if os.path.exists(audio_path):
                # 根据视频帧索引计算对应的音频时间段
                if fps > 0:
                    start_sec = float(indices[0]) / fps
                    end_sec = float(indices[-1] + 1) / fps
                    duration_sec = max(0.0, end_sec - start_sec)
                else:
                    # fallback: 使用 clip_len 和 fps_target 估算
                    if cfg.fps_target and cfg.fps_target > 0:
                        duration_sec = float(clip_len) / float(cfg.fps_target)
                    else:
                        duration_sec = 0.0
                    start_sec = 0.0
                
                # 读取对应时间段的音频片段
                audio = read_audio_segment(
                    audio_path, 
                    start_sec=start_sec, 
                    duration_sec=duration_sec,
                    target_sr=cfg.audio_sample_rate, 
                    mono=cfg.audio_mono,
                    pad_to_clip=cfg.audio_pad_to_clip
                )
                audio = audio.unsqueeze(0).to(device)  # 添加 batch 维度: (1,1,T)
            else:
                print(f"Warning: Audio file not found: {audio_path}, skipping audio processing")
                audio = None
            
            # 推理
            scale = [0, 1]
            video_rec, _ = wan(video, scale=scale)
            
            if audio is not None:
                audio_rec = dac(audio)["audio"]  # (1,1,T)
            else:
                audio_rec = None
            
            # 保存结果
            output_video_path = os.path.join(cfg.test_video_root, video_filename.replace(".mp4", "_rec.mp4"))
            save_video(video.cpu().numpy(), os.path.join(cfg.test_video_root, video_filename).replace(".mp4", "_origin.mp4"))
            save_video(video_rec.cpu().numpy(), output_video_path)
            
            if audio_rec is not None:
                output_audio_path = os.path.join(cfg.test_audio_root, os.path.basename(audio_path).replace(".wav", "_rec.wav"))
                save_wav(audio, os.path.join(cfg.test_audio_root, os.path.basename(audio_path)).replace(".wav", "_origin.wav"), sr=cfg.audio_sample_rate)
                save_wav(audio_rec, output_audio_path, sr=cfg.audio_sample_rate)
            
if __name__ == "__main__":
    main()