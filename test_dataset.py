import os
import time
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.videoAudioClip_dataset import VideoAudioClipDataset
from configs.video_vae_config import TrainVideoVAEConfig
from trainVideo_vae import save_video

# # IMPORTANT: use the repo's VAE implementation
# # from wan.modules.vae2_2 import WanVAE_   # if your PYTHONPATH is set
# from wan.modules.vae2_2 import WanVAE_  # assumes you run from repo root: Wan2.2/

def combine_video_audio(video: torch.Tensor, audio: torch.Tensor, save_path: str):
    """
    将一个 batch 的 video+audio 合成带音频的 mp4 方便肉眼检查。

    支持输入形状：
    - video: (B,C,T,H,W) / (C,T,H,W) / (T,H,W,C) ，值域通常为 [-1,1]
    - audio: (B,Ca,N) / (Ca,N) / (N,) ，值域通常为 [-1,1]

    行为：
    - 如果 video 是 batch，会默认把 batch 里的每个样本分别保存为单独文件：
      `save_path` 若是 `xxx.mp4`，则输出 `xxx_b0.mp4`, `xxx_b1.mp4`, ...
    - 如果音频为空（numel==0），则只保存无声视频。
    """
    import tempfile
    import subprocess
    import wave
    from pathlib import Path

    import numpy as np

    # video writer: prefer imageio (if ffmpeg plugin available), otherwise fallback to opencv
    try:
        import imageio
        _HAS_IMAGEIO = True
    except ImportError:
        imageio = None
        _HAS_IMAGEIO = False

    try:
        import cv2
        _HAS_CV2 = True
    except ImportError:
        cv2 = None
        _HAS_CV2 = False

    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        # 兜底：如果系统 PATH 里有 ffmpeg，也能跑
        ffmpeg_exe = "ffmpeg"

    def _to_numpy(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    def _as_thwc_uint8(v: np.ndarray) -> np.ndarray:
        """
        v: (C,T,H,W) or (T,H,W,C) or (T,H,W) or (T,H,W,1)
        return: (T,H,W,3) uint8
        """
        if v.ndim == 4 and (v.shape[0] == 1 or v.shape[0] == 3):
            # (C,T,H,W) -> (T,H,W,C)
            v = np.transpose(v, (1, 2, 3, 0))
        elif v.ndim == 3:
            # (T,H,W) -> (T,H,W,1)
            v = v[..., None]
        elif v.ndim != 4:
            raise ValueError(f"Unsupported video shape: {v.shape}")

        # normalize [-1,1] -> [0,255]
        v = np.clip(v, -1.0, 1.0)
        v = (v + 1.0) * 0.5
        v = (v * 255.0).astype(np.uint8)

        # ensure 3 channels
        if v.shape[-1] == 1:
            v = np.repeat(v, 3, axis=-1)
        elif v.shape[-1] > 3:
            v = v[..., :3]
        return v

    def _write_wav_pcm16(path: str, wav: np.ndarray, sr: int) -> None:
        """
        wav: (Ca,N) float32 in [-1,1] OR (N,) float32 in [-1,1]
        write: PCM16 wav
        """
        if wav.ndim == 1:
            wav = wav[None, :]
        if wav.ndim != 2:
            raise ValueError(f"Unsupported audio shape: {wav.shape}")

        ca, n = wav.shape
        wav = np.clip(wav, -1.0, 1.0)
        pcm = (wav * 32767.0).astype(np.int16)  # (Ca,N)

        # interleave to (N,Ca)
        interleaved = pcm.T.reshape(-1)

        with wave.open(path, "wb") as wf:
            wf.setnchannels(int(ca))
            wf.setsampwidth(2)  # int16
            wf.setframerate(int(sr))
            wf.writeframes(interleaved.tobytes())

    def _write_video_mp4(path: str, frames_thwc_u8: np.ndarray, fps: float) -> None:
        # Try imageio first
        if _HAS_IMAGEIO:
            try:
                writer = imageio.get_writer(
                    path,
                    fps=float(fps),
                    codec="libx264",
                    quality=8,
                    format="FFMPEG",
                    macro_block_size=None,
                )
                try:
                    for fr in frames_thwc_u8:
                        writer.append_data(fr)
                finally:
                    writer.close()
                return
            except Exception as e:
                # fall back to opencv (common if imageio FFMPEG plugin missing)
                _ = e

        if not _HAS_CV2:
            raise RuntimeError("无法写 mp4：imageio 不可用/缺少 ffmpeg 插件，且 opencv-python 未安装。")

        h, w = int(frames_thwc_u8.shape[1]), int(frames_thwc_u8.shape[2])
        for fourcc_str in ("mp4v", "avc1"):
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            vw = cv2.VideoWriter(path, fourcc, float(fps), (w, h))
            if vw.isOpened():
                for fr in frames_thwc_u8:
                    vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
                vw.release()
                return
        raise RuntimeError("OpenCV VideoWriter 打开失败（已尝试 mp4v/avc1）。")

    video_np = _to_numpy(video)
    audio_np = _to_numpy(audio)

    if video_np is None:
        raise ValueError("video is None")

    # single sample video
    frames = _as_thwc_uint8(video_np)
    T = int(frames.shape[0])

    # infer sr and fps: 优先让视频时长与音频时长匹配（更利于检查对齐）
    sr = 44100  # 与 config 默认一致；你也可以按需改成 cfg.audio_sample_rate
    fps = 5.0   # 与 train_vae.save_video 默认一致（无音频时）

    has_audio = audio_np is not None and hasattr(audio_np, "size") and audio_np.size > 0
    if has_audio:
        if audio_np.ndim == 3:
            # 不应到这里（batch 早已处理），做个兜底
            audio_np = audio_np[0]
        if audio_np.ndim == 1:
            n_samp = int(audio_np.shape[0])
        else:
            n_samp = int(audio_np.shape[-1])
        dur = float(n_samp) / float(sr) if sr > 0 else 0.0
        if dur > 1e-6 and T > 0:
            fps = float(T) / dur

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        tmp_video = str(td / "tmp_video.mp4")
        tmp_wav = str(td / "tmp_audio.wav")

        _write_video_mp4(tmp_video, frames, fps=fps)

        if not has_audio:
            # 直接输出无声视频
            Path(save_path).write_bytes(Path(tmp_video).read_bytes())
            return

        # 写 wav
        _write_wav_pcm16(tmp_wav, audio_np, sr=sr)

        # mux: video copy + audio aac
        cmd = [
            str(ffmpeg_exe),
            "-y",
            "-i", tmp_video,
            "-i", tmp_wav,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            "-movflags", "+faststart",
            save_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    config_kwargs = {}
    if args.out_dir is not None:
        config_kwargs['out_dir'] = args.out_dir
    cfg = TrainVideoVAEConfig(**config_kwargs)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = VideoAudioClipDataset(cfg)
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    # Loop through entire dataloader, but only test first 3 batches
    test_iter = 3
    batch_count = 0
    
    print(f"Starting to iterate through entire dataloader...")
    print(f"Will validate first {test_iter} batches\n")
    
    try:
        # Iterate through entire dataloader
        for batch_idx, batch in enumerate(dl):
            # Only validate first test_iter batches
            if batch_count < test_iter:
                print(f"\n{'='*30}")
                print(f"Batch {batch_idx + 1} (Test {batch_count + 1}/{test_iter})")
                print(f"{'='*30}")
                
                # Extract video tensor
                video = batch["video"]
                # print("save video...")
                # save_path = os.path.join(cfg.out_dir, f"batch_{batch_idx}.mp4")
                # save_video(video,save_path)
                
                # Validate video tensor
                print(f"Video shape: {video.shape}")
                print(f"Video dtype: {video.dtype}")
                print(f"Video min value: {video.min().item():.4f}")
                print(f"Video max value: {video.max().item():.4f}")
                print(f"Video mean value: {video.mean().item():.4f}")
                print(f"Video std value: {video.std().item():.4f}")
                
                # Check expected shape: (B, C, T, H, W) after batching
                expected_dims = 5
                if video.dim() != expected_dims:
                    print(f"WARNING: Expected {expected_dims} dimensions, got {video.dim()}")
                else:
                    B, C, T, H, W = video.shape
                    print(f"Batch size: {B}, Channels: {C}, Frames: {T}, Height: {H}, Width: {W}")
                    
                    # Validate shape matches config
                    if T != cfg.clip_len:
                        print(f"WARNING: Expected clip_len={cfg.clip_len}, got T={T}")
                    if H != cfg.size or W != cfg.size:
                        print(f"WARNING: Expected size={cfg.size}x{cfg.size}, got {H}x{W}")
                
                # Check value range: should be in [-1, 1]
                if video.min() < -1.1 or video.max() > 1.1:
                    print(f"WARNING: Video values out of expected range [-1, 1]")
                else:
                    print("[OK] Video values are in expected range [-1, 1]")
                
                # Check for NaN or Inf
                if torch.isnan(video).any():
                    print("ERROR: Video contains NaN values!")
                elif torch.isinf(video).any():
                    print("ERROR: Video contains Inf values!")
                else:
                    print("[OK] Video contains no NaN or Inf values")
                
                # Check audio if available
                if "audio" in batch:
                    audio = batch["audio"]
                    print(f"\nAudio shape: {audio.shape}")
                    print(f"Audio dtype: {audio.dtype}")
                    if audio.numel() > 0:
                        print(f"Audio min value: {audio.min().item():.4f}")
                        print(f"Audio max value: {audio.max().item():.4f}")
                    else:
                        print("[WARN] Audio is empty")
                        # Diagnose why audio is empty
                        if "meta" in batch:
                            meta = batch["meta"]
                            if isinstance(meta, dict):
                                print(f"  Duration: {meta.get('duration_sec', 'N/A')} seconds")
                                print(f"  FPS: {meta.get('fps', 'N/A')}")
                                print(f"  Video path: {meta.get('path', 'N/A')}")
                        # Check if torchaudio is available
                        try:
                            import torchaudio
                            print("  [OK] torchaudio is installed")
                        except ImportError:
                            print("  [ERR] torchaudio is NOT installed (install with: pip install torchaudio)")
                        # Check config
                        print(f"  load_audio config: {cfg.load_audio}")
                        print(f"  audio_sample_rate config: {cfg.audio_sample_rate}")
                
                # Check metadata if available
                if "meta" in batch:
                    meta = batch["meta"]
                    print(f"\nMetadata:")
                    if isinstance(meta, dict):
                        for key, value in meta.items():
                            if isinstance(value, torch.Tensor):
                                print(f"  {key}: tensor shape {value}")
                            else:
                                print(f"  {key}: {value}")

                # combine audio and video into a single video, and save it
                print("combine audio and video into a single video, and save it...")
                save_path = os.path.join(cfg.out_dir, f"batch_{batch_idx}_combined.mp4")
                combine_video_audio(video[0], audio[0], save_path)
                
                print(f"[OK] Batch {batch_idx + 1} validation passed")
                batch_count += 1
            else:
                # Continue iterating but skip validation for remaining batches
                pass
        
        print(f"\n{'='*30}")
        print(f"Completed iterating through entire dataloader")
        print(f"Total batches processed: {batch_idx + 1}")
        print(f"Batches validated: {batch_count}")
        print(f"{'='*30}")
        
    except Exception as e:
        print(f"\nERROR during dataloader iteration: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()






if __name__ == "__main__":
    main()
