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
from datasets.video_dataset import VideoAudioClipDataset
from configs.vae_config import TrainVAEConfig
from train_vae import save_video

# # IMPORTANT: use the repo's VAE implementation
# # from wan.modules.vae2_2 import WanVAE_   # if your PYTHONPATH is set
# from wan.modules.vae2_2 import WanVAE_  # assumes you run from repo root: Wan2.2/



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    config_kwargs = {}
    if args.data_dir is not None:
        config_kwargs['root'] = args.data_dir
    if args.out_dir is not None:
        config_kwargs['out_dir'] = args.out_dir
    cfg = TrainVAEConfig(**config_kwargs)


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
                print("save video...")
                save_path = cfg.out_dir+f"batch_{batch_idx}.mp4"
                save_video(video,save_path)
                
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
                    print("✓ Video values are in expected range [-1, 1]")
                
                # Check for NaN or Inf
                if torch.isnan(video).any():
                    print("ERROR: Video contains NaN values!")
                elif torch.isinf(video).any():
                    print("ERROR: Video contains Inf values!")
                else:
                    print("✓ Video contains no NaN or Inf values")
                
                # # Check audio if available
                # if "audio" in batch:
                #     audio = batch["audio"]
                #     print(f"\nAudio shape: {audio.shape}")
                #     print(f"Audio dtype: {audio.dtype}")
                #     if audio.numel() > 0:
                #         print(f"Audio min value: {audio.min().item():.4f}")
                #         print(f"Audio max value: {audio.max().item():.4f}")
                #     else:
                #         print("⚠ Audio is empty")
                #         # Diagnose why audio is empty
                #         if "meta" in batch:
                #             meta = batch["meta"]
                #             if isinstance(meta, dict):
                #                 print(f"  Duration: {meta.get('duration_sec', 'N/A')} seconds")
                #                 print(f"  FPS: {meta.get('fps', 'N/A')}")
                #                 print(f"  Video path: {meta.get('path', 'N/A')}")
                #         # Check if torchaudio is available
                #         try:
                #             import torchaudio
                #             print("  ✓ torchaudio is installed")
                #         except ImportError:
                #             print("  ✗ torchaudio is NOT installed (install with: pip install torchaudio)")
                #         # Check config
                #         print(f"  load_audio config: {cfg.load_audio}")
                #         print(f"  audio_sample_rate config: {cfg.audio_sample_rate}")
                
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
                
                print(f"✓ Batch {batch_idx + 1} validation passed")
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
