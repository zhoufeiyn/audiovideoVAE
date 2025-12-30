from dataclasses import dataclass, field
from typing import Tuple, List

@dataclass
class TrainAudioVAEConfig:
    # ==basic===
    out_dir_video: str = './run/vae_video'
    out_dir_audio: str = './run/vae_audio'
    resume_video: bool = False
    resume_audio: bool = False
    resume_ckpt_path_video: str = './run/vae_video/ckpt_final.pth'
    resume_ckpt_path_audio: str = './run/vae_audio/ckpt_final.pth'

    # === dataset===
    video_root: str = './data/video'
    audio_root: str = './data/audio'
    clip_len: int = 120
    size: int = 256
    # ---- temporal sampling ----
    fps_target: int = 0
    stride_choices: Tuple[int, ...] = (1, 2)
    random_stride: bool = False
    # ---- augmentation / preprocess ----
    random_crop: bool = False
    random_flip: bool = False
    color_jitter: float = 0.0
    keep_aspect_resize: bool = True
    # ---- audio options ----
    load_audio: bool = True
    audio_sample_rate: int = 44100
    audio_mono: bool = True
    audio_pad_to_clip: bool = True
    # ---- filtering ----
    min_video_frames: int = 0

    # ========= data =========
    batch_size: int = 1
    num_workers: int = 4

    # ========= wan2.2model =========
    dim: int = 160
    dec_dim: int = 256
    z_dim: int = 16
    dim_mult: Tuple[int, ...] = (1, 2, 4, 4)
    temperal_downsample: Tuple[bool, ...] = (True, True, False)

    # ========= audio generator model =========
    audio_encoder_dim: int = 64
    audio_encoder_rates: List[int] = field(default_factory=lambda: [2, 4, 8, 8])
    audio_decoder_dim: int = 1536
    audio_decoder_rates: List[int] = field(default_factory=lambda: [8, 8, 4, 2])
    n_codebooks: int = 9
    codebook_size: int = 1024
    codebook_dim: int = 8
    quantizer_dropout: float = 1.0

    # ========= audio discriminator model =========
    audio_rates: List[int] = field(default_factory=list)
    audio_periods: List[int] = field(default_factory=lambda: [2, 3, 5, 7, 11])
    audio_fft_sizes: List[int] = field(default_factory=lambda: [2048, 1024, 512])
    audio_bands: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
    )

    # ========= training =========
    lr: float = 1e-4
    wd: float = 1e-2
    max_steps: int = 10000
    grad_clip: float = 1.0
    log_every: int = 50
    save_every: int = 500

    # ========= system =========
    amp_dtype: str = "bf16"
    seed: int = 42

    # ========= loss =========
    w_l1: float = 1.0
    w_kl: float = 0.0
