from pathlib import Path
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple, Any
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
import math, random, os

# Video decode (fast)
try:
    from decord import VideoReader, cpu
    _HAS_DECORD = True
except Exception:
    _HAS_DECORD = False

# Audio decode (segment read)
try:
    import torchaudio
    _HAS_TORCHAUDIO = True
except Exception:
    _HAS_TORCHAUDIO = False


class VideoAudioClipDataset(Dataset):
    """
    Returns a dict:
      {
        "video": FloatTensor (C, T, H, W) in [-1, 1],
        "audio": FloatTensor (Ca, N) in [-1, 1] (may be empty if audio missing),
        "meta":  dict with path/fps/stride/start_frame/...
      }
    """

    VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")
    AUDIO_EXTS = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac")

    def __init__(self, cfg):
        self.video_root = cfg.video_root
        self.audio_root = cfg.audio_root
        self.clip_len = cfg.clip_len
        self.size = cfg.size
        
        # ---- temporal sampling ----
        self.fps_target = cfg.fps_target
        self.stride_choices = cfg.stride_choices
        self.random_stride = cfg.random_stride
        
        # ---- augmentation / preprocess ----
        self.random_crop = cfg.random_crop
        self.random_flip = cfg.random_flip
        self.color_jitter = cfg.color_jitter
        self.keep_aspect_resize = cfg.keep_aspect_resize
        
        # ---- audio options ----
        self.load_audio = cfg.load_audio
        self.audio_sample_rate = cfg.audio_sample_rate
        self.audio_mono = cfg.audio_mono
        self.audio_pad_to_clip = cfg.audio_pad_to_clip
        
        # ---- filtering ----
        self.video_paths = self._scan_videos(self.video_root)
        if len(self.video_paths) == 0:
            raise RuntimeError(f"No video files found under: {self.video_root}")

        self.audio_map = self._scan_audios(self.audio_root)

        # Optional: best-effort filter by length (cheap metadata is not always available)
        if getattr(cfg, "min_video_frames", 0) > 0 and _HAS_DECORD:
            filtered = []
            for p in self.video_paths:
                try:
                    vr = VideoReader(p, ctx=cpu(0))
                    if len(vr) >= cfg.min_video_frames:
                        filtered.append(p)
                except Exception:
                    # keep it; don't fail on one bad file
                    filtered.append(p)
            self.video_paths = filtered

        # Worker-local caches (each worker has its own process, so dict is safe)
        self._vr_cache: Dict[str, Any] = {}  # path -> VideoReader
        # Note: audio StreamReader is better created per call (seeking differs); we won't cache it.

        if not _HAS_DECORD:
            raise RuntimeError(
                "Decord is not installed. Please `pip install decord` for fast training-time decoding."
            )

        # Optional: print pairing stats once
        if self.load_audio:
            paired = 0
            for vp in self.video_paths:
                if self._get_audio_path(vp) is not None:
                    paired += 1
            missing = len(self.video_paths) - paired
            print(f"[VideoAudioClipDataset] videos={len(self.video_paths)} paired_audio={paired} missing_audio={missing}")

    # ------------------------
    # Scan all subfolders
    # ------------------------
    def _scan_videos(self, root: str) -> List[str]:
        paths: List[str] = []
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith(self.VIDEO_EXTS):
                    paths.append(os.path.join(dirpath, fn))
        paths.sort()
        return paths

    def _scan_audios(self,root:str) -> Dict[str,str]:
        """
        Build mapping: <stem> -> <audio file path>
        Example: data/audio/train1_10s.wav  -> key "train1_10s"
        """
        audio_map: Dict[str,str] = {}
        rootp = Path(root)
        if not rootp.exists():
            return audio_map
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith(self.AUDIO_EXTS):
                    key = Path(fn).stem
                    audio_map[key] = os.path.join(dirpath, fn)
        return audio_map

    def _get_audio_path(self, video_path: str) -> Optional[str]:
        """
        Get audio file path by video file path
        Example: data/video/train1_10s.mp4  -> audio file path data/audio/train1_10s.wav
        """
        video_stem = Path(video_path).stem
        return self.audio_map.get(video_stem, None)

    # ------------------------
    # Video decode helpers
    # ------------------------
    def _get_vr(self, path: str):
        vr = self._vr_cache.get(path, None)
        if vr is None:
            try:
                vr = VideoReader(path, ctx=cpu(0))
            except Exception as e:
                raise RuntimeError(f"Failed to create VideoReader for path '{path}': {type(e).__name__}: {e}")
            self._vr_cache[path] = vr
        # Debug check: ensure vr is a VideoReader object
        if not hasattr(vr, '__len__'):
            raise TypeError(f"VideoReader returned unexpected type: {type(vr)}, path: {path}. "
                          f"Expected VideoReader object with __len__ method.")
        return vr

    def _choose_stride(self, src_fps: float) -> int:
        """
        Combine fps_target and random stride:
          - If fps_target>0, compute base stride ~ round(src_fps / fps_target)
          - Then optionally multiply by random choice in stride_choices
        """
        base = 1
        if self.fps_target and src_fps and src_fps > 0:
            base = max(1, int(round(src_fps / float(self.fps_target))))

        if self.random_stride and self.stride_choices:
            mult = random.choice(self.stride_choices)
        else:
            mult = 1

        return max(1, base * mult)

    # ------------------------
    # Preprocess / augment
    # ------------------------
    def _preprocess_video(self, frames_thwc_u8: torch.Tensor) -> torch.Tensor:
        """
        frames_thwc_u8: (T,H,W,C) uint8
        returns: (C,T,H,W) float32 in [-1,1]
        """
        x = frames_thwc_u8.float() / 255.0  # [0,1]
        x = x.permute(0, 3, 1, 2)           # (T,C,H,W)

        # Resize
        if self.keep_aspect_resize:
            # resize shorter side to self.size, keep aspect ratio
            Tt, C, H, W = x.shape
            if H < W:
                new_h = self.size
                new_w = int(round(W * (self.size / float(H))))
            else:
                new_w = self.size
                new_h = int(round(H * (self.size / float(W))))
            x = TF.resize(x, [new_h, new_w], antialias=True)
        else:
            x = TF.resize(x, [self.size, self.size], antialias=True)

        # Crop to square self.size
        Tt, C, H, W = x.shape
        if H < self.size or W < self.size:
            # pad if somehow smaller
            pad_h = max(0, self.size - H)
            pad_w = max(0, self.size - W)
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
            Tt, C, H, W = x.shape

        if self.random_crop:
            top = random.randint(0, H - self.size) if H > self.size else 0
            left = random.randint(0, W - self.size) if W > self.size else 0
        else:
            top = (H - self.size) // 2 if H > self.size else 0
            left = (W - self.size) // 2 if W > self.size else 0

        x = x[:, :, top:top + self.size, left:left + self.size]  # (T,C,S,S)

        # Random horizontal flip
        if self.random_flip and random.random() < 0.5:
            x = torch.flip(x, dims=[3])  # flip W

        # Simple color jitter (brightness/contrast), optional
        if self.color_jitter and self.color_jitter > 0:
            # brightness
            b = 1.0 + random.uniform(-self.color_jitter, self.color_jitter)
            # contrast
            c = 1.0 + random.uniform(-self.color_jitter, self.color_jitter)
            mean = x.mean(dim=(2, 3), keepdim=True)
            x = torch.clamp((x - mean) * c + mean, 0.0, 1.0)
            x = torch.clamp(x * b, 0.0, 1.0)

        # Normalize to [-1,1]
        x = x * 2.0 - 1.0

        # to (C,T,H,W)
        x = x.permute(1, 0, 2, 3).contiguous()
        return x

    # ------------------------
    # Audio segment read
    # ------------------------
    def _read_audio_segment(self, audio_path: Optional[str], start_sec: float, duration_sec: float) -> torch.Tensor:
        if not self.load_audio or (duration_sec <= 0) or (audio_path is None):
            return torch.zeros((1, 0), dtype=torch.float32)

        if not _HAS_TORCHAUDIO:
            return torch.zeros((1, 0), dtype=torch.float32)

        try:
            # wav, flac usually OK; mp4 audio needs ffmpeg backend in torchaudio build
            wav, sr = torchaudio.load(audio_path)  # wav: (channels, num_samples)
            wav = wav.to(torch.float32)

            # mono
            if self.audio_mono and wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)

            # resample if needed
            if sr != self.audio_sample_rate:
                wav = torchaudio.functional.resample(wav, sr, self.audio_sample_rate)
                sr = self.audio_sample_rate

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
            if self.audio_pad_to_clip:
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

    # ------------------------
    # Get item
    # ------------------------
    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= len(self.video_paths):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.video_paths)}")
        
        path = self.video_paths[idx]
        vr = self._get_vr(path)
        total = len(vr)
        fps = float(getattr(vr, "get_avg_fps", lambda: 0.0)() or 0.0)

        stride = self._choose_stride(fps)

        # We need clip_len frames AFTER striding => span = (clip_len-1)*stride + 1 frames
        span = (self.clip_len - 1) * stride + 1
        if total < span:
            # loop-pad by wrapping indices (avoid decoding full video)
            # We'll just sample with modulo.
            start = 0
            indices = [(start + i * stride) % max(1, total) for i in range(self.clip_len)]
        else:
            # start = random.randint(0, total - span)

            start = 120 # determined video clip
            indices = [start + i * stride for i in range(self.clip_len)]

        # Decode only selected frames
        # decord supports get_batch for list indices
        frames = vr.get_batch(indices)  # (T,H,W,C) uint8 (NDArray), T is clip_len
        frames = torch.from_numpy(frames.asnumpy())  # torch uint8

        video = self._preprocess_video(frames)  # (T,H,W,C) => (C,T,size,size) float [-1,1]

        # Audio segment aligned to the sampled video span
        if fps > 0:
            start_sec = float(indices[0]) / fps
            end_sec = float(indices[-1] + 1) / fps
            duration_sec = max(0.0, end_sec - start_sec)
        else:
            # fallback: assume clip_len/fps_target or just 0
            if self.fps_target and self.fps_target > 0:
                duration_sec = float(self.clip_len) / float(self.fps_target)
            else:
                duration_sec = 0.0
            start_sec = 0.0

        audio_path = self._get_audio_path(path)
        audio = self._read_audio_segment(audio_path, start_sec=start_sec, duration_sec=duration_sec)

        return {
            "video": video,
            "audio": audio,
        }

        # return {
        #     "video": video,
        #     "audio": audio,
        #     "meta": {
        #         "path": path,
        #         "total_frames": total,
        #         "fps": fps,
        #         "stride": stride,
        #         "indices": indices,
        #         "start_sec": start_sec,
        #         "duration_sec": duration_sec,
        #     }
        # }
        # return video, audio