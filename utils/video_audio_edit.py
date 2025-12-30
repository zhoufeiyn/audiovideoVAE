"""
prepare_clips.py

purpose：
1) transfer data/audio/train1.m4a into .wav（sample rate 16k/44.1k、mono）
2) split data/video/train1.mp4 into 2s clips
3) video / audio align into 2s clips，save at data/clips/{video,audio}/

run：
  python prepare_clips.py

dependency：
  - install ffmpeg / ffprobe
"""

from __future__ import annotations

import subprocess
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
BASE = PROJECT_ROOT / "data"

# ====== params ======
AUDIO_IN = BASE / "audio" / "train1_10s.wav"
VIDEO_IN = BASE / "video" / "train1_10s.mp4"

# 音频标准化：建议 joint AV 先用 16k mono；如果你确定要 44.1k，也可以改成 44100
AUDIO_SR = 44100         # 16000 or 44100
AUDIO_CH = 1             # 1=mono, 2=stereo
AUDIO_SAMPLE_FMT = "s16" # pcm_s16le

# 视频标准化：建议固定 fps，且 keyframe 间隔 = fps * clip_seconds
FPS = 30
CLIP_SECONDS = 2

# 输出
AUDIO_WAV = BASE / "audio" / f"train1_{AUDIO_SR}hz_m{AUDIO_CH}.wav"
VIDEO_REKEY = BASE / "video" / f"train1_f{FPS}_g{FPS*CLIP_SECONDS}.mp4"
CLIPS_DIR = BASE / "clips"
VIDEO_CLIPS_DIR = CLIPS_DIR / "video"
AUDIO_CLIPS_DIR = CLIPS_DIR / "audio"
VIDEO_CLIP_PATTERN = VIDEO_CLIPS_DIR / "train1_%05d.mp4"
AUDIO_CLIP_PATTERN = AUDIO_CLIPS_DIR / "train1_%05d.wav"


def run(cmd: list[str]) -> None:
    """Run a command and raise if it fails."""
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def ffprobe_duration(path: Path) -> float:
    """Return duration (seconds) using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path)
    ]
    out = subprocess.check_output(cmd).decode("utf-8").strip()
    return float(out)



def main():
    # # 0) sanity check
    # if not AUDIO_IN.exists():
    #     raise FileNotFoundError(f"Audio not found: {AUDIO_IN}")
    # if not VIDEO_IN.exists():
    #     raise FileNotFoundError(f"Video not found: {VIDEO_IN}")
    #
    # VIDEO_CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    # AUDIO_CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    # # # 1) m4a -> wav (standardized)
    # run([
    #     "ffmpeg", "-y",
    #     "-i", str(AUDIO_IN),
    #     "-ac", str(AUDIO_CH),
    #     "-ar", str(AUDIO_SR),
    #     "-sample_fmt", AUDIO_SAMPLE_FMT,
    #     str(AUDIO_WAV)
    # ])

    # # 2) re-encode video to enable exact segmentation by time (force keyframe every CLIP_SECONDS)
    # gop = FPS * CLIP_SECONDS
    # run([
    #     "ffmpeg", "-y",
    #     "-i", str(VIDEO_IN),
    #     "-vf", f"fps={FPS}",
    #     "-g", str(gop),
    #     "-keyint_min", str(gop),
    #     "-sc_threshold", "0",
    #     "-pix_fmt", "yuv420p",
    #     str(VIDEO_REKEY)
    # ])

    # # 3) segment video into CLIP_SECONDS clips (now it will cut cleanly)
    # run([
    #     "ffmpeg", "-y",
    #     "-i", str(VIDEO_REKEY),
    #     "-map", "0:v:0",
    #     "-an",
    #     "-f", "segment",
    #     "-segment_time", str(CLIP_SECONDS),
    #     "-reset_timestamps", "1",
    #     str(VIDEO_CLIP_PATTERN)
    # ])
    #
    # # 4) segment audio into CLIP_SECONDS clips
    # run([
    #     "ffmpeg", "-y",
    #     "-i", str(AUDIO_WAV),
    #     "-f", "segment",
    #     "-segment_time", str(CLIP_SECONDS),
    #     "-reset_timestamps", "1",
    #     str(AUDIO_CLIP_PATTERN)
    # ])

    # # 5) add audio into video
    output = BASE/"video/train1_with_audio.mp4"
    # run([
    #     "ffmpeg", "-y",
    #     "-i", str(VIDEO_IN),
    #     "-i", str(AUDIO_IN),
    #     "-map", "0:v:0",
    #     "-map", "1:a:0",
    #     "-c:v", "copy",
    #     "-c:a", "aac",
    #     str(output)
    # ])

    # # 6) 10s clip
    clip= BASE/"video/train1_10s.mp4"
    # run( [
    #     "ffmpeg", "-y",
    #     "-i", str(output),
    #     "-t", "10",
    #     "-c", "copy",
    #     str(clip)
    # ])

    # # 7) seg video audio
    # audio = BASE/"audio/train1_10s.wav"
    # run(
    #     [
    #         "ffmpeg",
    #         "-i", str(clip),
    #         "-vn", str(audio)
    #     ]
    # )




    # **) quick report (durations + clip counts)
    v_dur = ffprobe_duration(VIDEO_IN)
    a_dur = ffprobe_duration(AUDIO_IN)

    print("\n===== REPORT =====")
    print(f"Raw video duration : {v_dur:.3f}s -> {VIDEO_IN}")
    print(f"Raw audio duration : {a_dur:.3f}s -> {AUDIO_IN}")






if __name__ == "__main__":
    main()
