import warnings
from pathlib import Path
import torch
import torchaudio
from decord import VideoReader
warnings.filterwarnings("ignore", category=UserWarning,module = r"torchaudio(\..*)?")

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
VIDEO = PROJECT_ROOT / "data"/"video"/"train1_10s.mp4"
AUDIO = PROJECT_ROOT/"data"/"audio"/"train1_10s.wav"

def main():
    vr = VideoReader(str(VIDEO))
    print("num Video frames:",len(vr))

    frame0 = (torch.from_numpy(vr[0].asnumpy()).permute(2,0,1).float()/255.0)
    print("raw frame shape:",frame0.shape,frame0.dtype)
    print("avg frame value:",frame0.mean().item())
    print("frame min:",frame0.min().item(),"frame max:",frame0.max().item())

    audio_samples, sample_rate = torchaudio.load(str(AUDIO))
    print("audio shape:",audio_samples.shape,"sample rate:",sample_rate)
    print("avg rms value:",audio_samples.pow(2).mean().sqrt().item())




if __name__ == "__main__":
    main()