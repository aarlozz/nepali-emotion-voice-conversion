"""
data_prep.py
------------
STEP 1.5 of the pipeline (run right after recording, before anything else):
cleans up raw recordings so every file is in a consistent format:
    - mono
    - 44100 Hz sample rate (or whatever you set with --target_sr)
    - normalized to a consistent peak level (avoids clipping / too-quiet files)
    - trims leading/trailing silence

Run this on BOTH your small reference set and your big existing dataset,
so everything downstream is consistent.

Usage:
    python data_prep.py --input_dir raw_recordings/ --output_dir clean_recordings/
"""

import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def clean_file(path: str, target_sr: int, top_db: int = 30):
    audio, sr = librosa.load(path, sr=target_sr, mono=True)

    # trim leading/trailing silence
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=top_db)

    # peak-normalize to -3 dBFS to leave headroom
    peak = np.max(np.abs(audio_trimmed)) if len(audio_trimmed) else 0
    if peak > 0:
        target_peak = 10 ** (-3 / 20)
        audio_trimmed = audio_trimmed * (target_peak / peak)

    return audio_trimmed, target_sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--target_sr", type=int, default=44100)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(list(input_dir.glob("*.wav")) + list(input_dir.glob("*.mp3"))
                    + list(input_dir.glob("*.m4a")))
    if not files:
        print(f"No audio files found in {input_dir}")
        return

    for f in files:
        try:
            audio, sr = clean_file(str(f), args.target_sr)
            out_path = output_dir / (f.stem + ".wav")
            sf.write(str(out_path), audio, sr, subtype="PCM_16")
            print(f"OK  {f.name} -> {out_path.name}  ({len(audio)/sr:.2f}s)")
        except Exception as e:
            print(f"FAILED {f.name}: {e}")

    print(f"\nCleaned {len(files)} files -> {output_dir}")


if __name__ == "__main__":
    main()
