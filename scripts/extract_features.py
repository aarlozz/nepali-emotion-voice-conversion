"""
extract_features.py
--------------------
STEP 2 of the pipeline: measure voice parameters (F0, energy, duration)
from your REFERENCE recordings (the small set where you recorded the
same sentence in neutral + happy + angry + sad).

These measurements are what "build_recipe.py" will later turn into a
recipe of *relative* shifts (semitones, % duration, dB) per emotion,
which is what makes this transferable to a different speaker's voice
(e.g. a male reference recipe used on female recordings), as long as
you use % / semitone units instead of raw Hz.

Usage:
    python extract_features.py --input_dir reference_recordings/ --out features.json

Expected folder layout for --input_dir:
    reference_recordings/
        sent01_neutral.wav
        sent01_happy.wav
        sent01_angry.wav
        sent01_sad.wav
        sent02_neutral.wav
        ...
(same sentence id, different emotion suffix)
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import parselmouth
from parselmouth.praat import call


def measure_utterance(wav_path: str) -> dict:
    """Extract mean F0 (semitone-friendly), F0 range, RMS energy (dB), and duration."""
    snd = parselmouth.Sound(wav_path)

    # --- Pitch ---
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=60, pitch_ceiling=400)
    f0_values = pitch.selected_array["frequency"]
    f0_values = f0_values[f0_values > 0]  # drop unvoiced frames

    if len(f0_values) == 0:
        mean_f0 = float("nan")
        f0_range_semitones = float("nan")
    else:
        mean_f0 = float(np.mean(f0_values))
        # express spread in semitones so it's speaker-scale independent
        f0_range_semitones = float(
            12 * np.log2(np.percentile(f0_values, 95) / np.percentile(f0_values, 5))
        )

    # --- Energy (intensity) ---
    intensity = snd.to_intensity()
    mean_db = float(np.nanmean(intensity.values))

    # --- Duration ---
    duration = snd.duration

    return {
        "mean_f0_hz": mean_f0,
        "f0_range_semitones": f0_range_semitones,
        "mean_intensity_db": mean_db,
        "duration_sec": duration,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Folder of reference wav files")
    ap.add_argument("--out", default="features.json")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    pattern = re.compile(r"(?P<sent>.+)_(?P<emotion>neutral|happy|angry|sad)\.wav$", re.IGNORECASE)

    results = {}
    for wav_file in sorted(input_dir.glob("*.wav")):
        m = pattern.match(wav_file.name)
        if not m:
            print(f"Skipping (name doesn't match pattern): {wav_file.name}")
            continue
        sent_id = m.group("sent")
        emotion = m.group("emotion").lower()
        feats = measure_utterance(str(wav_file))
        results.setdefault(sent_id, {})[emotion] = feats
        print(f"{wav_file.name}: F0={feats['mean_f0_hz']:.1f}Hz  "
              f"dur={feats['duration_sec']:.2f}s  "
              f"energy={feats['mean_intensity_db']:.1f}dB")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved measurements for {len(results)} sentences -> {args.out}")


if __name__ == "__main__":
    main()
