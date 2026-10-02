"""
evaluate.py
-----------
STEP 6 of the pipeline: evaluate how well the conversion worked, in two ways
(mirroring what Bulut et al. and Giovannini et al. did in their papers):

(A) OBJECTIVE metrics -- fast, automatic, good for iterating on your recipe:
    - F0 shift achieved (semitones) vs. what the recipe intended
    - Duration ratio achieved vs. intended
    - MCD-proxy: MFCC-based spectral distance between converted and original
      (sanity check that the spectral envelope moved but isn't destroyed)

(B) SUBJECTIVE listening test -- the real test of whether it "sounds happy/
    angry/sad". This script generates a randomized trial plan (CSV) you can
    use with real listeners, following the forced-choice design used in
    both Bulut (5-way: happy/angry/sad/neutral/other) and Giovannini
    (opposite-pair scale, e.g. sad(-3)...happy(+3)).

Usage:
    # objective check of one converted file against its neutral source
    python evaluate.py objective --neutral neutral.wav --converted happy_out.wav

    # generate a listening test plan from a folder of converted files
    python evaluate.py make_test --dataset_dir emotional_dataset/ --out test_plan.csv
"""

import argparse
import csv
import random
from pathlib import Path

import librosa
import numpy as np
import parselmouth


def get_mean_f0(path: str) -> float:
    snd = parselmouth.Sound(path)
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=60, pitch_ceiling=400)
    f0 = pitch.selected_array["frequency"]
    f0 = f0[f0 > 0]
    return float(f0.mean()) if len(f0) else 0.0


def get_duration(path: str) -> float:
    return parselmouth.Sound(path).duration


def mfcc_distance(path_a: str, path_b: str) -> float:
    """Rough spectral-distance proxy (not a full MCD, but same spirit)."""
    ya, sr = librosa.load(path_a, sr=None)
    yb, _ = librosa.load(path_b, sr=sr)
    mfcc_a = librosa.feature.mfcc(y=ya, sr=sr, n_mfcc=13)
    mfcc_b = librosa.feature.mfcc(y=yb, sr=sr, n_mfcc=13)
    n = min(mfcc_a.shape[1], mfcc_b.shape[1])
    diff = mfcc_a[:, :n] - mfcc_b[:, :n]
    return float(np.mean(np.sqrt(np.sum(diff ** 2, axis=0))))


def objective_report(neutral_path: str, converted_path: str):
    f0_n, f0_c = get_mean_f0(neutral_path), get_mean_f0(converted_path)
    dur_n, dur_c = get_duration(neutral_path), get_duration(converted_path)
    semitone_shift = 12 * np.log2(f0_c / f0_n) if f0_n > 0 else float("nan")
    duration_ratio = dur_c / dur_n if dur_n > 0 else float("nan")
    spectral_dist = mfcc_distance(neutral_path, converted_path)

    print(f"F0:        {f0_n:.1f} Hz -> {f0_c:.1f} Hz  ({semitone_shift:+.2f} semitones)")
    print(f"Duration:  {dur_n:.2f}s -> {dur_c:.2f}s  (ratio {duration_ratio:.2f})")
    print(f"Spectral distance (MFCC): {spectral_dist:.2f}  "
          f"(0 = identical; too close to 0 -> emotion barely changed the "
          f"spectrum; very large -> may sound distorted / unnatural)")


def make_listening_test(dataset_dir: str, out_csv: str, n_neutral_controls: int = 5):
    """
    Builds a CSV trial plan: one row per stimulus to play, with the correct
    answer hidden in a separate column for scoring later.
    Design mirrors Bulut's 5-way forced choice: happy / angry / sad / neutral / other.
    """
    dataset_dir = Path(dataset_dir)
    trials = []
    for emotion_dir in ["neutral", "happy", "angry", "sad"]:
        folder = dataset_dir / emotion_dir
        if not folder.exists():
            continue
        files = list(folder.glob("*.wav"))
        if emotion_dir == "neutral":
            files = random.sample(files, min(n_neutral_controls, len(files)))
        for f in files:
            trials.append({"file_path": str(f), "true_emotion": emotion_dir})

    random.shuffle(trials)
    for i, t in enumerate(trials, start=1):
        t["trial_number"] = i

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["trial_number", "file_path", "true_emotion"])
        writer.writeheader()
        writer.writerows(trials)

    print(f"Listening test plan with {len(trials)} trials -> {out_csv}")
    print("Give listeners the file_path column only (hide true_emotion) and have "
          "them choose one of: happy / angry / sad / neutral / other, per Bulut et al. "
          "Recommended: 10+ listeners, forced choice, headphones, random order "
          "(already randomized here).")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    obj = sub.add_parser("objective")
    obj.add_argument("--neutral", required=True)
    obj.add_argument("--converted", required=True)

    test = sub.add_parser("make_test")
    test.add_argument("--dataset_dir", required=True)
    test.add_argument("--out", default="test_plan.csv")

    args = ap.parse_args()
    if args.cmd == "objective":
        objective_report(args.neutral, args.converted)
    elif args.cmd == "make_test":
        make_listening_test(args.dataset_dir, args.out)


if __name__ == "__main__":
    main()
