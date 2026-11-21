"""
convert_emotion.py
-------------------
STEP 4 of the pipeline: apply a recipe (built in build_recipe.py) to a
SINGLE new neutral audio file, producing an emotional version.

This uses Praat's "Change Gender" algorithm under the hood (via
parselmouth), which is a PSOLA-based transform that can independently
control:
    - formant shift ratio (we leave this ~1.0 unless targeting "happy",
      which benefits from a small formant raise per Giovannini et al.)
    - new pitch median (we compute this from the recipe's semitone shift)
    - pitch range factor (how much to expand/compress pitch variation)
    - duration factor (speed up / slow down)

This is the GLOBAL transform (uniform across the whole utterance) as
described in Bulut et al. (2005) and Giovannini et al. (2024/2026).
It is the fastest path to a working system. Local (syllable-level)
refinement is a later upgrade -- see LOCAL_REFINEMENT note at the
bottom of this file.

Usage:
    python convert_emotion.py --input neutral.wav --recipe recipe.json \
        --emotion happy --output happy_out.wav
"""

import argparse
import json
import math

import parselmouth
from parselmouth.praat import call


FORMANT_SHIFT_BY_EMOTION = {
    # Giovannini et al. found a small formant raise helps signal "happy".
    # Angry/sad are left at 1.0 (no formant shift) by default -- tune as needed.
    "happy": 1.04,
    "angry": 1.00,
    "sad": 1.00,
}


def get_mean_f0(snd: parselmouth.Sound) -> float:
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=60, pitch_ceiling=400)
    import numpy as np
    f0 = pitch.selected_array["frequency"]
    f0 = f0[f0 > 0]
    return float(f0.mean()) if len(f0) else 0.0


def convert(input_path: str, recipe: dict, emotion: str, output_path: str):
    if emotion not in recipe:
        raise ValueError(f"No recipe found for emotion '{emotion}'. "
                          f"Available: {list(recipe.keys())}")

    params = recipe[emotion]
    snd = parselmouth.Sound(input_path)

    current_median_f0 = get_mean_f0(snd)
    if current_median_f0 <= 0:
        raise RuntimeError("Could not detect pitch in input file -- check the recording.")

    # Convert the semitone shift from the recipe into an absolute target Hz
    # for THIS file's own voice (this is what makes it transfer across speakers).
    new_pitch_median = current_median_f0 * (2 ** (params["pitch_shift_semitones"] / 12))

    formant_shift_ratio = FORMANT_SHIFT_BY_EMOTION.get(emotion, 1.0)

    converted = call(
        snd,
        "Change gender",
        60,                                  # pitch floor (Hz)
        400,                                 # pitch ceiling (Hz)
        formant_shift_ratio,                 # formant shift ratio
        new_pitch_median,                    # new pitch median (Hz)
        params["pitch_range_factor"],        # pitch range factor
        params["duration_factor"],           # duration factor
    )

    converted.save(output_path, "WAV")
    print(f"[{emotion}] {input_path} -> {output_path}  "
          f"(pitch {current_median_f0:.0f}Hz -> {new_pitch_median:.0f}Hz, "
          f"duration x{params['duration_factor']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--emotion", required=True, choices=["happy", "angry", "sad"])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with open(args.recipe, "r", encoding="utf-8") as f:
        recipe = json.load(f)

    convert(args.input, recipe, args.emotion, args.output)


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# LOCAL_REFINEMENT (optional, advanced upgrade):
# The global transform above applies one uniform shift to the whole sentence.
# Giovannini et al. (2024/2026) found COMBINED global+local shifts (extra
# emphasis on stressed/accented syllables) usually score highest, especially
# for "angry". To add this later:
#   1. Get phoneme/word timing with a forced aligner (e.g. Montreal Forced
#      Aligner, which has an experimental Nepali model, or a simple
#      energy/pitch-peak heuristic to guess stressed syllables).
#   2. Use parselmouth's Manipulation object + PitchTier/DurationTier to
#      apply extra scaling only inside those time windows, on top of the
#      global shift above.
# This is a worthwhile v2 improvement, not required to get a working v1.
# ---------------------------------------------------------------------------
