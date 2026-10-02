"""
build_recipe.py
----------------
STEP 3 of the pipeline: turn the raw measurements from extract_features.py
into a "recipe" per emotion, expressed in RELATIVE units:

    - pitch_shift_semitones : how many semitones to shift the pitch median
    - pitch_range_factor    : how much to expand/compress pitch variability
    - duration_factor       : how much to speed up (<1.0) or slow down (>1.0)
    - energy_shift_db       : how many dB louder/softer

Relative units are the key design choice that lets a recipe built from
ONE speaker (e.g. a male reference set) transfer reasonably well to a
DIFFERENT speaker/gender's neutral recordings (e.g. your female dataset),
because "40% faster" or "+3 semitones" means roughly the same perceptual
change regardless of the speaker's absolute pitch range.

Usage:
    python build_recipe.py --features features.json --out recipe.json
"""

import argparse
import json
import statistics


def semitone_diff(f0_a, f0_b):
    """Semitone difference between two F0 values (a relative to b)."""
    if f0_a <= 0 or f0_b <= 0:
        return 0.0
    import math
    return 12 * math.log2(f0_a / f0_b)


def build_recipe(features: dict) -> dict:
    emotions = ["happy", "angry", "sad"]
    per_emotion = {e: {"pitch_shift_semitones": [], "pitch_range_factor": [],
                        "duration_factor": [], "energy_shift_db": []} for e in emotions}

    for sent_id, by_emotion in features.items():
        if "neutral" not in by_emotion:
            continue
        neutral = by_emotion["neutral"]
        for e in emotions:
            if e not in by_emotion:
                continue
            target = by_emotion[e]

            pitch_shift = semitone_diff(target["mean_f0_hz"], neutral["mean_f0_hz"])
            per_emotion[e]["pitch_shift_semitones"].append(pitch_shift)

            if neutral["f0_range_semitones"] > 0:
                range_factor = target["f0_range_semitones"] / neutral["f0_range_semitones"]
                per_emotion[e]["pitch_range_factor"].append(range_factor)

            duration_factor = target["duration_sec"] / neutral["duration_sec"]
            per_emotion[e]["duration_factor"].append(duration_factor)

            energy_shift = target["mean_intensity_db"] - neutral["mean_intensity_db"]
            per_emotion[e]["energy_shift_db"].append(energy_shift)

    recipe = {}
    for e in emotions:
        vals = per_emotion[e]
        if not vals["pitch_shift_semitones"]:
            print(f"Warning: no data found for emotion '{e}', skipping.")
            continue
        recipe[e] = {
            "pitch_shift_semitones": round(statistics.mean(vals["pitch_shift_semitones"]), 2),
            "pitch_range_factor": round(statistics.mean(vals["pitch_range_factor"]), 2),
            "duration_factor": round(statistics.mean(vals["duration_factor"]), 2),
            "energy_shift_db": round(statistics.mean(vals["energy_shift_db"]), 2),
            "n_sentences_used": len(vals["pitch_shift_semitones"]),
        }
    return recipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", default="recipe.json")
    args = ap.parse_args()

    with open(args.features, "r", encoding="utf-8") as f:
        features = json.load(f)

    recipe = build_recipe(features)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(recipe, f, indent=2, ensure_ascii=False)

    print("Recipe built:")
    print(json.dumps(recipe, indent=2, ensure_ascii=False))
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
