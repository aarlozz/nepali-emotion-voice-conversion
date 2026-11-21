"""
batch_convert.py
-----------------
STEP 5 of the pipeline: take your ALREADY-COLLECTED neutral dataset
(the big pile of Nepali recordings you want to turn into an emotional
dataset) and run every file through convert_emotion.py for each
target emotion.

Usage:
    python batch_convert.py --input_dir my_neutral_dataset/ \
        --recipe recipe.json --output_dir emotional_dataset/

Output structure:
    emotional_dataset/
        happy/  <same filenames as input>
        angry/
        sad/
        neutral/  (original files copied over, for a complete 4-way dataset)
"""

import argparse
import json
import shutil
import traceback
from pathlib import Path

from convert_emotion import convert


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--emotions", nargs="+", default=["happy", "angry", "sad"])
    args = ap.parse_args()

    with open(args.recipe, "r", encoding="utf-8") as f:
        recipe = json.load(f)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    wav_files = sorted(input_dir.glob("*.wav"))

    if not wav_files:
        print(f"No .wav files found in {input_dir}")
        return

    # keep a neutral/ copy so the final dataset has all 4 classes together
    neutral_out = output_dir / "neutral"
    neutral_out.mkdir(parents=True, exist_ok=True)

    log = []
    for emotion in args.emotions:
        (output_dir / emotion).mkdir(parents=True, exist_ok=True)

    for wav in wav_files:
        shutil.copy(wav, neutral_out / wav.name)
        for emotion in args.emotions:
            out_path = output_dir / emotion / wav.name
            try:
                convert(str(wav), recipe, emotion, str(out_path))
                log.append({"file": wav.name, "emotion": emotion, "status": "ok"})
            except Exception as e:
                print(f"FAILED: {wav.name} -> {emotion}: {e}")
                traceback.print_exc()
                log.append({"file": wav.name, "emotion": emotion,
                            "status": "failed", "error": str(e)})

    n_ok = sum(1 for l in log if l["status"] == "ok")
    n_fail = sum(1 for l in log if l["status"] == "failed")
    print(f"\nDone. {n_ok} conversions succeeded, {n_fail} failed.")
    print(f"Dataset written to: {output_dir}")

    with open(output_dir / "conversion_log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
