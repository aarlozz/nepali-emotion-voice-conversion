# Nepali Emotional Voice Conversion — Full Build Guide

A research prototype pipeline that converts **neutral Nepali speech → happy / angry / sad**,
preserving the original speaker's identity. Built on the methodology from:

- Bulut, Narayanan & Syrdal (2005) — phoneme-level prosody + spectral envelope shifts
- Zhou, Sisman & Li (2020) — speaker-independent neural emotion VC (VAW-GAN + CWT F0)
- Giovannini et al. (2024, 2026) — global vs. local voice-source (LF-model) + tempo/formant shifts

---

## 1. Architecture Overview

```
                     ┌────────────────────────────┐
STEP 1  Record  ───▶ │  Reference set (small)      │
                     │  same sentences x 4 emotions│
                     └─────────────┬───────────────┘
                                   ▼
STEP 2  Clean   ───▶  data_prep.py  (mono, resample, trim, normalize)
                                   ▼
STEP 3  Measure ───▶  extract_features.py (F0, energy, duration per file)
                                   ▼
STEP 4  Recipe  ───▶  build_recipe.py
                       → converts raw measurements into RELATIVE shifts:
                         semitones, duration ratio, dB, pitch-range factor
                         (relative units = transfers across speakers/genders)
                                   ▼
                     ┌────────────────────────────┐
STEP 5  Apply   ───▶ │ Your EXISTING neutral       │
                     │ dataset (the big pile)      │
                     └─────────────┬───────────────┘
                                   ▼
                       convert_emotion.py / batch_convert.py
                       (Praat "Change Gender" PSOLA transform:
                        pitch shift + pitch range + duration + formant)
                                   ▼
                     ┌────────────────────────────┐
                     │ emotional_dataset/          │
                     │   neutral/ happy/ angry/sad/│
                     └─────────────┬───────────────┘
                                   ▼
STEP 6  Evaluate───▶  evaluate.py
                       (A) objective: F0/duration/spectral distance
                       (B) subjective: listening-test CSV generator
                                   ▼
STEP 7 (optional) ─▶  Train a neural model (StarGAN-EVC / CycleGAN-VC2)
                       on the generated dataset for smoother, learned
                       conversion once you have enough data.
```

**Why this design:** you told me you want *same speaker, same identity, just add
emotion*. That means you don't need Zhou's full speaker-independent neural
framework — a rule-based DSP pipeline (Bulut + Giovannini) gets you working
results with **zero training data**, and never touches the parts of the
signal (broad spectral/formant structure) that carry speaker identity. It only
reshapes pitch contour, duration, energy, and — for happy — a small formant
raise.

---

## 2. What You Need to Record (the "reference set")

This is a **small, separate** recording set whose only job is to teach the
system how emotion changes a voice. It is not your main dataset.

### How many sentences?
**8–15 sentences**, each read in all 4 states (neutral, happy, angry, sad) =
32–60 recordings total. This matches the scale used in Bulut (2005) and
Giovannini's studies — enough to average out noise in the pitch/duration
measurements without requiring a huge recording session.

### How many speakers?
**2–3+ speakers if possible**, and match the gender/voice-type of your main
dataset (you said your dataset is female — try to get at least one female
reference speaker). If you truly can only record yourself, that's a usable
v1, but expect the transfer to female voices to be less accurate — the
recipe format (relative semitones/ratios, not absolute Hz) is specifically
designed to minimize this problem, but it won't eliminate it entirely.

### Which emotions?
Stick to **neutral, happy, angry, sad** — this matches all three source
papers exactly and keeps the recipe format simple. (Bulut also tested
"cold-anger" as a subtler 4th state; skip that for v1.)

### Example sentence set (Nepali)
Read each of these in all 4 emotional states. These are short, everyday,
phonetically varied sentences — **please double check naturalness/grammar
with a native Nepali speaker before your real recording session**, since I
generated these for structural variety, not fluency-tested phrasing.

| # | Nepali | Rough gloss |
|---|--------|-------------|
| 1 | आज मौसम राम्रो छ। | The weather is good today. |
| 2 | म बजार जाँदैछु। | I am going to the market. |
| 3 | उनी अस्पताल पुगे। | He/she reached the hospital. |
| 4 | यो किताब मेरो हो। | This book is mine. |
| 5 | भोलि विद्यालय बन्द हुन्छ। | School will be closed tomorrow. |
| 6 | तिमीलाई धेरै धन्यवाद। | Thank you very much. |
| 7 | हामी घर फर्कियौं। | We returned home. |
| 8 | उसले खाना खायो। | He/she ate food. |
| 9 | बत्ती अचानक निभ्यो। | The light suddenly went off. |
| 10 | बाहिर पानी परिरहेको छ। | It is raining outside. |
| 11 | मलाई यो ठाउँ मन पर्यो। | I liked this place. |
| 12 | उसले फोन उठाएन। | He/she didn't pick up the phone. |
| 13 | हामीले समयमै काम सिध्यायौं। | We finished the work on time. |
| 14 | त्यो निर्णय गलत थियो। | That decision was wrong. |
| 15 | सबैजना कार्यक्रममा आए। | Everyone came to the event. |

**Recording instruction for the speaker:** read each sentence naturally as
if genuinely feeling that emotion — not exaggerated/theatrical unless that's
intentionally the style you want in your dataset. Both papers used
professionally/naturally acted emotional speech, not caricature.

### Can you use your laptop mic?
**Yes, for prototyping** — your laptop mic is completely fine to build and
test the pipeline end-to-end (feature extraction only needs clean-ish pitch
tracking, which built-in mics handle adequately in a quiet room). Two
practical tips:
- Record in a quiet room, close the laptop lid mic distance to ~20–30cm from
  your mouth, avoid fans/AC noise.
- Keep the *same mic and same room* for the whole reference set — inconsistent
  recording conditions add noise to your feature measurements.

**For your final/publishable dataset**, upgrade if you can: a USB condenser
mic (even a budget one, ~$30–50) run through free software like **Audacity**
or **Praat itself** (which can record directly) will give cleaner F0 tracking
and better-sounding final audio than a laptop mic. Record at **44.1kHz,
16-bit, mono, WAV** — that's what all the scripts here expect.

### File naming convention
```
sentXX_neutral.wav
sentXX_happy.wav
sentXX_angry.wav
sentXX_sad.wav
```
(`extract_features.py` parses filenames using exactly this pattern.)

---

## 3. Step-by-Step: Running the Pipeline

### Step 0 — Install dependencies
```bash
pip install praat-parselmouth soundfile librosa scipy numpy --break-system-packages
```

### Step 1 — Record
Record your reference set (Section 2 above) and put your existing neutral
dataset in its own folder, e.g.:
```
reference_recordings/     <- small set, 4 emotions x N sentences
my_neutral_dataset/       <- your big existing pile, neutral only
```

### Step 2 — Clean the audio
Run this on **both** folders:
```bash
python scripts/data_prep.py --input_dir reference_recordings/ --output_dir reference_recordings_clean/
python scripts/data_prep.py --input_dir my_neutral_dataset/ --output_dir my_neutral_dataset_clean/
```

### Step 3 — Measure the reference set
```bash
python scripts/extract_features.py --input_dir reference_recordings_clean/ --out features.json
```

### Step 4 — Build the recipe
```bash
python scripts/build_recipe.py --features features.json --out recipe.json
```
Open `recipe.json` and sanity-check the numbers: happy/angry should show
*positive* pitch shift, sad should show *negative* pitch shift and a
*duration_factor > 1* (slower). If a value looks wildly off, check that
specific reference recording for mic issues or misreads.

### Step 5 — Convert your real dataset
Single file (for testing):
```bash
python scripts/convert_emotion.py --input my_neutral_dataset_clean/data01.wav \
    --recipe recipe.json --emotion happy --output happy_out.wav
```
Whole dataset at once:
```bash
python scripts/batch_convert.py --input_dir my_neutral_dataset_clean/ \
    --recipe recipe.json --output_dir emotional_dataset/
```
This produces:
```
emotional_dataset/
    neutral/   (originals, copied)
    happy/
    angry/
    sad/
    conversion_log.json
```

### Step 6 — Evaluate
**Objective** (fast, run this first on a handful of files):
```bash
python scripts/evaluate.py objective --neutral my_neutral_dataset_clean/data01.wav \
    --converted emotional_dataset/happy/data01.wav
```
Check: does the semitone shift/duration ratio roughly match your recipe's
intended values? A big mismatch usually means the input file's pitch
detection failed (noisy recording, whispered speech, etc).

**Subjective** (the real test — do humans agree it sounds happy/angry/sad?):
```bash
python scripts/evaluate.py make_test --dataset_dir emotional_dataset/ --out test_plan.csv
```
This generates a randomized trial list. Recruit **10+ Nepali-speaking
listeners**, play them each file (hide the `true_emotion` column), and have
them pick one of: happy / angry / sad / neutral / other — this is the exact
5-way forced-choice design Bulut et al. used. Score accuracy per emotion; a
confusion matrix (which emotions get mistaken for which) is usually more
informative than a single accuracy number.

---

## 4. Tuning Tips

- **If "angry" sounds too fast/robotic:** PSOLA-based methods (Bulut notes
  this explicitly) degrade when duration/pitch ratios are pushed past ~2x.
  Cap `duration_factor` and pitch shifts in `recipe.json` manually if your
  reference speaker's angry reading was very extreme.
- **If "happy" doesn't sound happy enough:** Giovannini found happy responds
  best to *local* (word/syllable-level) changes rather than uniform global
  shifts. The current pipeline is global-only; see the `LOCAL_REFINEMENT`
  note at the bottom of `convert_emotion.py` for how to add this later using
  a forced aligner.
- **Cross-gender transfer (your case — male reference, female dataset):**
  double check Step 6 objective metrics specifically on female files; if
  semitone shifts feel off, consider recording at least a few reference
  sentences from a female speaker and building a **separate recipe** for
  female inputs vs. male inputs.

---

## 5. Optional Step 7 — Neural Upgrade

Once `emotional_dataset/` has enough converted pairs (hundreds+), you can
train a small speaker-dependent neural model (CycleGAN-VC2 or StarGAN-EVC)
to learn a smoother, more natural version of this same transformation,
using your rule-based output as training targets. This is a genuinely
separate project phase — get the rule-based pipeline producing convincing
audio first.

---

## 6. Files in This Project

| File | Purpose |
|---|---|
| `scripts/data_prep.py` | Clean/standardize raw recordings |
| `scripts/extract_features.py` | Measure F0/energy/duration from reference set |
| `scripts/build_recipe.py` | Turn measurements into relative emotion "recipe" |
| `scripts/convert_emotion.py` | Apply recipe to one neutral file |
| `scripts/batch_convert.py` | Apply recipe to a whole dataset folder |
| `scripts/evaluate.py` | Objective metrics + subjective listening-test generator |

All scripts were tested end-to-end on synthetic audio and run correctly.
Replace the synthetic data with your real recordings and the same commands
apply.
