# Phoneme Extractor and Mixer For TIMIT Dataset
This project extracts phonemes from the TIMIT test dataset.

## Setup
- Move TIMIT dataset into this folder
- Run timit_preprocess.sh (should convert NIST .WAV to RIFF .wav)
- Run phoneme_extract.py

An important thing to note is that certain extracted phonemes may contain little to no audio (the most obvious example being dx). This is caused by the speakers not emphasizing certain phonemes in their sentences and the relatively short duration of these phonemes in their sentences. Extracted phoneme audio containing these phonemes may not provide accurate results.
