# Phoneme Extractor and Mixer For TIMIT Dataset
This project extracts phonemes from the TIMIT test dataset.

## Setup
- Move TIMIT dataset into this folder
- Run timit_preprocess.sh (should convert NIST .WAV to RIFF .wav)
- Run phoneme_extract.py
- (optional) Run phoneme_stratify.py, which creates a directory containing phoneme signals that the LAS model was successfully able to predict. 

An important thing to note is that many extracted phoneme signals may contain little to no audio (the most obvious example being dx), or will have inaccurate pronunciations. This is caused by the speakers not emphasizing certain phonemes in their sentences and the relatively short duration of these phonemes in their sentences. Extracted phoneme audio containing these phonemes may not provide accurate results. phoneme_stratify.py will largely resolve these issues.
