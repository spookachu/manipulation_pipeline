# Voice Deepfake Forensic Pipeline

A Streamlit-based application for forensic assessment of synthetic speech. Includes functionality for:
- ingesting existing speech datasets;
- creating a background model of scores using state-of-the-art detection models;
- loading up a target sample;
- auditorily analizing the target sample;
- applying various audio manipulation blocks;
- creating a modified-background model of the speech dataset;¨
- comparing the M-BM with the target sample to infer applied proccessing steps;

TO DO:
- [ ] create validation tests for BM computation and manipulation steps;
- [ ] integrate acoustic analysis methods;
- [ ] create ability to generate a forensic report;
- [ ] integrate additional datasets and detection models;
- [ ] incorporate different prior brackets for detection threshold computation;
- [ ] improve aesthetics.
---

## What it does

The app is organised into three stages:

| Stage | Name | Purpose |
|---|---|---|
| 0 | Database | Load a dataset, compute and cache CM scores for the background model |
| 1 | Inspection | Upload a target utterance, run acoustic analysis and deepfake detection |
| 2 | Analysis | Investigate likelihood of synthetic content; assess whether manipulations shift the evidence |

The framework follows the likelihood ratio standard:

```
LR = P(E | Hp) / P(E | Hd)
```

where `Hp` is the prosecution hypothesis (the audio is synthetic) and `Hd` is the defence hypothesis (the audio is genuine).

### Platform Logic Summary

| Component | Implementation | Assumption / Limitation |
|---|---|---|
| Likelihood model | Gaussian KDE fitted to background-model CM scores | Common non-parametric estimator used in forensic LR literature |
| Hp distribution | Synthetic + partial synthetic CM scores pooled | Conservative choice; may inflate distribution variance |
| Hd distribution (BM) | Unmanipulated bonafide CM scores | Models genuine unmanipulated speech as the defence baseline |
| Hd distribution (M-BM) | Processed bonafide CM scores | Conditions Hd on the same processing as the target |
| CM score | Single utterance-level forward pass | Valid for fully bonafide/synthetic; for partial synthetic works as an approximation |
| Posterior | P(Hp \| E) = LR · prior / (LR · prior + (1 − prior)) | Bayes' theorem; requires analyst-specified prior |
| Decision threshold | τ = C_FP(1 − prior) / (C_FN · prior) | Bayes-optimal under analyst-specified asymmetric costs  |
| Calibration | Cllr (log-likelihood-ratio cost) computed from KDE-derived LRs on BM scores after each scoring run | Measures combined discrimination and calibration quality of the LR system on the background corpus; Cllr = 0 is perfect, Cllr = 1 is no better than chance, Cllr > 1 indicates the system is actively miscalibrated. |
| BM train/test split | Stratified by label, TTS model, duration, synthetic segment count | Speaker identity not stratified; different speakers may have systematically different CM scores |

---
## Installation

**Requirements:** Python 3.10+

### Option A — conda

```bash
git clone https://github.com/your-org/voice-deepfake-forensics.git
conda create -n deepfake-forensics python=3.10
conda activate deepfake-forensics
pip install -r requirements.txt
```

### Option B — pip (venv)

```bash
git clone https://github.com/your-org/voice-deepfake-forensics.git
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

To run the app, execute:

```bash
streamlit run app.py
```


### Configure paths

Open `config.py` and set your local paths:

```python
LLAMAPARTIALSPOOF_AUDIO_DIR  = r"path/to/LlamaPartialSpoof/audio"
LLAMAPARTIALSPOOF_LABEL_FILE = r"path/to/LlamaPartialSpoof/label_file.txt"
SOUNDSCAPE_DIR               = r"path/to/ISD/WAV_London_1/Audio"
```

## Assets
### Datasets
Available datasets:

**LlamaPartialSpoof**: contains bonafide, synthetic, and partially synthetic utterances. Download the dataset from the authors and note the path to:
- the audio directory (contains `.wav` files)
- the label file (`label_R01TTS.0.a.txt` or equivalent)

<em>Luong, H. T., Li, H., Zhang, L., Lee, K. A., & Chng, E. S. (2025, April). Llamapartialspoof: An llm-driven fake speech dataset simulating disinformation generation. In ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 1-5). IEEE.</em>


#### Adding a new dataset

1. Open `datasets.py` and append a new `DatasetSpec` entry to `DATASET_REGISTRY`:

```python
DatasetSpec(
    key          = "mydataset",
    display_name = "My Dataset",
    audio_dir    = MY_AUDIO_DIR,      # import from config.py
    label_files  = [MY_LABEL_FILE],   # import from config.py
    description  = "...",
    license      = "...",
    citation     = "...",
    train_ratio  = 0.8,
    split_seed   = 42,
),
```

2. Add the corresponding env vars to `config.py`.

### Models
Models are extracted from the [Speech Deepfake Arena](https://huggingface.co/spaces/Speech-Arena-2025/Speech-DF-Arena) hosted on HuggingFace. The usefulness criteria are based on 1. high performance on benchmark datasets 2. open-source availability 3. ease of integration.

**Speech-Arena-2025/DF_Arena_1B_V_1**: uses a non-commercial license which can be found [here](https://huggingface.co/Speech-Arena-2025/DF_Arena_1B_V_1/blob/main/LICENSE.txt).


### Additional

**Soundscape (ISD London subset)**: environmental noise, used as a manipulation block. Download from [Zenodo](https://zenodo.org/records/10672568). Only the `WAV_London_1` subfolder is used.


## Score cache

CM scores are cached in `cache/` as JSON files named `{dataset_key}_{model_key}.json`. The cache is read on startup and written incrementally; a run interrupted partway through will resume from where it left off. The `cache/` directory is excluded from version control (see `.gitignore`).

The background model (BM) train/test split is cached separately as `{dataset_key}_bm_split.json`

---

## Project structure

```
.
├── app.py                  # Streamlit entry point
├── background_model.py     # LR framework (KDE fitting, posterior, decision threshold)
├── config.py               # Environment variables
├── datasets.py             # Dataset registry (add new datasets here)
├── detection.py            # CM model inference and score caching
├── manipulations.py        # Audio processing blocks
├── session.py              # Streamlit session state initialisation
├── utils.py                # Shared data types, label parsing, audio I/O
├── requirements.txt
├── cache/                  # BM and M-BM score cache 
```

---

## Citation

If you use this tool in your research, please cite the relevant dataset and model papers listed in the app's Dataset Info panel.
