"""Central configuration. Edit the paths below to point to your local
dataset directories before running the app, or leave the defaults and
enter paths via the Stage 0 UI (useful for sharing the repo without
committing local paths).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# LlamaPartialSpoof dataset
# Download: https://zenodo.org/records/14214149 (only label file and R01TTS.0.a subset)
# ---------------------------------------------------------------------------
LLAMAPARTIALSPOOF_AUDIO_DIR:  str = r"path/to/LlamaPartialSpoof/audio"
LLAMAPARTIALSPOOF_LABEL_FILE: str = r"path/to/LlamaPartialSpoof/label_file.txt"

# ---------------------------------------------------------------------------
# Soundscape dataset (ISD London subset)
# Download: https://zenodo.org/records/10672568
# Only the WAV_London_1 subfolder is required.
# ---------------------------------------------------------------------------
SOUNDSCAPE_DIR: str = r"path/to/ISD/WAV_London_1/Audio"

# ---------------------------------------------------------------------------
# ASVspoof 2021 Logical Access dataset
# Audio:  https://zenodo.org/records/4837263  (ASVspoof2021_LA_eval/)
# Keys:   https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz
# ---------------------------------------------------------------------------
ASVSPOOF2021_LA_AUDIO_DIR:  str = r"path/to/ASVspoof2021_LA_eval/flac"
ASVSPOOF2021_LA_LABEL_FILE: str = r"path/to/LA-keys-full/LA/CM/trial_metadata.txt"

# ---------------------------------------------------------------------------
# Path helpers — dataset-agnostic
# ---------------------------------------------------------------------------
_PLACEHOLDER_PREFIX = "path/to"


def _clean_path(p: str) -> str:
    """Strip whitespace and surrounding quotes from a user-entered path."""
    p = p.strip()
    if len(p) >= 2 and p[0] in ('"', "'") and p[-1] == p[0]:
        p = p[1:-1]
    return p


def paths_configured() -> bool:
    """Return True when no DatasetSpec in the registry still has placeholder paths.

    Checks session-state overrides first so the result is correct after st.rerun()
    """
    try:
        import streamlit as st
        from datasets import DATASET_REGISTRY
        overrides = st.session_state.get("path_overrides", {})
        for spec in DATASET_REGISTRY:
            audio_dir   = overrides.get(spec.key, {}).get("audio_dir",   spec.audio_dir)
            label_files = overrides.get(spec.key, {}).get("label_files", spec.label_files)
            if audio_dir.startswith(_PLACEHOLDER_PREFIX):
                return False
            if any(lf.startswith(_PLACEHOLDER_PREFIX) for lf in label_files):
                return False

        soundscape = overrides.get("__soundscape__", SOUNDSCAPE_DIR)
        if soundscape.startswith(_PLACEHOLDER_PREFIX):
            return False
        return True
    except Exception:
        return False


def apply_path_overrides(overrides: dict) -> None:
    """Persist path overrides in session state and patch live DatasetSpec entries.

    Parameters
    ----------
    overrides:
        {dataset_key: {"audio_dir": str, "label_files": [str, ...]}, ...}
    """
    import streamlit as st
    from datasets import DATASET_REGISTRY

    st.session_state["path_overrides"] = overrides

    for spec in DATASET_REGISTRY:
        if spec.key in overrides:
            spec.audio_dir   = overrides[spec.key]["audio_dir"]
            spec.label_files = overrides[spec.key]["label_files"]


def load_path_overrides_from_session() -> None:
    """Re-apply session-state overrides to live DatasetSpec entries on every rerun."""
    try:
        import streamlit as st
        from datasets import DATASET_REGISTRY
        overrides = st.session_state.get("path_overrides", {})
    except Exception:
        return

    for spec in DATASET_REGISTRY:
        if spec.key in overrides:
            spec.audio_dir   = overrides[spec.key]["audio_dir"]
            spec.label_files = overrides[spec.key]["label_files"]

    # Restore soundscape path if set separately via the Stage 2 UI.
    import config as _cfg
    if "__soundscape__" in overrides:
        _cfg.SOUNDSCAPE_DIR = overrides["__soundscape__"]