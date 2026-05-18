"""Shared session-state helpers for Streamlit application."""
import streamlit as st

def init_state():
    """
    Initialize Streamlit session state with default values.
    """
    defaults = {
        "utterances": {},
        "audio_store": {},
        "processed": {},
        "pipeline_steps": [],
        "rating_pairs": [],
        "ratings": {},
        "pair_index": 0,
        "cm_scores": {},
        "id_map": {},
        "obf_records": [],
        "train_uids": [],
        "test_uids": [],
        "split_train_ratio": 0.8,
        "split_seed": 42,
        "selected_dataset": None,
        "db_cm_scores":       {},
        "insp_audio":         None,
        "insp_sr":            None,
        "insp_name":          "",
        "insp_pipeline":      [],
        "insp_det":           {},
        "insp_windowed":      {},
        "insp_occlusion":     {},
        "insp_acoustic":      {},
        "insp_auditory":      {},
        "insp_transcript":    None,
        "insp_lime":          {},
        "mini_combo_results": {},
        "an_preset_key":      None,
        "an_mbm_pipeline":    [],
        "an_done":            False,
        "an_mbm_cache_key":   None,
        "an_lr_results":      {},
        "analyst_notes":      "",
        "prior_prob":         0.01,
        "cost_fp":            1.0,
        "cost_fn":            10.0,
        "cache_status": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
