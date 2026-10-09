"""Cached dashboard data and presentation helpers."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from src.config import ProjectConfig
from src.data import load_data
from src.service import load_feature_importance

PALETTE = {
    "navy": "#12243A",
    "teal": "#00A896",
    "cyan": "#62D6C7",
    "coral": "#F07B62",
    "gold": "#E5B85C",
    "muted": "#7B8DA3",
}


@st.cache_data(ttl="15m", max_entries=2, show_spinner="Loading session data...")
def get_sessions() -> pd.DataFrame:
    return load_data()


@st.cache_data(ttl="2m", max_entries=2)
def get_metadata() -> dict[str, Any] | None:
    config = ProjectConfig.load()
    path = config.resolve(config.paths["metadata"])
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(ttl="2m", max_entries=2)
def get_model_comparison() -> pd.DataFrame:
    config = ProjectConfig.load()
    path = config.resolve(config.paths["metrics"])
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


@st.cache_data(ttl="2m", max_entries=2)
def get_importance() -> pd.DataFrame:
    return load_feature_importance()


@st.cache_data(ttl="2m", max_entries=2)
def get_shap_importance() -> pd.DataFrame:
    config = ProjectConfig.load()
    path = config.resolve(config.paths["shap_importance"])
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def style_figure(figure: Any, *, height: int = 360) -> Any:
    figure.update_layout(
        template="plotly_white",
        height=height,
        margin={"l": 12, "r": 12, "t": 34, "b": 12},
        font={"family": "Inter, sans-serif", "color": PALETTE["navy"]},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "y": 1.12, "x": 0},
    )
    figure.update_xaxes(showgrid=False, linecolor="#E7EDF3")
    figure.update_yaxes(gridcolor="#EDF2F6", zerolinecolor="#EDF2F6")
    return figure


def title_block(title: str, description: str, eyebrow: str) -> None:
    st.caption(eyebrow.upper())
    st.title(title)
    st.write(description)


def model_state() -> tuple[bool, dict[str, Any] | None]:
    metadata = get_metadata()
    return metadata is not None, metadata


def require_trained_model() -> dict[str, Any] | None:
    ready, metadata = model_state()
    if not ready:
        st.warning(
            "The analytics are ready; live model results will appear after training. "
            "From the project folder, run `python -m src.train`."
        )
        return None
    return metadata


def format_percent(value: float) -> str:
    return f"{value:.1%}"
