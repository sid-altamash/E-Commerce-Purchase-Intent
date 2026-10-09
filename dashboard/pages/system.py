import platform

import streamlit as st

from dashboard.shared import get_model_comparison, get_sessions, model_state, title_block
from src.config import ProjectConfig

title_block(
    "Keep the pipeline in view",
    "Artifact readiness, data quality, model metadata, and reproducibility settings.",
    "Operations · System health",
)

sessions = get_sessions()
ready, metadata = model_state()
config = ProjectConfig.load()
checks = [
    ("Session CSV", "Available", True),
    ("Schema", "17 features + target", True),
    ("Model artifact", "Loaded" if ready else "Train required", ready),
    ("Comparison report", "Available" if not get_model_comparison().empty else "Pending",
     not get_model_comparison().empty),
]
with st.container(horizontal=True):
    for label, value, passed in checks:
        st.metric(label, value, border=True)

left, right = st.columns(2)
with left:
    with st.container(border=True):
        st.subheader("Data-quality snapshot")
        quality = {
            "Rows": f"{len(sessions):,}",
            "Columns": f"{len(sessions.columns)}",
            "Missing values": f"{int(sessions.isna().sum().sum()):,}",
            "Duplicate rows": f"{int(sessions.duplicated().sum()):,}",
            "Purchase rate": f"{sessions['Converted'].mean():.1%}",
        }
        st.table(quality)
with right:
    with st.container(border=True):
        st.subheader("Reproducible run settings")
        settings = {
            "Project version": config.project["version"],
            "Random seed": config.project["random_seed"],
            "Cross-validation folds": config.training["cv_folds"],
            "Python": platform.python_version(),
            "Threshold policy": "Validation expected value",
            "PageValues": "Included · late-session use",
            "Model": metadata["best_model"] if metadata else "Not trained",
        }
        st.table(settings)

if metadata:
    st.subheader("Held-out model metrics")
    st.json(metadata["test_metrics"], expanded=False)
else:
    st.info(
        "The service and dashboard do not retrain on load. Run `python -m src.train` "
        "once to create the model, metrics, and explanation artifacts."
    )
