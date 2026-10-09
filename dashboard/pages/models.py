import plotly.express as px
import streamlit as st

from dashboard.shared import (
    PALETTE,
    get_model_comparison,
    get_metadata,
    style_figure,
    title_block,
)

title_block(
    "Benchmark the decision engine",
    "Compare stratified cross-validation performance, then inspect the held-out "
    "test results at the value-optimized operating threshold.",
    "Model intelligence · Evaluation",
)

metadata = get_metadata()
comparison = get_model_comparison()
if comparison.empty or metadata is None:
    st.warning("No saved model report yet. Run `python -m src.train` from the project folder.")
else:
    metrics = metadata["test_metrics"]
    with st.container(horizontal=True):
        st.metric("Selected model", metadata["best_model"], border=True)
        st.metric("Test PR-AUC", f"{metrics['pr_auc']:.3f}", border=True)
        st.metric("Test recall", f"{metrics['recall']:.1%}", border=True)
        st.metric("Chosen threshold", f"{metadata['threshold']:.2f}", border=True)

    with st.container(border=True):
        st.subheader("Cross-validated model comparison")
        chart = px.bar(
            comparison.sort_values("cv_pr_auc", ascending=True),
            x="cv_pr_auc",
            y="model",
            orientation="h",
            error_x="cv_pr_auc_std",
            color="cv_pr_auc",
            color_continuous_scale=[PALETTE["cyan"], PALETTE["teal"], PALETTE["navy"]],
            labels={"cv_pr_auc": "Mean PR-AUC", "model": "Classifier"},
        )
        style_figure(chart, height=390)
        st.plotly_chart(
            chart,
            width="stretch",
            alt="Mean stratified cross-validation precision-recall area for six classifiers",
        )
        st.caption("Primary ranking metric: PR-AUC, appropriate for the minority purchase class.")
        st.dataframe(
            comparison,
            hide_index=True,
            column_config={
                "cv_pr_auc": st.column_config.NumberColumn("CV PR-AUC", format="%.3f"),
                "cv_pr_auc_std": st.column_config.NumberColumn("Fold spread", format="%.3f"),
                "cv_precision": st.column_config.NumberColumn("CV precision", format="%.1%"),
                "cv_recall": st.column_config.NumberColumn("CV recall", format="%.1%"),
                "cv_accuracy": st.column_config.NumberColumn("CV accuracy", format="%.1%"),
                "cv_roc_auc": st.column_config.NumberColumn("CV ROC-AUC", format="%.3f"),
                "cv_f1": st.column_config.NumberColumn("CV F1", format="%.3f"),
            },
            alt="Cross-validation comparison for all trained candidate models",
        )

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.subheader("Held-out confusion matrix")
            matrix = metrics["confusion_matrix"]
            chart = px.imshow(
                matrix,
                text_auto=True,
                x=["Predicted no", "Predicted purchase"],
                y=["Actual no", "Actual purchase"],
                color_continuous_scale=["#F5F8FA", PALETTE["teal"]],
                labels={"x": "", "y": "", "color": "Sessions"},
            )
            style_figure(chart, height=320)
            st.plotly_chart(
                chart,
                width="stretch",
                alt="Held-out confusion matrix at the value-selected threshold",
            )
    with right:
        with st.container(border=True):
            st.subheader("Operating-point economics")
            st.metric(
                "Expected value · validation",
                f"${metadata['validation_expected_value']:,.0f}",
            )
            st.metric(
                "Expected value · held-out test",
                f"${metadata['test_expected_value']:,.0f}",
            )
            st.caption(
                f"Scenario assumes ${metadata['conversion_value']:,.0f} contribution per "
                f"captured purchase and ${metadata['intervention_cost']:,.0f} per "
                "intervention. Adjust these assumptions in config.yaml before training."
            )
