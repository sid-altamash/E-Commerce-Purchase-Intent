import plotly.express as px
import streamlit as st

from dashboard.shared import (
    PALETTE,
    get_importance,
    get_shap_importance,
    get_sessions,
    require_trained_model,
    style_figure,
    title_block,
)
from src.config import FeatureSpec, ProjectConfig
from src.data import engineer_features
from src.explain import local_lime_values, local_shap_values
from src.service import load_bundle, predict_session

title_block(
    "Open the model black box",
    "Inspect global feature importance and explain an individual session with SHAP "
    "and an independent LIME view.",
    "Model transparency · Explainability",
)

metadata = require_trained_model()
importance = get_importance()
shap_importance = get_shap_importance()
if not shap_importance.empty:
    with st.container(border=True):
        st.subheader("Global SHAP importance")
        chart = px.bar(
            shap_importance.head(12).sort_values("mean_absolute_shap"),
            x="mean_absolute_shap",
            y="feature",
            orientation="h",
            color="mean_absolute_shap",
            color_continuous_scale=[PALETTE["cyan"], PALETTE["teal"], PALETTE["navy"]],
            labels={"mean_absolute_shap": "Mean absolute SHAP value", "feature": ""},
        )
        style_figure(chart, height=430)
        st.plotly_chart(
            chart,
            width="stretch",
            alt="Global SHAP feature importance aggregated to original session behaviors",
        )

if not importance.empty:
    with st.container(border=True):
        st.subheader("Global permutation importance")
        chart = px.bar(
            importance.head(12).sort_values("permutation_importance"),
            x="permutation_importance",
            y="feature",
            orientation="h",
            error_x="importance_std",
            color="permutation_importance",
            color_continuous_scale=[PALETTE["cyan"], PALETTE["teal"], PALETTE["navy"]],
            labels={"permutation_importance": "Change in validation PR-AUC", "feature": ""},
        )
        style_figure(chart, height=430)
        st.plotly_chart(
            chart,
            width="stretch",
            alt="Global permutation importance ranking for the saved model",
        )
        st.caption(
            "Permutation importance measures validation PR-AUC change when a feature "
            "is shuffled; it is a model diagnostic, not a causal effect."
        )

if metadata:
    config = ProjectConfig.load()
    spec = FeatureSpec.from_config(config)
    sessions = get_sessions()
    candidates = sessions[sessions["Converted"] == 1].head(250)
    with st.form("local_explanation_form"):
        row_number = st.number_input(
            "Example session row",
            min_value=0,
            max_value=max(0, len(candidates) - 1),
            value=0,
            step=1,
        )
        submitted = st.form_submit_button(
            "Explain this session", type="primary", icon=":material/search_insights:"
        )
    if submitted:
        raw = candidates.iloc[[int(row_number)]].loc[:, spec.input_features].copy()
        raw_session = {
            name: raw.iloc[0][name] for name in spec.input_features
        }
        result = predict_session(raw_session, include_explanations=False)
        model, _, _, _ = load_bundle()
        model_input = engineer_features(raw, spec).loc[:, spec.model_features]
        try:
            shap_values = local_shap_values(model, model_input).iloc[0]
        except (ImportError, ValueError, TypeError, AttributeError) as exc:
            st.error(f"SHAP could not explain this model: {exc}")
        else:
            shap_frame = (
                shap_values.rename("SHAP impact")
                .rename_axis("Behavior")
                .reset_index()
            )
            shap_frame["absolute impact"] = shap_frame["SHAP impact"].abs()
            shap_frame = shap_frame.nlargest(10, "absolute impact").sort_values(
                "SHAP impact"
            )
            chart = px.bar(
                shap_frame,
                x="SHAP impact",
                y="Behavior",
                orientation="h",
                color="SHAP impact",
                color_continuous_scale=["#F07B62", "#F4F6F8", PALETTE["teal"]],
                color_continuous_midpoint=0,
                labels={"SHAP impact": "Contribution to purchase score"},
            )
            style_figure(chart, height=400)
            st.plotly_chart(
                chart,
                width="stretch",
                alt="Local SHAP contributions to this session's purchase prediction",
            )
            st.metric(
                "Predicted purchase probability",
                f"{result['conversion_probability']:.1%}",
            )
            try:
                lime_values = local_lime_values(
                    model,
                    engineer_features(sessions, spec)
                    .loc[:, spec.model_features]
                    .sample(min(1000, len(sessions)), random_state=42),
                    model_input,
                )
            except (ImportError, ValueError, TypeError, AttributeError) as exc:
                st.warning(f"LIME explanation unavailable: {exc}")
            else:
                st.subheader("Independent LIME view")
                st.dataframe(
                    lime_values.rename("LIME weight")
                    .rename_axis("Encoded behavior")
                    .reset_index(),
                    hide_index=True,
                    alt="Local LIME feature weights for the selected session",
                )
                st.caption(
                    "LIME discretizes transformed input features locally. Compare the "
                    "direction of its strongest signals with grouped SHAP contributions."
                )

with st.expander("Model card · intended use & limitations"):
    st.markdown(
        """
        **Intended use:** prioritize late-session browsing patterns for analyst-led,
        controlled conversion experiments.

        **Not intended for:** identifying people, credit/pricing decisions, denying
        service, or claiming that an intervention causes a purchase.

        **Timing:** PageValues is deliberately included for stronger late-session
        scoring; do not deploy this model for early-session decisions unless that
        input is genuinely available and a separate evaluation supports it.

        **Limitations:** observational data, class imbalance, potential temporal and
        channel drift, proxy effects in technical/source identifiers, and scenario-
        dependent threshold economics.
        """
    )
