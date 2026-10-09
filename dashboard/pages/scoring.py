import pandas as pd
import streamlit as st

from dashboard.shared import get_sessions, require_trained_model, title_block
from src.config import FeatureSpec, ProjectConfig
from src.service import predict_session

title_block(
    "Score a live session",
    "Enter the current visit signals to estimate conversion likelihood and inspect "
    "the strongest local drivers.",
    "Decision workspace · Session scoring",
)

metadata = require_trained_model()
sessions = get_sessions()
config = ProjectConfig.load()
spec = FeatureSpec.from_config(config)
converted_sessions = sessions[sessions["Converted"] == 1]
example_numeric = converted_sessions.select_dtypes(include="number").median()

if metadata:
    with st.container(border=True):
        st.caption(
            f"Model: {metadata['best_model']} · Decision threshold: "
            f"{metadata['threshold']:.2f} · Input timing: late session"
        )
        st.caption(
            "Example inputs start from the purchasing-session medians and modes; "
            "edit them to reflect the scenario you want to score."
        )
        with st.form("session_scoring_form"):
            st.subheader("Session behavior")
            first, second, third = st.columns(3)
            payload = {}
            numeric_fields = [
                ("Administrative", "Admin pages", 0.0, 100.0),
                ("Administrative_Duration", "Admin time (sec)", 0.0, 3600.0),
                ("Informational", "Info pages", 0.0, 100.0),
                ("Informational_Duration", "Info time (sec)", 0.0, 3600.0),
                ("ProductRelated", "Product pages", 1.0, 100.0),
                ("ProductRelated_Duration", "Product time (sec)", 300.0, 10000.0),
                ("BounceRates", "Bounce rate", 0.02, 0.0),
                ("ExitRates", "Exit rate", 0.05, 0.0),
                ("PageValues", "Page value", 0.0, 200.0),
                ("SpecialDay", "Special-day proximity", 0.0, 1.0),
            ]
            columns = [first, second, third]
            for index, (name, label, default, maximum) in enumerate(numeric_fields):
                default = float(example_numeric.get(name, default))
                with columns[index % 3]:
                    if name in {"Administrative", "Informational", "ProductRelated"}:
                        payload[name] = st.number_input(
                            label,
                            min_value=0,
                            max_value=int(maximum),
                            value=int(round(default)),
                            step=1,
                            key=f"score_{name}",
                        )
                    elif name in {"BounceRates", "ExitRates", "SpecialDay"}:
                        payload[name] = st.number_input(
                            label,
                            min_value=0.0,
                            max_value=1.0,
                            value=default,
                            step=0.01,
                            format="%.3f",
                            key=f"score_{name}",
                        )
                    else:
                        payload[name] = st.number_input(
                            label,
                            min_value=0.0,
                            max_value=float(maximum),
                            value=float(default),
                            step=1.0 if name in {"Administrative", "Informational", "ProductRelated"} else 10.0,
                            key=f"score_{name}",
                        )
            category_values = {
                "Month": config.data["months"],
                "OperatingSystems": sorted(sessions["OperatingSystems"].unique()),
                "Browser": sorted(sessions["Browser"].unique()),
                "Region": sorted(sessions["Region"].unique()),
                "TrafficType": sorted(sessions["TrafficType"].unique()),
                "VisitorType": sorted(sessions["VisitorType"].unique()),
                "Weekend": sorted(sessions["Weekend"].unique()),
            }
            st.subheader("Session context")
            category_columns = st.columns(4)
            for index, (name, options) in enumerate(category_values.items()):
                positive_modes = converted_sessions[name].mode()
                default_category = (
                    positive_modes.iloc[0]
                    if not positive_modes.empty and positive_modes.iloc[0] in options
                    else options[0]
                )
                with category_columns[index % 4]:
                    payload[name] = st.selectbox(
                        name.replace("_", " "),
                        options,
                        index=options.index(default_category),
                        key=f"score_category_{name}",
                    )
            submitted = st.form_submit_button(
                "Generate session forecast", type="primary", icon=":material/bolt:"
            )

    if submitted:
        result = predict_session(payload)
        probability = result["conversion_probability"]
        st.space("small")
        first, second, third = st.columns([1, 1, 1.5])
        first.metric("Purchase likelihood", f"{probability:.1%}", border=True)
        second.metric(
            "Decision", "Likely" if result["prediction"] else "Unlikely", border=True
        )
        third.metric("Confidence", f"{result['confidence']:.1%}", border=True)
        st.caption(
            "Confidence is the larger of the predicted class probabilities; it is "
            "not a calibrated probability-quality guarantee."
        )
        if result["contributors"]:
            impacts = pd.DataFrame(result["contributors"])
            impacts["direction"] = impacts["impact"].map(
                lambda value: "Supports purchase" if value > 0 else "Reduces likelihood"
            )
            st.subheader("What influenced this estimate")
            st.dataframe(
                impacts[["feature", "direction", "impact"]],
                hide_index=True,
                column_config={
                    "feature": st.column_config.TextColumn("Behavior"),
                    "direction": st.column_config.TextColumn("Direction"),
                    "impact": st.column_config.NumberColumn(
                        "SHAP impact", format="%.4f"
                    ),
                },
                alt="Top five local SHAP feature impacts",
            )
            st.caption(
                "SHAP contributions explain the model output; they do not imply that "
                "changing a behavior will cause a purchase."
            )
        else:
            st.warning(
                "This estimator does not expose a supported local SHAP explanation. "
                "The probability is valid, but the attribution is unavailable."
            )
