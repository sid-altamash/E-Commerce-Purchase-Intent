import plotly.express as px
import streamlit as st

from dashboard.shared import (
    PALETTE,
    format_percent,
    get_sessions,
    model_state,
    style_figure,
    title_block,
)
from src.config import ProjectConfig

sessions = get_sessions()
ready, metadata = model_state()
config = ProjectConfig.load()
rate = float(sessions["Converted"].mean())
buyers = int(sessions["Converted"].sum())

title_block(
    "Know which visits are ready to convert.",
    "A transparent purchase-intent workspace for session-level decisions, "
    "conversion patterns, and responsible experimentation.",
    "Commerce intelligence · Executive overview",
)

with st.container(horizontal=True):
    st.metric("Sessions analyzed", f"{len(sessions):,}", border=True)
    st.metric("Observed conversion", format_percent(rate), border=True)
    st.metric("Purchases", f"{buyers:,}", border=True)
    st.metric(
        "Decision model",
        metadata["best_model"] if metadata else "Not trained",
        border=True,
    )

st.space("small")
left, right = st.columns([1.55, 1], gap="large")
with left:
    with st.container(border=True):
        st.subheader("Conversion momentum")
        monthly = (
            sessions.groupby("Month", observed=True)["Converted"]
            .agg(sessions="size", conversions="sum", conversion_rate="mean")
            .reset_index()
        )
        month_order = config.data["months"]
        monthly["sort"] = monthly["Month"].map(
            {month: index for index, month in enumerate(month_order)}
        )
        monthly = monthly.sort_values("sort")
        chart = px.bar(
            monthly,
            x="Month",
            y="sessions",
            color="conversion_rate",
            color_continuous_scale=[PALETTE["cyan"], PALETTE["teal"], PALETTE["navy"]],
            hover_data={"conversions": True, "conversion_rate": ":.1%"},
            labels={"sessions": "Sessions", "conversion_rate": "Conversion rate"},
        )
        style_figure(chart)
        st.plotly_chart(
            chart,
            width="stretch",
            alt="Monthly session counts shaded by observed conversion rate",
        )
        st.caption("Takeaway: compare seasonal traffic volume with conversion quality.")

with right:
    with st.container(border=True):
        st.subheader("Intent signals")
        by_outcome = (
            sessions.assign(
                outcome=sessions["Converted"].map({0: "No purchase", 1: "Purchase"})
            )
            .groupby("outcome", observed=True)
            .agg(
                page_value=("PageValues", "median"),
                exit_rate=("ExitRates", "median"),
                product_pages=("ProductRelated", "median"),
            )
            .reset_index()
        )
        chart = px.bar(
            by_outcome.melt(
                id_vars="outcome", var_name="behavior", value_name="median value"
            ),
            x="behavior",
            y="median value",
            color="outcome",
            barmode="group",
            facet_col="behavior",
            facet_col_wrap=3,
            color_discrete_sequence=[PALETTE["muted"], PALETTE["teal"]],
        )
        chart.update_yaxes(matches=None)
        chart.update_xaxes(showticklabels=False, title_text="")
        style_figure(chart, height=360)
        st.plotly_chart(
            chart,
            width="stretch",
            alt="Median page value, exit rate, and product pages by purchase outcome",
        )
        st.caption(
            "Takeaway: product engagement, exit behavior, and page value each separate "
            "outcomes on their own scale."
        )

if metadata:
    st.success(
        f"Model ready · {metadata['best_model']} · validation threshold "
        f"{metadata['threshold']:.2f} · test PR-AUC "
        f"{metadata['test_metrics']['pr_auc']:.3f}"
    )
else:
    st.info(
        "Next step: train and evaluate the six-model benchmark with "
        "`python -m src.train`. Dashboard analytics are available now."
    )

with st.expander("How to interpret this score"):
    st.write(
        "Predictions are designed for late-session use and include PageValues. "
        "They estimate association, not causal response to a discount or banner. "
        "Use randomized experiments to measure whether an intervention changes outcomes."
    )
