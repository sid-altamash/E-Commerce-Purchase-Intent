import plotly.express as px
import streamlit as st

from dashboard.shared import PALETTE, get_sessions, style_figure, title_block
from src.config import ProjectConfig

title_block(
    "Read the journey behind conversion",
    "Explore how seasonality, visitor type, product engagement, and exit behavior "
    "relate to purchase outcomes.",
    "Behavior intelligence · Funnel & cohorts",
)

sessions = get_sessions()
config = ProjectConfig.load()
with st.sidebar:
    st.subheader("Analytics filters")
    visitor_types = sorted(sessions["VisitorType"].unique())
    selected_visitors = st.multiselect(
        "Visitor cohorts", visitor_types, default=visitor_types
    )
    include_weekends = st.toggle("Include weekend sessions", value=True)

filtered = sessions[sessions["VisitorType"].isin(selected_visitors)]
if not include_weekends:
    filtered = filtered[filtered["Weekend"] == "False"]

with st.container(horizontal=True):
    st.metric("Filtered sessions", f"{len(filtered):,}", border=True)
    st.metric(
        "Filtered conversion",
        f"{filtered['Converted'].mean():.1%}" if len(filtered) else "—",
        border=True,
    )
    st.metric(
        "Returning visitors",
        f"{(filtered['VisitorType'] == 'Returning_Visitor').mean():.1%}"
        if len(filtered)
        else "—",
        border=True,
    )

monthly = (
    filtered.groupby("Month", observed=True)
    .agg(
        sessions=("Converted", "size"),
        conversion_rate=("Converted", "mean"),
        conversions=("Converted", "sum"),
    )
    .reset_index()
)
order = {month: index for index, month in enumerate(config.data["months"])}
monthly["sort_order"] = monthly["Month"].map(order)
monthly = monthly.sort_values("sort_order")
left, right = st.columns(2)
with left:
    with st.container(border=True):
        st.subheader("Conversion by month")
        chart = px.line(
            monthly,
            x="Month",
            y="conversion_rate",
            markers=True,
            color_discrete_sequence=[PALETTE["teal"]],
            hover_data={"sessions": True, "conversions": True, "conversion_rate": ":.1%"},
            labels={"conversion_rate": "Conversion rate"},
        )
        chart.update_yaxes(tickformat=".0%")
        style_figure(chart)
        st.plotly_chart(
            chart,
            width="stretch",
            alt="Observed purchase conversion rate by month",
        )
        st.caption("Takeaway: examine whether high-intent periods align with campaign timing.")

with right:
    with st.container(border=True):
        st.subheader("Visitor cohort performance")
        cohort = (
            filtered.groupby("VisitorType", observed=True)["Converted"]
            .agg(sessions="size", conversion_rate="mean")
            .reset_index()
        )
        chart = px.bar(
            cohort,
            x="VisitorType",
            y="conversion_rate",
            color="VisitorType",
            text_auto=".1%",
            color_discrete_sequence=[PALETTE["teal"], PALETTE["gold"], PALETTE["coral"]],
            hover_data={"sessions": True},
        )
        chart.update_yaxes(tickformat=".0%")
        style_figure(chart)
        st.plotly_chart(
            chart,
            width="stretch",
            alt="Conversion rate by new, returning, and other visitor groups",
        )
        st.caption("Takeaway: evaluate interventions separately for each visitor cohort.")

with st.container(border=True):
    st.subheader("Special-day proximity")
    special_day = (
        filtered.groupby("SpecialDay", observed=True)["Converted"]
        .agg(sessions="size", conversion_rate="mean")
        .reset_index()
        .sort_values("SpecialDay")
    )
    chart = px.bar(
        special_day,
        x="SpecialDay",
        y="conversion_rate",
        color="conversion_rate",
        color_continuous_scale=[PALETTE["cyan"], PALETTE["teal"], PALETTE["navy"]],
        hover_data={"sessions": True, "conversion_rate": ":.1%"},
        labels={"conversion_rate": "Conversion rate", "SpecialDay": "Proximity"},
    )
    chart.update_yaxes(tickformat=".0%")
    style_figure(chart, height=300)
    st.plotly_chart(
        chart,
        width="stretch",
        alt="Purchase rate by the session's proximity to a special shopping day",
    )
    st.caption(
        "Takeaway: special-day context is associated with conversion patterns, but "
        "validate each period before shifting campaign budgets."
    )

with st.container(border=True):
    st.subheader("Behavioral funnel")
    funnel = (
        filtered.assign(
            engagement=filtered["ProductRelated"].map(
                lambda pages: "No product pages" if pages == 0 else "Product pages viewed"
            )
        )
        .groupby(["engagement", "Converted"], observed=True)
        .size()
        .reset_index(name="sessions")
    )
    funnel["outcome"] = funnel["Converted"].map({0: "No purchase", 1: "Purchase"})
    chart = px.bar(
        funnel,
        x="engagement",
        y="sessions",
        color="outcome",
        barmode="group",
        color_discrete_map={"No purchase": PALETTE["muted"], "Purchase": PALETTE["teal"]},
        labels={"engagement": "Product engagement"},
    )
    style_figure(chart, height=320)
    st.plotly_chart(
        chart,
        width="stretch",
        alt="Session outcomes split by whether product pages were viewed",
    )
    st.caption(
        "Takeaway: product-page engagement is an observational signal, not proof of "
        "incremental campaign impact."
    )

left, right = st.columns(2)
with left:
    with st.container(border=True):
        st.subheader("Page value and exit behavior")
        chart = px.scatter(
            filtered.sample(min(len(filtered), 1800), random_state=42),
            x="ExitRates",
            y="PageValues",
            color=filtered.sample(min(len(filtered), 1800), random_state=42)[
                "Converted"
            ].map({0: "No purchase", 1: "Purchase"}),
            opacity=0.48,
            color_discrete_map={
                "No purchase": PALETTE["muted"],
                "Purchase": PALETTE["teal"],
            },
            labels={"ExitRates": "Exit rate", "PageValues": "Page value"},
        )
        style_figure(chart, height=320)
        st.plotly_chart(
            chart,
            width="stretch",
            alt="Relationship between session exit rate and page value, colored by outcome",
        )
        st.caption("Takeaway: PageValues can dominate after it becomes available late in a visit.")

with right:
    with st.container(border=True):
        st.subheader("Dataset snapshot")
        st.dataframe(
            filtered[["Month", "VisitorType", "Weekend", "ProductRelated", "PageValues", "Converted"]]
            .head(12),
            hide_index=True,
            alt="Sample of filtered e-commerce session records",
        )
