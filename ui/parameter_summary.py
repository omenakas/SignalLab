from __future__ import annotations

import streamlit as st

from ui.parameter_guidance import (
    get_research_guidance,
)

from strategies.registry import (
    get_strategy,
)


def render_parameter_summary(
    comparison_parameters: dict[
        str,
        dict[str, int | float],
    ],
) -> None:
    """
    Display a summary of the currently selected
    strategy parameters.
    """

    st.markdown(
        "### 📋 Current Research Configuration"
    )

    for (
        strategy_name,
        parameter_values,
    ) in comparison_parameters.items():

        strategy = get_strategy(
            strategy_name
        )

        with st.expander(
            strategy_name,
            expanded=True,
        ):

            labels = {
                parameter.name: parameter.label
                for parameter in strategy.parameters
            }

            for (
                parameter_name,
                value,
            ) in parameter_values.items():

                label = labels.get(
                    parameter_name,
                    parameter_name,
                )

                label_column, value_column = st.columns(
                    [3, 1]
                )

                with label_column:
                    st.markdown(
                        f"**{label}**"
                    )

                with value_column:
                    st.markdown(
                        f"`{value}`"
                    )

            guidance_items = get_research_guidance(
                strategy_name,
                parameter_values,
            )

            for guidance in guidance_items:
                st.success(
                    f"📚 {guidance}"
                )