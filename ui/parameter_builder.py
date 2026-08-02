import streamlit as st

from strategies.registry import StrategyDefinition


def build_parameter_inputs(
    strategy: StrategyDefinition,
    key_prefix: str = "",
) -> dict:
    """
    Build Streamlit input widgets for every parameter
    defined by the selected strategy.
    """

    values = {}

    for parameter in strategy.parameters:

        widget_key = (
            f"{key_prefix}_{parameter.name}"
            if key_prefix
            else parameter.name
        )

        if parameter.parameter_type == "int":

            values[parameter.name] = st.number_input(
                parameter.label,
                min_value=int(parameter.minimum),
                max_value=int(parameter.maximum),
                value=int(parameter.default),
                step=int(parameter.step),
                key=widget_key,
            )

        elif parameter.parameter_type == "float":

            values[parameter.name] = st.number_input(
                parameter.label,
                min_value=float(parameter.minimum),
                max_value=float(parameter.maximum),
                value=float(parameter.default),
                step=float(parameter.step),
                key=widget_key,
            )

    return values