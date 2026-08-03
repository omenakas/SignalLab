import streamlit as st

from strategies.registry import StrategyDefinition
from decimal import Decimal


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

def _build_parameter_values(
    start: int | float,
    stop: int | float,
    step: int | float,
    parameter_type: str,
) -> list[int | float]:
    """
    Generate inclusive parameter values without accumulating
    floating-point errors.
    """

    start_decimal = Decimal(str(start))
    stop_decimal = Decimal(str(stop))
    step_decimal = Decimal(str(step))

    if step_decimal <= 0:
        raise ValueError("Optimization step must be positive.")

    values: list[int | float] = []
    current = start_decimal

    while current <= stop_decimal:
        if parameter_type == "int":
            values.append(int(current))
        else:
            values.append(float(current))

        current += step_decimal

    return values


def build_optimization_grid_inputs(
    strategy: StrategyDefinition,
    key_prefix: str = "optimizer",
) -> dict[str, list[int | float]]:
    """
    Build optimization-range controls for every registered strategy
    parameter and return the resulting parameter grid.
    """

    parameter_grid: dict[str, list[int | float]] = {}

    columns = st.columns(
        min(len(strategy.parameters), 3)
    )

    for index, parameter in enumerate(strategy.parameters):
        column = columns[index % len(columns)]

        with column:
            st.markdown(f"**{parameter.label}**")

            if parameter.parameter_type == "int":
                selected_range = st.slider(
                    f"{parameter.label} range",
                    min_value=int(parameter.minimum),
                    max_value=int(parameter.maximum),
                    value=(
                        int(parameter.optimization_minimum),
                        int(parameter.optimization_maximum),
                    ),
                    step=int(parameter.step),
                    key=f"{key_prefix}_{parameter.name}_range",
                )

                selected_step = st.number_input(
                    f"{parameter.label} optimizer step",
                    min_value=1,
                    max_value=max(
                        1,
                        int(parameter.maximum)
                        - int(parameter.minimum),
                    ),
                    value=int(parameter.optimization_step),
                    step=1,
                    key=f"{key_prefix}_{parameter.name}_step",
                )

            else:
                selected_range = st.slider(
                    f"{parameter.label} range",
                    min_value=float(parameter.minimum),
                    max_value=float(parameter.maximum),
                    value=(
                        float(parameter.optimization_minimum),
                        float(parameter.optimization_maximum),
                    ),
                    step=float(parameter.step),
                    key=f"{key_prefix}_{parameter.name}_range",
                )

                selected_step = st.number_input(
                    f"{parameter.label} optimizer step",
                    min_value=float(parameter.step),
                    max_value=float(
                        parameter.maximum - parameter.minimum
                    ),
                    value=float(parameter.optimization_step),
                    step=float(parameter.step),
                    key=f"{key_prefix}_{parameter.name}_step",
                )

            parameter_grid[parameter.name] = (
                _build_parameter_values(
                    start=selected_range[0],
                    stop=selected_range[1],
                    step=selected_step,
                    parameter_type=parameter.parameter_type,
                )
            )

    return parameter_grid