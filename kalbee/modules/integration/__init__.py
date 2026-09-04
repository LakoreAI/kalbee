from kalbee.modules.integration.factor_graph import FactorGraphExporter

__all__ = ["filter_dataframe", "filter_series", "FactorGraphExporter"]


def filter_dataframe(df, filter_obj, measurement_columns=None, **kwargs):
    """
    Auto-detect DataFrame type and run filter.

    Supports both pandas and Polars DataFrames.
    """
    # Detect DataFrame type
    module_name = type(df).__module__

    if "polars" in module_name:
        try:
            from kalbee.modules.integration.polars import filter_dataframe as _filter

            return _filter(df, filter_obj, measurement_columns, **kwargs)
        except ImportError:
            raise ImportError(
                "Polars is required for Polars DataFrames. "
                "Install it with: pip install kalbee[polars]"
            )
    elif "pandas" in module_name:
        try:
            from kalbee.modules.integration.pandas import filter_dataframe as _filter

            return _filter(df, filter_obj, measurement_columns, **kwargs)
        except ImportError:
            raise ImportError(
                "pandas is required for pandas DataFrames. "
                "Install it with: pip install kalbee[pandas]"
            )
    else:
        raise TypeError(
            f"Unsupported DataFrame type: {type(df).__name__}. "
            "Use pandas or Polars DataFrames."
        )


def filter_series(series, filter_obj, dt=1.0):
    """
    Auto-detect Series type and run filter.

    Supports both pandas and Polars Series.
    """
    module_name = type(series).__module__

    if "polars" in module_name:
        try:
            from kalbee.modules.integration.polars import filter_series as _filter

            return _filter(series, filter_obj, dt)
        except ImportError:
            raise ImportError(
                "Polars is required for Polars Series. "
                "Install it with: pip install kalbee[polars]"
            )
    elif "pandas" in module_name:
        try:
            from kalbee.modules.integration.pandas import filter_series as _filter

            return _filter(series, filter_obj, dt)
        except ImportError:
            raise ImportError(
                "pandas is required for pandas Series. "
                "Install it with: pip install kalbee[pandas]"
            )
    else:
        raise TypeError(
            f"Unsupported Series type: {type(series).__name__}. "
            "Use pandas or Polars Series."
        )
