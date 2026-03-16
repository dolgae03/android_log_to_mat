"""Standalone Android GNSS txt parser."""

from importlib import import_module

__all__ = [
    "AndroidGnssTxtParser",
    "convert_txt_to_tsv",
    "dataframe_to_measurements",
    "parse_txt_to_dataframe",
    "parse_txt_to_rows",
    "rows_to_measurements",
    "write_measurements_tsv",
]


def __getattr__(name):
    if name in __all__:
        module = import_module(".parser", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
