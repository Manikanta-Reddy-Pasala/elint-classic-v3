import csv
import re
from pathlib import Path

def try_float_int(value: str) -> bool | float | int | str:
    if re.match(r"^[+-]?\d+$", value):
        return int(value)
    elif re.search(r"^[+-]?\d+\.\d+$", value):
        return float(value)
    elif value in ["True", "False"]:
        return True if value == "True" else False
    else:
        return value


def read_test_parameters(csv_file_path: Path, data_file: str) -> list[tuple[bool | float | int | str, ...]]:
    """Read CSV file and return a list of integer tuples"""
    csv_path = csv_file_path / data_file
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)
        return [tuple(try_float_int(cell) for cell in row) for row in reader]


def _parse_emitters_id(value: str | int) -> list[int]:
    if isinstance(value, str):
        return [int(x) for x in value.split()]
    return [int(value)]