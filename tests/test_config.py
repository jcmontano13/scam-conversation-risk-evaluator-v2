"""
test_config.py

Description:
    Checks that config.py contains the COMP903 settings and that the output
    paths are valid. This confirms every new script can rely on config.py
    for its paths and settings.

Run:
    - pytest tests/test_config.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import config


def get_output_paths():
    """Returns every Path setting in config.py that points inside outputs/."""
    return {
        name: value
        for name, value in vars(config).items()
        if name.isupper() and isinstance(value, Path) and "outputs" in value.parts
    }


def test_seed_is_unchanged():
    assert config.SEED == 42


def test_five_models_are_listed():
    assert len(config.MODEL_NAMES) == 5
    assert len(set(config.MODEL_NAMES)) == 5
    assert config.REFERENCE_MODEL in config.BASELINE_MODELS


def test_feature_columns_match_comp902():
    assert config.FEATURE_COLUMNS == [
        "turn_count",
        "aggregate_num_urls",
        "aggregate_money_terms",
        "ratio_request_messages",
    ]
    assert config.TARGET_COLUMN == "thread_label"


def test_output_paths_are_unique():
    paths = list(get_output_paths().values())
    assert len(paths) == len(set(paths)), "Two settings point to the same output file"


def test_output_paths_are_inside_project():
    for name, path in get_output_paths().items():
        assert PROJECT_ROOT in path.parents, f"{name} is outside the project folder"



def test_repeated_cv_seed_differs_from_tuning_seed():
    assert config.REPEATED_CV_SEED != config.SEED