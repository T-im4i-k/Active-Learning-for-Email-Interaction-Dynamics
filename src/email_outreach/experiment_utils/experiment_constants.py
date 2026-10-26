from pathlib import Path

from email_outreach.utils.constants import ROOT_FOLDER

EXPERIMENTS_FOLDER = ROOT_FOLDER / "experiments"
TEST_SET_FOLDER = EXPERIMENTS_FOLDER / "test_set"

CONTEXTUAL_BANDIT_FOLDER_NAME = "contextual_bandit"
EXPERIMENT_RESULTS_FOLDER_NAME = "exp"

EXPERIMENTS_FOLDER.mkdir(parents=True, exist_ok=True)
GRID_SEARCH_FILE_NAME = "grid_search_config.json"


def get_experiment_folder_path(sender_id: int, model_name: str, version: str) -> Path:
    return EXPERIMENTS_FOLDER / str(sender_id) / model_name / version


def get_test_folder_path(sender_id: int, model_name: str, version: str) -> Path:
    if version is None:
        return Path()
    return TEST_SET_FOLDER / str(sender_id) / model_name / version
