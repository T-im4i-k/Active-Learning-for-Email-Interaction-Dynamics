import argparse
import logging
from pathlib import Path

from email_outreach.utils.experiment_folder import ExperimentFolder, DEFAULT_CONFIG_NAME
from email_outreach.utils.json_config import JsonConfig

from email_outreach.ml.shallow_autoencoder.contextual_bandit_with_autoencoder import ContextualBanditWithAutoencoder

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(name)s : %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def validate_grid_search_config(config: JsonConfig) -> None:
    required_fields = ["param_grid"]
    missing_fields = [field for field in required_fields if field not in config.data]
    if missing_fields:
        raise ValueError(f"Missing required fields in config: {missing_fields}")


def _parse_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Run grid search based on a configuration file"
    )

    parser.add_argument(
        "--folder-path", type=Path, required=True, help="Path to config file directory"
    )
    parser.add_argument(
        "--config-name", type=str, default=DEFAULT_CONFIG_NAME, help="Config file name"
    )

    return parser.parse_args()


def main():
    args: argparse.Namespace = _parse_args()
    grid_search_experiment_folder: ExperimentFolder = ExperimentFolder(path=args.folder_path, config_name=args.config_name)
    grid_search_config: JsonConfig = JsonConfig.load(grid_search_experiment_folder)
    validate_grid_search_config(config=grid_search_config)

    logger.info(f"Starting grid search for config: {grid_search_experiment_folder.config_path}")
    ContextualBanditWithAutoencoder.grid_search(
        experiment_folder = grid_search_experiment_folder,
        config = grid_search_config
    )
    logger.info("Grid search completed!")


if __name__ == "__main__":
    main()
