import argparse
import logging
from pathlib import Path
from email_outreach.ml.shallow_autoencoder.contextual_bandit_with_autoencoder import (
    ContextualBanditWithAutoencoder,
)

from email_outreach.utils.experiment_folder import DEFAULT_CONFIG_NAME, ExperimentFolder
from utils.json_config import JsonConfig

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(name)s : %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Run tests based on a configuration file"
    )

    parser.add_argument(
        "--folder-path", type=Path, required=True, help="Path to config file directory"
    )
    parser.add_argument(
        "--config-name", type=str, default=DEFAULT_CONFIG_NAME, help="Config file name"
    )

    return parser.parse_args()


def validate_test_config(config: JsonConfig) -> None:
    ...


def main():
    args: argparse.Namespace = _parse_args()
    test_experiment_folder: ExperimentFolder = ExperimentFolder(path=args.folder_path, config_name=args.config_name)
    test_config: JsonConfig = JsonConfig.load(test_experiment_folder)
    validate_test_config(config=test_config)

    logger.info(f"Starting testing for config in: {test_experiment_folder.config_path}")
    ContextualBanditWithAutoencoder.test(
        experiment_folder=test_experiment_folder,
        config=test_config
    )
    logger.info("Testing completed!")


if __name__ == "__main__":
    main()
