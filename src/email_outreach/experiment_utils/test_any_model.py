import argparse
import logging
import re
import sys
from pathlib import Path

from email_outreach.experiment_utils.experiment_constants import (
    EXPERIMENT_RESULTS_FOLDER_NAME,
)
from email_outreach.ml.baselines.ddqn_linkedin import DoubleDQNTrainer
from email_outreach.ml.baselines.factor_ucb import FactorUCBModel
from email_outreach.ml.baselines.fmfcdb import FMFCDBModel
from email_outreach.ml.baselines.fmpsal import FMPSALModel
from email_outreach.ml.baselines.fmrsal import FMRSALModel
from email_outreach.ml.baselines.model_based_rl import ModelBasedRL
from email_outreach.ml.baselines.random_model import RandomModel
from email_outreach.ml.baselines.thompson_sampling import ThompsonSamplingModel
from email_outreach.ml.shallow_autoencoder.contextual_bandit_with_autoencoder import (
    ContextualBanditWithAutoencoder,
)

# Ensure ROOT_FOLDER is in the system path
logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(name)s : %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument("--sender_id", type=int, help="Sender ID")
parser.add_argument(
    "--repetitions",
    type=int,
    default=1,
    help="Number of repetitions for each experiment",
)
parser.add_argument(
    "--split_sizes",
    nargs="+",
    default=[10],
    type=int,
    help="How should the dataset be split?",
)
parser.add_argument("--experiment_version", type=str, help="Experiment version")
parser.add_argument("--model", type=str, default="contextual_bandit", help="Model name")
parser.add_argument("--n_jobs", type=int, default=8, help="Number of parallel jobs")


def main():
    args = parser.parse_args([] if "__file__" not in globals() else None)
    sender_id = args.sender_id
    repetitions = args.repetitions
    experiment_version = args.experiment_version
    split_sizes = args.split_sizes
    model = args.model
    n_jobs = args.n_jobs
    logger.info(f"Experiment version: {experiment_version} for model {model}")

    if model == "contextual_bandit":
        model_class = ContextualBanditWithAutoencoder
    elif model == "ddqn":
        model_class = DoubleDQNTrainer
    elif model == "thompson_sampling":
        model_class = ThompsonSamplingModel
    elif model == "random":
        model_class = RandomModel
    elif model == "fmrsal":
        model_class = FMRSALModel
    elif model == "fmpsal":
        model_class = FMPSALModel
    elif model == "fmfcdb":
        model_class = FMFCDBModel
    elif model == "model_based_rl":
        model_class = ModelBasedRL
    elif model == "factor_ucb":
        model_class = FactorUCBModel
    else:
        raise ValueError(f"Unknown model: {model}")

    model_class.test(
        sender_id=sender_id,
        split_sizes=split_sizes,
        repetitions=repetitions,
        version=experiment_version,
        n_jobs = n_jobs
    )


if __name__ == "__main__":
    logger.info(f"Testing started...")
    main()
    logger.info(f"Starting testing {EXPERIMENT_RESULTS_FOLDER_NAME}")
