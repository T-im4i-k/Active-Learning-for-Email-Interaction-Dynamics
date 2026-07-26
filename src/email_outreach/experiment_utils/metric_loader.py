from email_outreach.ml.shallow_autoencoder.contextual_bandit_with_autoencoder import ContextualBanditMetrics

import re
from pathlib import Path


class ResultsFileFinder:
    """Finds results.csv.gz files inside exp_X or exp_X_Y subfolders."""

    # Matches folder names like exp_1, exp_12, exp_1_2, etc.
    DEFAULT_FOLDER_PATTERN: re.Pattern = re.compile(r"^exp_\d+(_\d+)?$")
    DEFAULT_RESULTS_FILENAME: str = "results.csv.gz"

    def __init__(self, root: str | Path):
        self.root: Path = Path(root)

    def find(self, folder_pattern: re.Pattern = DEFAULT_FOLDER_PATTERN,
             results_file_name: str = DEFAULT_RESULTS_FILENAME) -> list[Path]:
        """Return paths to all results.csv.gz files in matching exp_ folders."""
        found: list[Path] = []
        for entry in sorted(self.root.iterdir()):
            if entry.is_dir() and folder_pattern.match(entry.name):
                candidate: Path = entry / results_file_name
                if candidate.is_file():
                    found.append(entry)
        return found


def load_experiment_metrics(experiment_folder: Path) -> list[list[ContextualBanditMetrics]]:
    results_files: list[Path] = ResultsFileFinder(experiment_folder).find()
    return [ContextualBanditMetrics.from_csv_gz(results_file) for results_file in results_files]
