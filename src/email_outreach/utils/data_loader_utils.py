import logging
from pathlib import Path

import pandas as pd

from email_outreach.utils.constants import ROOT_FOLDER

logger = logging.getLogger(__name__)

DATA_FOLDER = ROOT_FOLDER / Path("data") / "subsets"

def load_data_for_sender(sender_id: int) -> pd.DataFrame:
    """Load data for a sender from a parquet file."""
    try:
        return pd.read_parquet(DATA_FOLDER / f"sender_mails_{sender_id}.parquet")
    except FileNotFoundError as e:
        logger.warning(f"No data found for sender {sender_id}")
        raise e
