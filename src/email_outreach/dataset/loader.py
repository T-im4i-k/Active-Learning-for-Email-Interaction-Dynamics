from email_outreach.dataset.dataset import MailshotUserDataset
from email_outreach.dataset.pipeline import DataFrameTransformPipeline

from pathlib import Path
import pandas as pd


def _load_parquet_dataset(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_mailshot_user_dataset(path: Path, pipeline: DataFrameTransformPipeline) -> MailshotUserDataset:
    df_raw: pd.DataFrame = _load_parquet_dataset(path=path)
    df_transformed: pd.DataFrame = pipeline.transform(df_raw)

    return MailshotUserDataset(
        data=df_transformed
    )
