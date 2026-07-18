from email_outreach.dataset.pipeline import (
    DataFrameTransformPipeline,
    ColumnSelector,
    DuplicatesDropper,
    ActiveUsersFilter,
    LargeMailshotsFilter,
    DataFramePivoter,
)

REQUIRED_COLUMNS: tuple[str, ...] = ("mailshot_id", "user_id", "opened", "time_to_open")
ID_COLUMNS: tuple[str, ...] = ("mailshot_id", "user_id")
MAILSHOT_MIN_SIZE: int = 100
USER_MIN_ACTIVITY: int = 1

PIVOT_INDEX: str = "mailshot_id"
PIVOT_COLUMN: str = "user_id"
PIVOT_VALUES: list[str] = ["opened", "time_to_open"]

DEFAULT_PIPELINE: DataFrameTransformPipeline = DataFrameTransformPipeline([
    ColumnSelector(columns=REQUIRED_COLUMNS),
    DuplicatesDropper(subset=ID_COLUMNS),
    ActiveUsersFilter(min_activity=USER_MIN_ACTIVITY),
    LargeMailshotsFilter(min_size=MAILSHOT_MIN_SIZE),
    DataFramePivoter(index=PIVOT_INDEX, column=PIVOT_COLUMN, values=PIVOT_VALUES),
])
