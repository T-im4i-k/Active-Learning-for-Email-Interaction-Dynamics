from abc import ABC, abstractmethod
import pandas as pd


class DataFrameBaseTransform(ABC):
    @abstractmethod
    def transform(self, mails: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class DataFrameTransformPipeline(DataFrameBaseTransform):
    def __init__(self, steps: list[DataFrameBaseTransform]):
        self.steps: list[DataFrameBaseTransform] = steps

    def transform(self, mails: pd.DataFrame) -> pd.DataFrame:
        for step in self.steps:
            mails = step.transform(mails)

        return mails


class ActiveUsersFilter(DataFrameBaseTransform):
    def __init__(self, min_activity: int) -> None:
        self.min_activity: int = min_activity

    def transform(self, mails: pd.DataFrame) -> pd.DataFrame:
        if self.min_activity > 0:
            user_activity = mails.groupby("user_id")["opened"].sum()
            active_users: pd.Index = user_activity[user_activity >= self.min_activity].index
            return mails[mails["user_id"].isin(active_users)]

        return mails


class LargeMailshotsFilter(DataFrameBaseTransform):
    def __init__(self, min_size: int) -> None:
        self.min_size: int = min_size

    def transform(self, mails: pd.DataFrame) -> pd.DataFrame:
        if self.min_size > 0:
            mailshot_sizes = mails.groupby("mailshot_id")["user_id"].count()
            large_mailshots: pd.Index = mailshot_sizes[mailshot_sizes >= self.min_size].index
            return mails[mails["mailshot_id"].isin(large_mailshots)]

        return mails


class DuplicatesDropper(DataFrameBaseTransform):
    def __init__(self, subset: tuple[str, ...]) -> None:
        self.subset: tuple[str, ...] = subset

    def transform(self, mails: pd.DataFrame) -> pd.DataFrame:
        return mails.drop_duplicates(subset=self.subset)


class ColumnSelector(DataFrameBaseTransform):
    def __init__(self, columns: tuple[str, ...]) -> None:
        self.columns: tuple[str, ...] = columns

    def transform(self, mails: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(mails[self.columns])


class DataFramePivoter(DataFrameBaseTransform):
    def __init__(self, index: str, column: str, values: list[str]) -> None:
        self.index: str = index
        self.column: str = column
        self.values: list[str] = values

    def transform(self, mails: pd.DataFrame) -> pd.DataFrame:
        return mails.pivot(index=self.index, columns=self.column, values=self.values)
