from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True, slots=True)
class MailshotUserDataset:
    data: pd.DataFrame

    @property
    def opened(self) -> pd.DataFrame:
        return self.data["opened"]

    @property
    def time_to_open(self) -> pd.DataFrame:
        return self.data["time_to_open"]

    @property
    def shape(self) -> tuple[int, int]:
        return self.data["opened"].shape

    @property
    def mailshot_ids(self) -> pd.Index:
        return self.opened.index

    @property
    def num_mailshots(self) -> int:
        return len(self.mailshot_ids)

    @property
    def num_users(self) -> int:
        return len(self.user_ids)

    @property
    def user_ids(self) -> pd.Index:
        return self.opened.columns