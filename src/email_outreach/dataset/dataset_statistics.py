# TODO


# import pandas as pd
# import torch
#
#
# def get_user_open_rate(mails: pd.DataFrame) -> torch.Tensor:
#     opens = mails.groupby("user_id")["opened"].sum().sort_index()
#     count = mails.groupby("user_id")["opened"].count().sort_index()
#     return torch.tensor(opens / count)
#
#
# def count_opens_for_user(mails: pd.DataFrame, user_id: int) -> int:
#     return int(mails[mails["user_id"] == user_id]["opened"].sum())
