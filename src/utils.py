import pandas as pd
import os
import logging
from datetime import datetime


def persist_df(df: pd.DataFrame,
               file_path: str,
               name: str) -> None:
    """
    Persist a dataframe to a specified file path and file name

    Parameters
    __________________
    df: pd.DataFrame
        the dataframe to persist

    file_path: str
        the path to create if it does not exist example your/path

    name: str
        name of the file

    Return
    __________________
    None
    """
    # persist file
    today = datetime.now().isoformat()
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    file_name = f"{file_path}/{name}_{today}.csv"
    logging.info(f"persisting file:\t{file_name}")
    df.to_csv(file_name)