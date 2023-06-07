import pandas as pd
import logging
import sys
import os
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from datetime import datetime


"""
Feature engineering for the columns in the dataset. Object columns
such as state_id, and category will be rainbow encoded. All numerical
columns will be lagged, logged or differenced.
"""
# Reusuable functions
def init_logger() -> None:
    """Initialize logging, creating necessary folder if it doesn't already exist"""
    # Assume script is called from top-level directory
    log_dir = Path("logs")
    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    # Configue handlers to print logs to file and std out
    file_handler = logging.FileHandler(filename="logs/feature_engine.log")
    stdout_handler = logging.StreamHandler(sys.stdout)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[file_handler, stdout_handler],
    )

######################## Numerical Features ########################

def log_transform(df: pd.DataFrame, col: any, alpha: float = 0.01) -> None:
    """
    perform log transformation on the column of choice
    Paramters
    ______________
    df: pd.DataFrame
        the dataframe hosting time series values
    col: list or str
        the column name or list of column names
    alpha: float
        value to add to the column values to prevent inf values

    Returns
    ______________
    None
    """
    if isinstance(col, list):
        for x in col:
            logging.info(f"performing log transform on {x}")
            vals = df[x] + alpha
            df[f"{x}_log"] = np.log(vals)
    else:
        logging.info(f"performing log transform on {col}")
        vals = df[col] + alpha
        df[f"{col}_log"] = np.log(vals)


def diff_transform(df: pd.DataFrame, col: any, val: int = 1) -> None:
    """
    perform difference transformation on the column of choice
    Paramters
    ______________
    df: pd.DataFrame
        the dataframe hosting time series values
    col: list or str
        the column name or list of column names
    val: int
        The difference value

    Returns
    ______________
    None
    """
    if isinstance(col, list):
        for x in col:
            # df.sort_values(by="Month")
            df[f"{x}_{str(val)}_diff"] = df[x].diff(val)
    else:
        # df.sort_values(by="Month")
        df[f"{col}_{str(val)}_diff"] = df[col].diff(val)


def lag_transform(df: pd.DataFrame, col: any, val: int = 1) -> None:
    """
    perform lag transformation on the column of choice
    Paramters
    ______________
    df: pd.DataFrame
        the dataframe hosting time series values
    col: list or str
        the column name or list of column names
    val: int
        The difference value

    Returns
    ______________
    None
    """
    if isinstance(col, list):
        for x in col:
            # logging.info(f"performing lag transform on {x}")
            # df.sort_values(by="date")
            df[f"{x}_{str(val)}_lag"] = df[x].shift(val)
    else:
        # logging.info(f"performing log transform on {col}")
        # df.sort_values(by="date")
        df[f"{col}_{str(val)}_lag"] = df[col].shift(val)

def rolling_window(df:pd.DataFrame,
                   col: str,
                   by: any,
                   window_size: int = 4,
                   agg: str = "mean") -> pd.DataFrame:
    """
    Calculates the rolling window average of the passed column
    based on the groupby passed
    Parameters:
    _______________
    df: pd.DataFrame - dataframe that will be transformed
    col: str - the column the average will be calculated on
    by: How to groupby the dataframe
    agg: str - can be either mean or median aggregation pass 'mean' or 'median'
    """
    # iterate over all the groupby df and calculate the rolling window
    dfs = []
    logging.info(f"calculating the rolling window for {col} groupby {by}...")
    for group, group_df in df.groupby(by=by):
        # sort the data by date_max
        group_df.sort_values(by="date_max", inplace=True)
        if agg == 'mean':
            group_df[f"rolling_avg_{col}_{window_size}"] = group_df[col].rolling(window=window_size,
                                                                                 min_periods=2,
                                                                                 closed='left').mean()
            group_df[f"rolling_avg_{col}_{window_size}"].fillna(0, inplace=True)
        else:
            group_df[f"rolling_median_{col}_{window_size}"] = group_df[col].rolling(window=window_size,
                                                                                    min_periods=2,
                                                                                    closed='left').median()
            group_df[f"rolling_median_{col}_{window_size}"].fillna(0, inplace=True)
        dfs.append(group_df)
    # Concat the dataframes
    logging.info("Concatenating the dataframes...")
    df_rolling = pd.concat(dfs)
    logging.info(f"SUCCESS: Rolling window calculated.")
    return df_rolling


################## object column transforms ##################

def rainbow_encode(df: pd.DataFrame,
                   col: str,
                   target: str="sales_sum",
                   agg: str="sum",
                   persist: bool=False) -> pd.Series:
    """
    Function performs rainbow encoding on a given column based on the aggregations
    function performed. Function can take any of the following params for agg - sum
    mean, max, std
    """
    # get the aggregation and rank
    logging.info(f"performing the rainbow enconding on {col}:\n"
                 f"target variable:\t{target}\n"
                 f"Aggregate function:\t{agg}")
    # check if file exists already
    if os.path.isfile(f"{col}_encode.json"):
        logging.info(f"json was persisted, loading from disk for {col}")
        # load in the encodings
        encode_dict = json.load(open(f"{col}_encode.json"))
        results = df[col].replace(encode_dict)
        logging.info("Encoding completed")
        return results
    else:
        df_agg = df[[col, target]].groupby(col).aggregate({target: agg})
        df_agg.reset_index(inplace=True)
        logging.info(df_agg)
        df_agg["rank"] = df_agg[target].rank(method="min")
        logging.info(df_agg)
        # create the dictionary
        encode_dict = {}
        for vals in df_agg[[col, "rank"]].values:
            encode_dict[vals[0]] = vals[1]
        # encode the values
        # logging.info(encode_dict)
        results = df[col].replace(encode_dict)
        if persist:
            with open(f"{col}_encode.json", "w") as outfile:
                json.dump(encode_dict, outfile)
            logging.info(f"Successfully persisted {col}_encode.json")
        return results

def pct_change(df:pd.DataFrame,
               col: str,
               sort_val: str,
               by: any) -> pd.DataFrame:
    """
    Function gets the pct change based on the groupby and sort
    """
    dfs = []
    logging.info(f"performing pct change transformaiton on {col}:\n"
                 f"grouped by:\t{by}\n"
                 f"sorted by:\t{sort_val}")
    for group, group_df in tqdm(df.groupby(by=by)):
        # sort the group_df
        group_df.sort_values(by=sort_val, inplace=True)
        # get the percentage change
        group_df[f"{col}_pct_change"] = group_df[col].pct_change(fill_method='ffill')
        dfs.append(group_df)
    # append all the dataframes
    return pd.concat(dfs)


if __name__ == "__main__":
    init_logger()
    # get the files that were persisted
    CA = "data/merged_salescal_price_validation_CA_2023-05-20 16:51:26.125909"
    # CA_val = "data/merged_salescal_price_validation_CA_2023-05-09 12:16:56.815556"
    TX = "data/merged_salescal_price_TX_2023-05-08 14:56:43.755301"
    WI = "data/merged_salescal_price_WI_2023-05-08 14:57:01.107656"

    dfs = []
    logging.info(f"Loading in data...")
    df_data = pd.read_csv(CA)
    logging.info(f"Data loaded successfully: {df_data.shape}")
    # Drop the unnamed class if it exists
    df_data.drop(columns=["Unnamed: 0", "id"], axis=1, inplace=True, errors="ignore")
    # Convert date_max column to datetime
    df_data["date_max"] = pd.to_datetime(df_data["date_max"])
    # get the numerical columns
    # num_cols = [x for x in df_data.columns if df_data[x].dtypes != "object" and df[x].dtypes != "Datetime[64]"]
    # transform the snap columns
    logging.info(f"Transforming the snap columns")
    df_data["snap_CA_sum"] = np.where(df_data["state_id"] == 'CA', df_data["snap_CA_sum"], 0)
    df_data["snap_TX_sum"] = np.where(df_data["state_id"] == 'TX', df_data["snap_TX_sum"], 0)
    df_data["snap_WI_sum"] = np.where(df_data["state_id"] == 'WI', df_data["snap_WI_sum"], 0)
    logging.info(f"snap columns fully transformed")
    # perform the log transform on the desired numerical columns
    logging.info(f"performing log transformations")
    log_transform(df=df_data, col=["sales_sum", "sales_min", "sales_mean", "sales_max"])
    # perform feature engineering on prices
    logging.info(f"filtering the data to only weeks where each item was sold\ncurrent shape {df_data.shape}")
    df_data["sold"] = np.where(df_data["sell_price"].isna(), 0, 1)
    # filter the data to only when that item was sold
    df_data = df_data[df_data['sold'] == 1]
    logging.info(f"Shape after filtering {df_data.shape}")
    group_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'month']
    logging.info(f"performing pct change transformation on the sell_price column...")
    df_data = pct_change(df=df_data, col="sell_price", sort_val="wm_yr_wk", by=group_cols)
    # fill the remaining null values with 0
    df_data["sell_price_pct_change"].fillna(0.0, inplace=True)
    # perform the lagged values on the numerical data
    logging.info("performing the lag transoformation on the sales data")
    for group, group_df in df_data.groupby(group_cols):
        group_df.sort_values(by="wm_yr_wk", inplace=True)
        lag_transform(df=group_df, col=["sales_min_log", "sales_mean_log", "sales_max_log", "sales_sum_log"])
        dfs.append(group_df)
    df_data = pd.concat(dfs)

    # perform rolling window
    df_data = rolling_window(df=df_data, col="sales_sum_log", by=group_cols, window_size=4, agg="mean")
    df_data = rolling_window(df=df_data, col="sales_sum_log", by=group_cols, window_size=4, agg="median")

    # perform rainbow encoding on the categorical data
    logging.info(f"performing rainbow encoding...")
    for col in tqdm(group_cols):
        if col == 'month':
            continue
        df_data[col] = rainbow_encode(df=df_data, col=col, target="sales_sum_log", agg="mean", persist=True)

    # persist the data
    logging.info(f"persisting training set state_id CA")
    logging.info(f"columns: {df_data.columns}")
    df_data_train = df_data[df_data["date_max"] < "2016-04-24"]
    logging.info(f"training set shape:\t{df_data_train.shape}")
    df_data_train.to_csv(f"data/feature_engineer_state_CA_training_{datetime.today().isoformat()}.csv")
    df_data_test = df_data[df_data["date_max"] >= "2016-04-24"]
    logging.info(f"Testing set shape:\t{df_data_test.shape}")
    df_data_test.to_csv(f"data/feature_engineer_state_CA_testing_{datetime.today().isoformat()}.csv")



