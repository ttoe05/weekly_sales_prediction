"""
Preprocess the data by merging the data and splitting
the data into 3 subparts by state. Sales in one state
should be independent of the other assuming distribution
does not affect the other.
"""

import pandas as pd
import numpy as np
import logging
import sys
from utils import persist_df
from pathlib import Path
from datetime import datetime


FILE_PATH = "data/data_processor"
def init_logger() -> None:
    """Initialize logging, creating necessary folder if it doesn't already exist"""
    # Assume script is called from top-level directory
    log_dir = Path("logs")
    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    # Configue handlers to print logs to file and std out
    file_handler = logging.FileHandler(filename="logs/dataprocessor.log")
    stdout_handler = logging.StreamHandler(sys.stdout)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[file_handler, stdout_handler],
    )


def sales_clean(df_sales:pd.DataFrame) -> pd.DataFrame:
    """
    Clean and pivot the sales data to prep for merging
    """
    # melt the data so the day columns are rows
    ids = [x for x in df_sales.columns if "_id" in x or x == "id"]
    values = [x for x in df_sales.columns if "d_" in x]
    logging.info(f"melting the data by ids:\t{ids}")
    df_sales_melt = pd.melt(frame=df_sales,
                            id_vars=ids,
                            value_vars=values)
    df_sales_melt.rename(columns={"variable": "d", "value": "sales"}, inplace=True)
    logging.info(f"columns in the dataframe:\t{df_sales_melt.columns}")
    return df_sales_melt


def merge_sales_cal(df_sales: pd.DataFrame,
                    df_cal: pd.DataFrame,
                    persist: bool=False) -> pd.DataFrame:
    """
    merging the datasets sales and calendar date
    """
    logging.info(f"Shape of sales:\t{df_sales.shape}")
    logging.info(f"Shape of calendar:\t{df_cal.shape}")
    df_merged = df_sales.merge(df_cal, how="left", on="d")
    logging.info(f"Shape after merge:\t{df_merged.shape}")
    logging.info(f"Columns after merging:\t{df_merged.columns}")
    if persist:
        states = ["CA", "TX", "WI"]
        for state in states:
            df_state = df_merged[df_merged["state_id"] == state]
            file_nm = f"sales_calendar_{state}"
            persist_df(df=df_state, file_path=FILE_PATH, name=file_nm)
        logging.info(f"Successfully persisted the states.")
    return df_merged

def merge_sales_cal_cleaner(df: pd.DataFrame,
                            persist: bool = False) -> None:
    """ clean the merged sales and calendar data """
    # delete the unnecessary columns if they exist
    # logging.info(f"prepping data for {state}")
    # drop_snap_cols = []
    # for st in ["CA", "TX", "WI"]:
    #     if state == st:
    #         continue
    #     drop_snap_cols.append(f"snap_{st}")
    drop_cols = ["Unnamed: 0", "weekday", "wday",
                 "event_name_1", "event_name_2"]
    logging.info(f"dropping the following columns:\t{drop_cols}")
    df.drop(columns=drop_cols,
            axis=1,
            errors="ignore",
            inplace=True)
    # convert the date column to pandas datetime
    df['date'] = pd.to_datetime(df['date'])
    # fill in na values for the event columns
    df["event_type_1"].fillna("None", inplace=True)
    df["event_type_2"].fillna("None", inplace=True)
    # convert the values to 1 if an event occured
    logging.info("transforming the event_type columns")
    df["event_type_1"] = np.where(df["event_type_1"] == "None", 0, 1)
    df["event_type_2"] = np.where(df["event_type_2"] == "None", 0, 1)
    logging.info(f"Sales calender cleaner columns:\t{df.columns}")
    # persist the data
    if persist:
        states = ["CA", "TX", "WI"]
        for state in states:
            df_state = df[df["state_id"] == state]
            file_nm = f"merged_salescal_{state}"
            persist_df(df=df_state, file_path=FILE_PATH, name=file_nm)


def merge_sales_transform(df: pd.DataFrame,
                          persist: bool = False) -> pd.DataFrame:
    """
    Make feature transformation to the dataset. Data will be
    aggregated up a week.
    1. one hot encode the categorical variables for date
    2. group by the id columns
        a. sum up revenue
        b. sum up the calendar encodings by the week
            - add the events together
        c. take the max date
        d add the snap cash for that week
    """
    # create aggregation dictionary
    agg_dict = {
        "event_type_1": "sum",
        "event_type_2": "sum",
        "date": "max",
        "snap_CA": "sum",
        "snap_TX": "sum",
        "snap_WI": "sum",
        "sales": ["mean", "sum", "max", "min"]
    }
    logging.info(f"columns before sales calendar transformation:\t{df.columns}")
    df_grouped = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'wm_yr_wk', 'month']).aggregate(agg_dict)
    df_grouped.columns = ['_'.join(col) for col in df_grouped.columns]
    df_grouped.reset_index(inplace=True)
    # add the event types together
    df_grouped["event_sum"] = df_grouped["event_type_1_sum"] + df_grouped["event_type_2_sum"]
    # drop the event columns that are no longer needed
    df_grouped.drop(columns=["event_type_1_sum", "event_type_2_sum"], axis=1, inplace=True)
    # persist by state
    if persist:
        states = ["CA", "TX", "WI"]
        for state in states:
            df_state = df_grouped[df_grouped["state_id"] == state]
            file_nm = f"transform_salescal_{state}"
            persist_df(df=df_state, file_path=FILE_PATH, name=file_nm)

    return df_grouped


def merge_prices(df_x: pd.DataFrame,
                 df_y: pd.DataFrame,
                 on: any,
                 persist: bool = False) -> pd.DataFrame:
    """
    Function merges the prices information with the merged
    sales and calendar. df_x is on the left of the join
    """
    df_merged = df_x.merge(df_y, how='left', on=on)
    if persist:
        states = ["CA", "TX", "WI"]
        for state in states:
            df_state = df_merged[df_merged["state_id"] == state]
            logging.info(f"{state} shape of dataframe after cleaning:\t{df_state.shape}")
            file_nm = f"merged_salescal_price_training_{state}"
            persist_df(df=df_state, file_path=FILE_PATH, name=file_nm)
    return df_merged




if __name__ == "__main__":

    init_logger()
    states = ["CA", "TX", "WI"]
    df_cal = pd.read_csv("data/original/calendar.csv")
    df_sales = pd.read_csv("data/original/sales_train_evaluation.csv")
    # clean the sales data
    df_sales_clean = sales_clean(df_sales=df_sales)
    # merge the sales data with the calendar meta data
    df_merged = merge_sales_cal(df_sales=df_sales_clean, df_cal=df_cal, persist=False)


    # roll up the merged data
    merge_sales_cal_cleaner(df_merged, persist=False)
    df_transform = merge_sales_transform(df_merged, persist=False)
    # get the prices data
    df_prices = pd.read_csv("data/original/sell_prices.csv")
    cols = ['store_id', 'item_id', 'wm_yr_wk']
    df_merged_prices = merge_prices(df_transform, df_prices, on=cols, persist=True)
    # print(df_merged_prices.head(100))
