import pandas as pd
import seaborn as sns
import logging
import sys
import os
import numpy as np
import statsmodels.api as sm
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
from statsmodels.tsa.stattools import adfuller

"""
Run eda on the data and produce statistical output and plots
"""

CAL_FILE = "data/merged_salescal_price_CA_2023-05-08 14:56:20.991359"
TX_FILE = "data/transform_salescal_TX_2023-05-03 17:11:02.609918"
WIS_FILE = "data/transform_salescal_WI_2023-05-03 17:11:19.599778"

def init_logger() -> None:
    """Initialize logging, creating necessary folder if it doesn't already exist"""
    # Assume script is called from top-level directory
    log_dir = Path("logs")
    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    # Configue handlers to print logs to file and std out
    file_handler = logging.FileHandler(filename="logs/data_analysis.log")
    stdout_handler = logging.StreamHandler(sys.stdout)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[file_handler, stdout_handler],
    )


def create_plots_dir() -> None:
    """
    Create plots directory
    """
    if not os.path.exists("plots"):
        os.makedirs("plots")


# get general statistical stats on columns

def get_stats_numerical(df:pd.DataFrame,
                        state: str="CA") -> None:
    """
    get statistical stats for any numerical columns in the dataframe
    """
    logging.info(f"Getting statistical informaiton for {state}")
    # get the numerical columns
    num_cols = [x for x in df.columns if df[x].dtypes != "object" and df[x].dtypes != "Datetime[64]"]
    logging.info(f"numerical columns:\t{num_cols}")
    # get the mean for the numerical columns
    for col in num_cols:
        mean_val = df[col].mean()
        median_val = df[col].median()
        max_val = df[col].max()
        min_val = df[col].min()
        try:
            std_val = df[col].values.std(ddof=1)
        except Exception as e:
            logging.warning(f"Unable to get the std for col {col}\n{e}")
        stats_msg = (
            f"\nStats for {col} - {state}:\n"
            "#############################\n"
            f"Mean Value:\t{mean_val}\nMin Value:\t{min_val}\nMax Value:\t{max_val}\nStandard Deviation:\t{std_val}\n"
            f'Median Value:\t{median_val}'
        )
        logging.info(stats_msg)


def get_stats_category(df:pd.DataFrame,
                       state: str="CA") -> None:
    """
    Get the statistical stats for the categorical columns
    """
    logging.info(f"Getting statistical information for {state}")
    # get the object columns
    obj_cols = [x for x in df.columns if df[x].dtypes == "object"]
    logging.info(f"Categorical columns:\t{obj_cols}")
    for col in obj_cols:
        logging.info(f"Number of unique elements for {col} - {state}:\t{len(df[col].unique().tolist())}")
        # get weekly sales
        df_sales = df[[col, "sales_sum"]].groupby(col).aggregate({"sales_sum": "mean"}).reset_index()
        df_sales.columns = [col, "weekly_sales_mean"]
        df_sales.sort_values(by="weekly_sales_mean", ascending=False, inplace=True)
        logging.info(f"top 5 by mean {df_sales.head()}")
        logging.info(f"{df[col].value_counts()}")


def get_numerical_plots(df:pd.DataFrame,
                        by: any=None,
                        state: str="CA") -> None:
    """
    generate plots for the numerical data
    df: pd.DataFrame
        Dataframe
    by: list or string
        group by values aggregate would be sum of sales
    """
    logging.info(f"Getting statistical informaiton for {state}")
    # get the numerical columns
    num_cols = [x for x in df.columns if df[x].dtypes != "object" and df[x].dtypes != "Datetime[64]"]
    palette = "ch:s=-.2,r=.6"
    for col in num_cols:
        if by is None:
            logging.info("No Group by setting as the original df")
            # sort the values
            df.sort_values(by=["date_max"], inplace=True)
            # generate the distribtioin and line plots
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            sns.histplot(df,
                         x=col,
                         hue=by,
                         binwidth=0.4,
                         multiple="dodge",
                         palette=palette,
                         log_scale=(False, True),
                         ax=axes[0]).set_title(f"Distribution of {col} - {state}")

            sns.lineplot(df,
                         x="date_max",
                         y=col,
                         hue=by,
                         palette=palette,
                         ax=axes[1]).set_title(f"{col} - {state}")
            plt.ylabel("Date")
            # save the figures once the plot has been generated
            plt.savefig(f"plots/{col}_{state}_plot{datetime.today().isoformat()}.jpg",
                        bbox_inches="tight")
        else:
            for group, group_df in df.groupby(by):
                logging.info(f"grouping by {group}")
                group_df.reset_index(inplace=True)
                # sort the values
                group_df.sort_values(by=["date_max"], inplace=True)
                # generate the distribtioin and line plots
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                sns.histplot(group_df,
                             x=col,
                             hue=by,
                             binwidth=0.4,
                             multiple="dodge",
                             palette=palette,
                             log_scale=(False, True),
                             ax=axes[0]).set_title(f"Distribution of weekly Sales by {group} - {state}")

                sns.lineplot(group_df,
                             x="date_max",
                             y=col,
                             hue=by,
                             palette=palette,
                             ax=axes[1]).set_title(f"Weekly Sales by {group} - {state}")
                plt.ylabel("Date")
                # save the figures once the plot has been generated
                plt.savefig(f"plots/{col}_{state}_{group}_plot{datetime.today().isoformat()}.jpg",
                            bbox_inches="tight")
                plt.close()

def get_correlations(df:pd.DataFrame,
                     state: str="CA",
                     by: any=None) -> None:
    """
    Function creates an image of a correlation plot between numerical columns
    """
    # get numerical columns
    num_cols = [x for x in df.columns if df[x].dtypes != "object" and df[x].dtypes != "Datetime[64]"]
    # plot the correlation plot of the numerical cols
    if by is None:
        logging.info(f"producing corr plot for sales")
        corr = df[num_cols].corr().round(3)
        sns.heatmap(corr, annot=True)
        plt.savefig(f"plots/Corr_{state}_plot_train{datetime.today().isoformat()}.png",
                    bbox_inches="tight")
    else:
        for group, group_df in df.groupby(by):
            corr = group_df[num_cols].corr().round(3)
            sns.heatmap(corr, annot=True)
            logging.info(f"producing corr plot for sales - {group}")
            plt.savefig(f"plots/Corr_{state}_plot_train{group}_{datetime.today().isoformat()}.png",
                        bbox_inches="tight")
            plt.close()


def run_kruskal(df:pd.DataFrame,
                state: str="CA") -> None:
    """
    Run the kruskal wallace test for all categorical variables in the
    data set:

    The kruskal wallace test is used when you have a categorical independent variable
    and a continuous depenedent variable. This is a nonparametric test.
    """
    # get the object columns
    obj_cols = [x for x in df.columns if df[x].dtypes == "object"]
    # run the kruskal wallace test
    for col in obj_cols:
        if col == "item_id" or col == "dept_id" or col == "store_id":
            logging.info(f"{col} has too many values to compute, skipping ...")
            continue
        cat_dict = {}
        for val in df[col].unique().tolist():
            # create the values for the test
            cat_dict[val] = df[df[col] == val]["sales_sum"].values

        logging.info(list(cat_dict.values()))
        # create the list value aruguments to pass to the test
        values = list(cat_dict.values())
        res = getattr(stats, 'kruskal')(*values)
        logging.info(f"Kruskal wallace test for {col} - {state}:\t{res}")


####################################### Time series plots and analysis code #######################################


def get_autocorr(df: pd.DataFrame,
                 by: any=None,
                 state: str="CA") -> None:
    """
    Plot the autocorrealtion plots for sales sum data
    """
    # generate auot_cor plots
    if by is None:
        logging.info("No Group by setting as the original df")
        sm.graphics.tsa.plot_acf(df.sort_values(by="date_max")["sales_sum"])
        plt.title(f"Autocorrelation of Weekly Sales - {state}")
        plt.savefig(f"plots/sales_{state}_autocorr{datetime.today().isoformat()}.png",
                    bbox_inches="tight")
    else:
        for group, group_by_df in df.groupby(by):
            logging.info(f"Generating auoto cor plot for {group}")
            group_by_df = group_by_df.groupby(by="date_max").aggregate({"sales_sum": "sum"})
            group_by_df.reset_index(inplace=True)
            group_by_df.columns = ["date_max", "sales_sum"]
            sm.graphics.tsa.plot_acf(group_by_df.sort_values(by="date_max")["sales_sum"])
            plt.title(f"Autocorrelation of {group} - {state}")
            plt.savefig(f"plots/{group}_{state}{datetime.today().isoformat()}.png",
                        bbox_inches="tight")
            plt.close()


def get_decompose(df:pd.DataFrame,
                  state: str="CA",
                  by: any=None,
                  periods: int=12,
                  model: str="multiplicable") -> None:
    """
    Get the decomposition of sales sum
    """
    if by is None:
        logging.info("No Group by setting as the original df")
        result = seasonal_decompose(df['sales_sum'], model=model, period=periods)
        result.set_title(f'Sales - {state}')
        plt.savefig(f"plots/SALES_{state}_autocorr{datetime.today().isoformat()}.png",
                    bbox_inches="tight")
    else:
        for group, group_df in df.groupby(by):
            # iterate over the groups
            logging.info(f"Group by:\t{group}")
            group_df = group_df.groupby(by="date_max").aggregate({"sales_sum": "sum"})
            group_df.reset_index(inplace=True)
            group_df.columns = ["date_max", "sales_sum"]
            result = seasonal_decompose(group_df['sales_sum'], model=model, period=periods).plot()
            plt.title(f"Sales by {group} - {state}")
            plt.savefig(f"plots/SALES_autocorr{group}_{state}{datetime.today().isoformat()}.png",
                        bbox_inches="tight")
            plt.close()



def test_stationarity(df:pd.DataFrame,
                      state: str="CA",
                      by: any=None) -> None:
    """
    Test for stationarity amongst the grouped columnns or just sales
    using the augmented dicky fuller test
    """
    if by is None:
        logging.info("No Group by setting as the original df")
        # run the augmented dicky fuller test
        logging.info(f"Getting the ADF results for sales_sum - {state}")
        X = df["sales_sum"].values
        result = adfuller(X)
        logging.info(f"State:\t{state}")
        logging.info('ADF Statistic: %f' % result[0])
        logging.info('p-value: %f' % result[1])
        logging.info('Critical Values:')
        for key, value in result[4].items():
            logging.info('\t%s: %.3f' % (key, value))
    else:
        for group, group_df in df.groupby(by):
            logging.info(f"Getting the ADF results for sales_sum by {group}")
            group_df = group_df.groupby(by="date_max").aggregate({"sales_sum": "sum"})
            group_df.reset_index(inplace=True)
            group_df.columns = ["date_max", "sales_sum"]
            X = group_df["sales_sum"].values
            result = adfuller(X)
            logging.info(f"State:\t{state}")
            logging.info('ADF Statistic: %f' % result[0])
            logging.info('p-value: %f' % result[1])
            logging.info('Critical Values:')
            for key, value in result[4].items():
                logging.info('\t%s: %.3f' % (key, value))


def save_stationairty(df:pd.DataFrame,
                      by: any=None,
                      persist: bool=False) -> pd.DataFrame:
    """
    function is primarily for the items column of the dataset
    it saves the results of the test statistics from the augmented
    dicky fuller test

    P-value ≤ significance level
    Test statistic ≤ critical value = stationary
    """
    group_list = []
    adf_statistic_list = []
    p_value_list = []
    for group, group_df in df.groupby(by):
        logging.info(f"Getting the ADF results for sales_sum by {group}")
        group_list.append(group[-1])
        group_df = group_df.groupby(by="date_max").aggregate({"sales_sum": "sum"})
        group_df.reset_index(inplace=True)
        group_df.columns = ["date_max", "sales_sum"]
        group_df.sort_values(by="date_max", inplace=True)
        X = group_df["sales_sum"].values
        result = adfuller(X)
        logging.info('ADF Statistic: %f' % result[0])
        adf_statistic_list.append(result[0])
        logging.info('p-value: %f' % result[1])
        p_value_list.append(result[1])
        logging.info('Critical Values:')
        for key, value in result[4].items():
            logging.info('\t%s: %.3f' % (key, value))
    # create the dataframe
    df_results = pd.DataFrame
    df_results["group"] = group_list
    df_results["adf_statistic"] = adf_statistic_list
    df_results["pvalue"] = p_value_list
    # create the stationary indicator
    df_results["stationarity"] = np.where(df_results < 0.05, 1, 0)


if __name__ == "__main__":

    init_logger()
    create_plots_dir()
    logging.info("############################ Data Analysis ############################")
    file_list = [CAL_FILE]
    states = ["CA"]
    # groupby = ["cat_id", "date_max"]
    groupby2 = "cat_id"

    for file_state in zip(file_list, states):
        df_state = pd.read_csv(file_state[0])
        df_state["date_max"] = pd.to_datetime(df_state["date_max"])
        df_state["date_max"] = df_state["date_max"].dt.date
        # drop unnamed column if it exists
        df_state.drop(columns=["Unnamed: 0"], axis=1, inplace=True, errors="ignore")
        df_state["date_max"] = pd.to_datetime(df_state["date_max"], format = '%Y-%m-%d')
        # get the stats for the numerical columns
        # get_stats_numerical(df=df_state, state=file_state[1])
        # # # get the stats for the categorical columns
        # get_stats_category(df=df_state, state=file_state[1])
        # # get the numerical plots
        # get_numerical_plots(df=df_state, by=groupby2, state=file_state[1])
        # # create the correlation plots
        # get_correlations(df=df_state, state=file_state[1], by=groupby2)
        # # run the kruskal wallace test
        run_kruskal(df=df_state, state=file_state[1])
        # # get the autocorrelation plots
        # get_autocorr(df=df_state, by=groupby2, state=file_state[1])
        # # get the decompose of the data
        # get_decompose(df=df_state, state=file_state[1], by=groupby2)
        # # test for stationarity
        # test_stationarity(df=df_state, state=file_state[1], by=groupby2)










