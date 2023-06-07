import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import os
import sys
import logging
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

X_FEATURES = [
    'item_id', 'dept_id', 'cat_id', 'store_id',
    'snap_CA_sum', 'event_sum', 'sell_price_pct_change',
    'sales_sum_log_1_lag', 'month', 'rolling_avg_sales_sum_log_4', 'rolling_median_sales_sum_log_4',
    'sales_min_log_1_lag', 'sales_mean_log_1_lag', 'sales_max_log_1_lag'
]

TARGET_VAR = 'sales_sum_log'

SALES_ACTUALS = 'sales_sum'

PARAMS = {
    'n_estimators': [10, 50, 100, 150],
    'max_depth': [3, 10, 25, 60]
}


def init_logger() -> None:
    """Initialize logging, creating necessary folder if it doesn't already exist"""
    # Assume script is called from top-level directory
    log_dir = Path("logs")
    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    # Configue handlers to print logs to file and std out
    file_handler = logging.FileHandler(filename="logs/rf_training.log")
    stdout_handler = logging.StreamHandler(sys.stdout)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[file_handler, stdout_handler],
    )


def train_model(x: pd.DataFrame,
                y: any,
                actuals: any,
                params: dict,
                persist: bool=False) -> pd.DataFrame:
    """
    Train the random forest model using gridsearch
    """
    logging.info(f"trainging RF model\n\Params set to\t{params}")
    regr = RandomForestRegressor(random_state=0, n_jobs=4, max_leaf_nodes=20, max_samples=0.7)
    clf = GridSearchCV(regr, params)
    clf.fit(x, y)
    logging.info(f"Model done training {datetime.today().isoformat()}")
    if persist:
        logging.info("persisting the model")
        if not os.path.exists("models"):
            os.makedirs("models")
        model_file = f'models/model{datetime.today().isoformat()}.pkl'
        joblib.dump(clf.best_estimator_, model_file)
    else:
        model_file = None
    # get the training results
    y_pred_train = clf.predict(x)
    # transform predicted back to original state
    y_pred_train = np.exp(y_pred_train) - 0.01
    # get the metric for training set
    mse = mean_squared_error(y_true=actuals, y_pred=y_pred_train)
    logging.info(f"Training MSE value:\t{mse}")
    rmse = mean_squared_error(y_true=y, y_pred=actuals, squared=False)
    logging.info(f"Training RMSE value:\t{rmse}")
    # grabbing important features
    important_ft = clf.best_estimator_.feature_importances_
    important_ft_name = clf.best_estimator_.feature_names_in_
    # creating feature importance dataframe
    df_important = pd.DataFrame()
    df_important["feature"] = important_ft_name
    df_important["impurity score"] = important_ft
    # save the scores
    df_important.sort_values(by="impurity score", ascending=False, inplace=True)
    logging.info("Saving feature importance...")
    df_important.to_csv(f"data/feature_importance{datetime.today().isoformat()}")
    errors = actuals - y_pred_train
    results = pd.DataFrame(zip(actuals, y_pred_train, errors), columns=['actuals_training', 'predictions_training', 'error_training'])
    return results, model_file


def test_model(model_params: str,
               x: any,
               y: any):
    """
    Load the model results to make prediction
    """
    # load the model
    # Reload from file
    # with open(model_params, 'rb') as f:
    #     loaded_model = pickle.load(f)
    loaded_model = joblib.load(model_params)
    # make prediction on test inputs
    predictions = loaded_model.predict(x)
    predictions = np.exp(predictions) - 0.01
    # get the test scores
    mse = mean_squared_error(y_true=y, y_pred=predictions)
    logging.info(f"Testing MSE value:\t{mse}")
    rmse = mean_squared_error(y_true=y, y_pred=predictions, squared=False)
    logging.info(f"Testing RMSE value:\t{rmse}")
    errors = y - predictions
    results = pd.DataFrame(zip(y, predictions, errors),
                           columns=['actuals', 'predictions', 'error'])
    return results

def plot_resid(df: pd.DataFrame,
               pred: str,
               actuals: str,
               errors: str) -> None:
    """
    save the plot of the residuals
    """
    palette = "ch:s=-.2,r=.6"
    sns.histplot(df, x=errors, log_scale=(False, True), palette=palette).set_title(f"Errors Distribution")
    plt.savefig(f"plots/errors_hist{datetime.today().isoformat()}.jpg",
                bbox_inches="tight")
    # clear the figure
    plt.close()
    # create the scatter plot
    sns.scatterplot(df, x=pred, y=actuals, palette=palette).set_title(f"Predicted vs Actuals")
    plt.savefig(f"plots/pred_act_scatter{datetime.today().isoformat()}.jpg",
                bbox_inches="tight")
    plt.close()
    logging.info(f"Plots of training data saved.")


if __name__ == "__main__":

    init_logger()
    df_train = pd.read_csv("data/feature_engineer_state_CA_training_2023-05-21T12:01:41.013963.csv")
    df_test = pd.read_csv("data/feature_engineer_state_CA_testing_2023-05-21T12:02:38.378170.csv")

    logging.info(f"dropping NA values...\n{df_train.isna().sum()}")
    df_train.dropna(inplace=True)

    logging.info(f"dropping NA values...\n{df_test.isna().sum()}")
    df_test.dropna(inplace=True)

    logging.info("NA values have been dropped")
    y_train = df_train[TARGET_VAR]

    logging.info(f"Target variable set for training:\t{y_train.head()}")

    y_actuals_training = df_train[SALES_ACTUALS]
    y_actual_testing = df_test[SALES_ACTUALS]

    test_dates = df_test["date_max"]

    logging.info(f"Actuals variable set for training:\t{y_actuals_training.head()}")
    logging.info(f"Actuals variable set for testing:\t{y_actual_testing.head()}")

    df_train = df_train[X_FEATURES]
    df_test = df_test[X_FEATURES]

    logging.info(f"Training features set:\t{df_train.head()}")
    logging.info(f"Training features set:\t{df_test.head()}")
    results_df, model_file = train_model(x=df_train,
                                         y=y_train,
                                         actuals=y_actuals_training,
                                         params=PARAMS,
                                         persist=True)
    logging.info(f"Getting test results")
    # model_file = "models/model2023-05-09T11:57:49.440375.pkl"
    test_results = test_model(model_params=model_file, x=df_test, y=y_actual_testing)
    loaded_model_tr = joblib.load(model_file)
    predicted = loaded_model_tr.predict(df_train)
    predicted = np.exp(predicted) - 0.01
    df_tr_results = pd.DataFrame()
    df_tr_results["predicted"] = predicted
    df_tr_results["actuals"] = y_actuals_training
    df_tr_results["errors"] = df_tr_results["actuals"] - df_tr_results["predicted"]
    mse = mean_squared_error(y_true=y_actuals_training, y_pred=predicted)
    logging.info(f"Training MSE value:\t{mse}")
    rmse = mean_squared_error(y_true=y_actuals_training, y_pred=predicted, squared=False)
    logging.info(f"Training RMSE value:\t{rmse}")
    # plot the results
    plot_resid(df=df_tr_results, pred='predicted', actuals='actuals', errors="errors")
    plot_resid(df=test_results, pred='predictions', actuals='actuals', errors="error")
    # save the test results
    test_cols = df_test.columns
    logging.info("persisting the test results")
    test_results[test_cols] = df_test[test_cols]
    test_results["date"] = test_dates
    test_results.to_csv("data/test_results2.csv")


