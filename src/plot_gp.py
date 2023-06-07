import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == "__main__":

    df_evela = pd.read_csv("data/merged_salescal_price_validation_CA_2023-05-20 16:51:26.125909")
    df_evela = df_evela[["cat_id", "date_max", "sales_sum"]].groupby(["cat_id", "date_max"]).aggregate({"sales_sum": "sum"}).reset_index()
    df_evela.columns = ["category", "date", "sales"]
    df_evela["date"] = pd.to_datetime(df_evela["date"])
    df_evela.sort_values(by="date", inplace=True)
    palette = "ch:s=-.2,r=.6"
    # generate the distribtioin and line plots
    # fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df_evela,
                 x="sales",
                 hue="category",
                 binwidth=0.2,
                 multiple="dodge",
                 palette=palette,
                 log_scale=(False, True)).set_title(f"Distribution of Weekly Sales by Category")

    # sns.lineplot(df_evela,
    #              x="date",
    #              y="sales",
    #              hue="category",
    #              palette=palette).set_title(f"Sales by Category")
    # save the figures once the plot has been generated
    plt.savefig(f"plots/sales_dist.jpg",
                bbox_inches="tight")