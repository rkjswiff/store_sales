import pandas as pd
import config
import utilities
import argparse


def parse_argument():
    parser = argparse.ArgumentParser(prog="StoreSales")
    parser.add_argument('--mode', required=True, help="train/predict")
    args = parser.parse_args()

    return args


class StoreSalesEtl:
    def __init__(self, mode):
        self.mode = mode
        self.data_dir = config.paths["data_dir"]
        self.input_data_path = utilities.CreatePath(self.data_dir)
        self.stage_dir = config.paths["stage_dir"]
        self.stage_data_path = utilities.CreatePath(self.stage_dir)

    def read_data(self):
        print("Reading data....")
        features_df = pd.read_csv(self.input_data_path.filepath(config.paths["features"]), parse_dates=["Date"])
        sales_df = pd.read_csv(self.input_data_path.filepath(config.paths["sales_data"]), parse_dates=["Date"])
        store_df = pd.read_csv(self.input_data_path.filepath(config.paths["store_data"]))

        return features_df, sales_df, store_df

    def read_clean_data(self):
        features_df, sales_df, store_df = self.read_data()
        print("Cleaning data....")

        # standardize weekly date
        features_df['Date'] = features_df['Date'] - pd.to_timedelta(features_df['Date'].dt.weekday, unit='d')
        features_df = features_df.groupby(['Store', 'Date'], as_index=False).mean()
        features_df = features_df.drop(columns = 'IsHoliday', axis=1)

        sales_df['Date'] = sales_df['Date'] - pd.to_timedelta(sales_df['Date'].dt.weekday, unit='d')
        sales_df = sales_df.groupby(["Store", "Date", "Dept"], as_index=False).agg({'Weekly_Sales': ['sum'],
                                                                                    'IsHoliday': 'max'})
        sales_df.columns = sales_df.columns.droplevel(1)

        return features_df, sales_df, store_df

    def prepare_data(self):
        """

        :return:
        """
        features_df, sales_df, store_df = self.read_clean_data()

        feature_sales_df = features_df.merge(sales_df, on=["Store", "Date"])
        full_df = feature_sales_df.merge(store_df, on=['Store'])

        # split train test
        full_df["rank"] = full_df.groupby(["Store", "Dept"])["Date"].transform(
            lambda x: x.rank(method='dense', ascending=False))
        full_df["year"] = full_df.Date.dt.year
        full_df["month"] = full_df.Date.dt.month
        full_df["day"] = full_df.Date.dt.isocalendar().week

        full_df = full_df.drop("Date", axis=1)

        print("Saving data....")

        if self.mode == "train":
            # train set
            train_df = full_df[full_df["rank"] > 6]
            train_df = train_df.drop("rank", axis=1)
            train_df.to_csv(self.stage_data_path.filepath(config.paths["train"]))

            # validation set
            validation_df = full_df[full_df["rank"].between(4, 6)]
            validation_df = validation_df.drop('rank', axis=1)
            validation_df.to_csv(self.stage_data_path.filepath(config.paths["validation"]))

        if self.mode == "predict":
            test_df = full_df[full_df["rank"] <= 3]
            test_df = test_df.drop("rank", axis=1)
            test_df.to_csv(self.stage_data_path.filepath(config.paths["test"]))


if __name__ == "__main__":
    args = parse_argument()
    store_sales = StoreSalesEtl(mode=args.mode)
    store_sales.prepare_data()





