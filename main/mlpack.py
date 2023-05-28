import joblib
import pandas as pd
import config
import utilities
import argparse
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


def parse_argument():
    parser = argparse.ArgumentParser(prog="StoreSales")
    parser.add_argument('--mode', required=True, help="train/predict")
    args = parser.parse_args()

    return args


class StoreSalesModel:
    def __init__(self, mode):
        self.mode = mode
        self.result_dir = config.paths["result_dir"]
        self.output_data_path = utilities.CreatePath(self.result_dir)
        self.stage_dir = config.paths["stage_dir"]
        self.stage_data_path = utilities.CreatePath(self.stage_dir)

    def preprocess(self, data, val_pred = False):
        """

        :param data: data to preprocess
        :param val_pred: is it a validation or testing data, default= False
        :return:
        """

        # fill na
        print("filling NAs...")
        data = data.fillna(0)

        # handle categorical data
        onehotencoder = utilities.OneHotEncoder(columns=["Type"], save_dir=self.stage_dir)

        if not val_pred:
            print("Categorical Encoding....")
            data = onehotencoder.fit_transform(data)

            print("Scaling data....")
            scaler = StandardScaler()
            scaled_df = scaler.fit_transform(data)
            scaled_df = pd.DataFrame(scaled_df, columns=data.columns)
            joblib.dump(scaler, os.path.join(self.stage_dir, "scaler"))

        else:
            print("Categorical Encoding....")
            data = onehotencoder.transform(data)

            print("Scaling data....")
            scaler = joblib.load(os.path.join(self.stage_dir, "scaler"))
            scaled_df = scaler.tranform(data)

        return scaled_df

    def train(self):
        print("Loading data....")
        train_df = pd.read_csv(self.stage_data_path.filepath(config.paths["train"]))
        validation_df = pd.read_csv(self.stage_data_path.filepath(config.paths["validation"]))

        # X Y split
        print("Splitting X and Y....")
        train_x, train_y = train_df.drop("Weekly_Sales", axis=1), train_df["Weekly_Sales"]
        validation_x, validation_y = validation_df.drop("Weekly_Sales", axis=1), validation_df["Weekly_Sales"]

        # preprocess X
        print("Preprocessing files....")
        train_x = self.preprocess(train_x)
        validation_x = self.preprocess(validation_x)

        # model training and saving
        print("Model training in progress....")
        model = RandomForestRegressor(n_estimators=100, n_jobs=-1, max_depth=10, verbose=1)
        model.fit(train_x, train_y)
        joblib.dump(model, self.output_data_path.filepath("model.pkl"))

        # calculate feature importance
        print("Calculating feature importance....")
        importances = model.feature_importances_
        feature_importance_df = pd.Series(importances, index=train_x.columns)
        feature_importance_df.to_csv(self.output_data_path.filepath("feature_importance.csv"))

        # predictions
        print("Calculating scores...")
        validation_hat = model.predict(validation_x)
        train_hat = model.predict(train_x)

        # evaluate results
        # training score
        print("")
        print("Training score : ")
        print("==================")
        print("R2 Score       : {}".format(r2_score(y_true=train_y, y_pred=train_hat)))
        print("WMAPE          : {}".format(utilities.wmape(y_true=train_y, y_pred=train_hat)))

        # validation score
        print("")
        print("Validation score : ")
        print("==================")
        print("R2 Score       : {}".format(r2_score(y_true=validation_y, y_pred=validation_hat)))
        print("WMAPE          : {}".format(utilities.wmape(y_true=validation_y, y_pred=validation_hat)))
        print("")

    def predict(self):
        print("Loading data....")
        prediction_df = pd.read_csv(self.stage_data_path.filepath(config.paths["test"]))

        # X Y split
        print("spliting X and Y...")
        predict_x, predict_y = prediction_df.drop("Weekly_Sales", axis=1), prediction_df["Weekly_Sales"]

        # preprocess X
        print("Preprocessing.....")
        predict_x = self.preprocess(predict_x, val_pred=True)

        # predictions
        print("Predicting........")
        model = joblib.load(self.output_data_path.filepath("model.pkl"))
        predict_hat = model.predict(predict_x)
        predict_hat.to_csv(self.output_data_path.filepath("predictions.csv"))


if __name__ == "__main__":

    args = parse_argument()
    store_sales = StoreSalesModel(mode=args.mode)

    if args.mode == "train":
        store_sales.train()
    elif args.mode == "predict":
        store_sales.predict()
    else:
        ValueError("Wrong value for Mode")




