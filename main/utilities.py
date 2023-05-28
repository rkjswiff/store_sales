import os
import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CreatePath:

    def __init__(self, mainpath):
        self.mainpath = mainpath

        # create dir if it does not exists
        if not os.path.exists(self.mainpath):
            os.makedirs(self.mainpath)

    def filepath(self, filename):

        return os.path.join(self.mainpath, filename)


class OneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns, save_dir = "./"):
        self.columns = columns
        self.save_dir = save_dir

    def fit(self, data):
        before_cols = list(data.columns)
        data = pd.get_dummies(data, columns=self.columns)
        after_cols = set(data.columns)
        encoded_columns = set(after_cols).difference(set(before_cols))
        encoded_columns = list(encoded_columns)

        # save columns state
        joblib.dump(encoded_columns, os.path.join(self.save_dir, "onehotEncodeedcols.tx"))

        return self

    def transform(self, data):
        before_cols = list(data.columns)
        data = pd.get_dummies(data, columns=self.columns)
        after_cols = set(data.columns)
        encoded_columns = set(after_cols).difference(set(before_cols))
        encoded_columns = list(encoded_columns)
        mandatory_columns = joblib.load(os.path.join(self.save_dir, "onehotEncodeedcols.tx"))
        missing_columns = set(mandatory_columns).difference(set(encoded_columns))
        extra_columns = set(encoded_columns).difference(set(mandatory_columns))
        if missing_columns:
            data.loc[:, missing_columns] = 0
        if extra_columns:
            data = data.drop(extra_columns, axis=1)

        data.columns = data.columns.str.replace(' ', '_')

        return data


def wmape(y_true, y_pred):

    wmape = (y_true.sum() - y_pred.sum())/y_pred.sum()
    wmape = abs(wmape)

    return wmape