import pandas as pd

from tabular.dataset import TabularDataset, TabularDataType
from utils.tabular.preprocessing.preprocessing_base import PrepBase
from utils.tabular.preprocessing.validation_base import ArgsBase


class CustomizedTemplate:
    _extend = False
    _allowed_data_types = [TabularDataType.NUMBER.value]
    _missing_values_allowed = False
    _dependent_variable_required = False

    class Args(ArgsBase):
        pass

    def __init__(self, **kwargs):
        self.args = self.Args(**kwargs).dict()

    def fit(self, df: pd.DataFrame):
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class PrepCustomized(PrepBase):
    PREPROCESSING_MAPPING = dict()

    def __init__(self, targeted_cols, strategy, args):
        self.targeted_cols = targeted_cols
        self.strategy = strategy
        self.args = args
        self.preprocessor = None
        self.dtypes = None

    def fit(self, df):
        self.preprocessor = self.PREPROCESSING_MAPPING[self.strategy](**self.args)
        self._validate_preprocessing(
            df=df,
            preprocessor=self.preprocessor,
            targeted_cols=self.targeted_cols,
            args=self.args,
        )
        self.preprocessor.fit(df[self.targeted_cols])
        return self

    def transform(self, df):
        return df

    def fit_transform(self, df):
        return self.fit(df).transform(df)
