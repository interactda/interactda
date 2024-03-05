from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    Normalizer,
    MaxAbsScaler,
)
from tabular.dataset import TabularDataType
from utils.tabular.preprocessing.preprocessing_base import PrepBase
from typing import List, Dict
import pandas as pd
from utils.tabular.preprocessing.validation_base import ArgsBase


class DAMinMaxScaler(MinMaxScaler):
    _extend = True
    _allowed_data_types = [TabularDataType.NUMBER.value]
    _missing_values_allowed = True
    _dependent_variable_required = False

    class Args(ArgsBase):
        pass


class DAStandardScaler(StandardScaler):
    _extend = True
    _allowed_data_types = [TabularDataType.NUMBER.value]
    _missing_values_allowed = True
    _dependent_variable_required = False

    class Args(ArgsBase):
        pass


class DARobustScaler(RobustScaler):
    _extend = True
    _allowed_data_types = [TabularDataType.NUMBER.value]
    _missing_values_allowed = True
    _dependent_variable_required = False

    class Args(ArgsBase):
        pass


class DANormalizer(Normalizer):
    _extend = True
    _allowed_data_types = [TabularDataType.NUMBER.value]
    _missing_values_allowed = False
    _dependent_variable_required = False

    class Args(ArgsBase):
        pass


class DAMaxAbsScaler(MaxAbsScaler):
    _extend = True
    _allowed_data_types = [TabularDataType.NUMBER.value]
    _missing_values_allowed = True
    _dependent_variable_required = False

    class Args(ArgsBase):
        pass


class PrepFeatureEngineeringScaling(PrepBase):
    PREPROCESSOR_MAPPING = {
        "min_max_scaler": DAMinMaxScaler,
        "standard_scaler": DAStandardScaler,
        "robust_scaler": DARobustScaler,
        "max_abs_scaler": DAMaxAbsScaler,
        "normalizer": DANormalizer,
    }

    def __init__(self, targeted_cols: List[str], strategy: str, args: Dict):
        self.targeted_cols = targeted_cols
        self.strategy = strategy
        self.args = args
        self.preprocessor = None
        self.dtypes = None

    def fit(self, df: pd.DataFrame):
        self.targeted_cols = self.targeted_cols
        self.preprocessor = self.PREPROCESSOR_MAPPING[self.strategy](**self.args)
        self.preprocessor.fit(df[self.targeted_cols])

    def transform(self, df: pd.DataFrame):
        df[self.targeted_cols] = self.preprocessor.transform(df[self.targeted_cols])
        return df
