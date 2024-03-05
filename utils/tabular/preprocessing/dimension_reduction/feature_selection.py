from typing import List, Dict
import pandas as pd
from pydantic import Field
from sklearn.feature_selection import VarianceThreshold
from tabular.dataset import TabularDataType
from utils.tabular.preprocessing.preprocessing_base import PrepBase
from utils.tabular.preprocessing.validation_base import ArgsBase


class DADropCols:
    _extend = False
    _allowed_data_types = "all"
    _missing_values_allowed = True
    _dependent_variable_required = False

    class Args(ArgsBase):
        drop: List[str] = Field(..., min_items=1)

    def __init__(self, **kwargs):
        self.args = self.Args(**kwargs).dict()
        self.drop_cols = None

    def fit(self, df: pd.DataFrame):
        self.drop_cols = [col for col in df.columns if col in self.args["drop"]]
        assert len(self.drop_cols) == len(
            self.args["drop"]
        ), f"columns {list(set(self.args['drop']-set(df.columns)))} not found in DataFrame"

    def transform(self, df: pd.DataFrame):
        df = df.drop(columns=self.drop_cols)
        return df


class DAKeepCols:
    _extend = False
    _allowed_data_types = "all"
    _missing_values_allowed = True
    _dependent_variable_required = False

    class Args(ArgsBase):
        keep: List[str] = Field(..., min_items=1)

    def __init__(self, **kwargs):
        self.args = self.Args(**kwargs).dict()
        self.keep_cols = None

    def fit(self, df: pd.DataFrame):
        self.keep_cols = [col for col in df.columns if col in self.args["keep"]]
        assert len(self.keep_cols) == len(
            self.args["keep"]
        ), f"columns: {list(set(self.args['keep']-set(df.columns)))} not found in DataFrame"

    def transform(self, df: pd.DataFrame):
        return df[self.keep_cols]


class DAMissingValuesThreshold:
    _extend = False
    _allowed_data_types = "all"
    _missing_values_allowed = True
    _dependent_variable_required = False

    class Args(ArgsBase):
        threshold_ratio: float | None = None
        threshold_freq: int | None = None

    def __init__(self, **kwargs):
        self.args = self.Args(**kwargs).dict()
        self.dropped_cols = None

    def fit(self, df: pd.DataFrame):
        drop_cols = []
        if self.args.get("threshold_ratio"):
            drop_cols += list(
                df.columns[df.isnull().mean() > self.args["threshold_ratio"]]
            )
        if self.args.get("threshold_freq"):
            drop_cols += list(
                df.columns[df.isnull().sum() > self.args["threshold_freq"]]
            )
        self.dropped_cols = drop_cols

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=self.dropped_cols)


class DAUniqueValuesThreshold:
    _extend = False
    _allowed_data_types = "all"
    _missing_values_allowed = True
    _dependent_variable_required = False

    class Args(ArgsBase):
        threshold_ratio: float | None = None
        threshold_freq: int | None = None

    def __init__(self, **kwargs):
        self.args = self.Args(**kwargs).dict()
        self.dropped_cols = None

    def fit(self, df: pd.DataFrame):
        drop_cols = []
        if self.args.get("threshold_ratio"):
            drop_cols += list(
                df.columns[df.nunique() / len(df) > self.args["threshold_ratio"]]
            )
        if self.args.get("threshold_freq"):
            drop_cols += list(df.columns[df.nunique() > self.args["threshold_freq"]])
        self.dropped_cols = drop_cols

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=self.dropped_cols)


class DAVarianceThreshold(VarianceThreshold):
    _extend = True
    _allowed_data_types = [TabularDataType.NUMBER.value]
    _missing_values_allowed = True
    _dependent_variable_required = False

    class Args(ArgsBase):
        threshold: float = 0

    def __init__(self, **kwargs):
        self.args = self.Args(**kwargs).dict()
        super().__init__(**self.args)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        kept_cols = list(self.get_feature_names_out())
        return df[kept_cols]


class PrepDimensionReductionFeatureSelection(PrepBase):
    PREPROCESSOR_MAPPING = {
        "drop_cols": DADropCols,
        "keep_cols": DAKeepCols,
        "unique_threshold": DAUniqueValuesThreshold,
        "missing_threshold": DAMissingValuesThreshold,
        "variance_threshold": DAVarianceThreshold,
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

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.preprocessor.transform(df[self.targeted_cols])
        return df
