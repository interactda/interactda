from sklearn.impute import SimpleImputer
from tabular.dataset import TabularDataType
from utils.tabular.preprocessing.preprocessing_base import PrepBase
import pandas as pd
from pydantic import Field
from utils.tabular.preprocessing.validation_base import ArgsBase
from typing import List, Dict


class DAMeanSimpleImputer(SimpleImputer):
    _extend = True
    _allowed_data_types = [TabularDataType.NUMBER.value]
    _missing_values_allowed = True
    _dependent_variable_required = False

    class Args(ArgsBase):
        pass


class DAMedianSimpleImputer(SimpleImputer):
    _extend = True
    _allowed_data_types = [TabularDataType.NUMBER.value]
    _missing_values_allowed = True
    _dependent_variable_required = False

    class Args(ArgsBase):
        pass


class DAMostFrequentSimpleImputer(SimpleImputer):
    _extend = True
    _allowed_data_types = [TabularDataType.CATEGORY.value, TabularDataType.NUMBER.value]
    _missing_values_allowed = True
    _dependent_variable_required = False

    class Args(ArgsBase):
        pass


class DAConstantSimpleImputer(SimpleImputer):
    _extend = True
    _allowed_data_types = [TabularDataType.CATEGORY.value, TabularDataType.NUMBER.value]
    _missing_values_allowed = True
    _dependent_variable_required = False

    class Args(ArgsBase):
        fill_value: float | int | str | None = Field(None)

        class Config:
            extra = "allow"

    def __init__(self, **kwargs):
        self.args = self.Args(**kwargs).dict()
        self.args["strategy"] = "constant"
        super().__init__(**self.args)


class PrepMissingValuesSimpleImputer(PrepBase):
    PREPROCESSOR_MAPPING = {
        "mean": DAMeanSimpleImputer,
        "median": DAMedianSimpleImputer,
        "most_frequent": DAMostFrequentSimpleImputer,
        "constant": DAConstantSimpleImputer,
    }

    def __init__(self, targeted_cols: List[str], strategy: str, args: Dict):
        self.targeted_cols = targeted_cols
        self.strategy = strategy
        self.args = args
        self.preprocessor = None

    def fit(self, df: pd.DataFrame):
        self.preprocessor = self.PREPROCESSOR_MAPPING[self.strategy](
            missing_values=pd.NA, strategy=self.strategy, **self.args
        )
        self._validate_preprocessing(
            df, self.preprocessor, self.targeted_cols, self.args
        )

        self.preprocessor.fit(df[self.targeted_cols])

    def transform(self, df: pd.DataFrame):
        original_dtypes = df.dtypes
        df[self.targeted_cols] = self.preprocessor.transform(df[self.targeted_cols])
        df = df.astype(original_dtypes)
        return df
