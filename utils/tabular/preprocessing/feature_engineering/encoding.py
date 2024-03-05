from sklearn.preprocessing import LabelEncoder, TargetEncoder, OneHotEncoder
from category_encoders import CountEncoder, GLMMEncoder
from pydantic import BaseModel, field_validator, Field
from tabular.dataset import TabularDataType, TabularDataset
from utils.tabular.preprocessing.preprocessing_base import PrepBase
import pandas as pd
from utils.tabular.preprocessing.validation_base import ArgsBase
from typing import List, Dict, Callable
import numpy as np


class DATargetEncoder:
    _extend = False
    _allowed_data_types = [TabularDataType.CATEGORY.value]
    _missing_values_allowed = False
    _dependent_variable_required = True

    class Args(ArgsBase):
        target: str = Field(...)
        cv: int = Field(5, gt=1, lt=21)
        shuffle: bool = Field(True)
        random_state: int | None = None
        smooth: float | str = "auto"
        target_type: str = Field("auto", pattern="auto|binary|continuous|multiclass")

        @field_validator("smooth")
        def smooth_validator(cls, v):
            if isinstance(v, str):
                if v != "auto":
                    raise ValueError("smooth must be a float or 'auto'")
            elif isinstance(v, float):
                if v < 0:
                    raise ValueError(
                        "smooth must be a float greater than or equal to 0"
                    )
            return v

    def __init__(self, **kwargs):
        self.args = self.Args(**kwargs).dict()
        self.args.pop("target")
        self.encoder = None

    def fit(self, df: pd.DataFrame, target: pd.Series):
        dtype = TabularDataset(pd.DataFrame(target)).dtypes[target.name]
        if dtype == TabularDataType.NUMBER.value:
            target_type = "continuous"
        elif dtype == TabularDataType.CATEGORY.value:
            if target.nunique() > 2:
                target_type = "multiclass"
            elif target.nunique() == 2:
                target_type = "binary"

        self.args["target_type"] = target_type
        self.encoder = TargetEncoder(**self.args)
        self.encoder.fit(df, target)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.encoder.transform(df)


class DALabelEncoder:
    _extend = False
    _allowed_data_types = [TabularDataType.CATEGORY.value]
    _missing_values_allowed = True
    _dependent_variable_required = False

    class Args(ArgsBase):
        pass

    def __init__(self, **kwargs):
        self.encoders = dict()
        self.args = self.Args(**kwargs).dict()

    def fit(self, df: pd.DataFrame):
        for col in df.columns:
            self.encoders[col] = LabelEncoder().fit(df[col])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            df[col] = self.encoders[col].transform(df[col])
        return df


class DAFrequencyEncoder(CountEncoder):
    _extend = True
    _allowed_data_types = [TabularDataType.CATEGORY.value]
    _missing_values_allowed = True
    _dependent_variable_required = False

    class Args(ArgsBase):
        pass


class DAOneHotEncoder:
    _extend = False
    _allowed_data_types = [TabularDataType.CATEGORY.value]
    _missing_values_allowed = True
    _dependent_variable_required = False

    class Args(ArgsBase):
        sparse_output: bool | None = False
        handle_unknown: str | None = Field(
            "error", pattern="error|ignore|infrequent_if_exist"
        )
        max_categories: int | None = None
        min_frequency: int | float | None = None
        feature_name_combiner: str | Callable = Field("concat", pattern="concat")
        dtype: int | float = np.float64
        drop: str | list | None = None

    def __init__(self, **kwargs):
        self.args = self.Args(**kwargs).dict()
        self.encoder = None

    def fit(self, df):
        self.encoder = OneHotEncoder(**self.args)
        self.encoder.fit(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        transformed_df = pd.DataFrame(
            self.encoder.transform(df),
            columns=self.encoder.get_feature_names_out(),
        )
        return transformed_df


class PrepFeatureEngineeringEncoding(PrepBase):
    PREPROCESSOR_MAPPING = {
        "one_hot_encoder": DAOneHotEncoder,
        "label_encoder": DALabelEncoder,
        "frequency_encoder": DAFrequencyEncoder,
        "kfold_target_encoder": DATargetEncoder,
    }

    def __init__(self, targeted_cols: List[str], strategy: str, args: Dict):
        self.targeted_cols = targeted_cols
        self.strategy = strategy
        self.args = args
        self.preprocessor = None

    def fit(self, df: pd.DataFrame):
        self.preprocessor = self.PREPROCESSOR_MAPPING[self.strategy](**self.args)
        self._validate_preprocessing(
            df, self.preprocessor, self.targeted_cols, self.args
        )
        if self.args.get("target"):
            self.preprocessor.fit(df[self.targeted_cols], df[self.args["target"]])
        else:
            self.preprocessor.fit(df[self.targeted_cols])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(self.preprocessor, DAOneHotEncoder):
            transformed_df = self.preprocessor.transform(df[self.targeted_cols])
            df = df.drop(columns=self.targeted_cols)
            df = pd.concat([df, transformed_df], axis=1)
        else:
            df[self.targeted_cols] = self.preprocessor.transform(df[self.targeted_cols])
        return df
