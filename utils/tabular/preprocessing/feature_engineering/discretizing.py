from pydantic import Field
from sklearn.preprocessing import KBinsDiscretizer, Binarizer
import pandas as pd
from tabular.dataset import TabularDataType
from utils.tabular.preprocessing.preprocessing_base import PrepBase
from utils.tabular.preprocessing.validation_base import ArgsBase
from typing import List, Dict


class DABinarizer:
    _extend = True
    _allowed_data_types = [TabularDataType.NUMBER.value]
    _missing_values_allowed = False
    _dependent_variable_required = False

    class Args(ArgsBase):
        threshold: float | dict | None = Field(0.0)
        thresholds: dict | None = Field({})

    def __init__(self, **kwargs):
        self.args = self.Args(**kwargs).dict()
        self.binarizer = None

    def fit(self, df: pd.DataFrame):
        if self.args.get("threshold"):
            self.binarizer = Binarizer(**self.args)
            self.binarizer.fit(df)
        elif self.args.get("thresholds"):
            self.binarizer = []

            for col, threshold in self.args.get("thresholds").items():
                if col in df.columns:
                    self.binarizer.append(Binarizer(threshold=threshold).fit(df[[col]]))
        else:
            raise KeyError(
                "Either threshold or thresholds must be provided in the args"
            )

    def transform(self, df: pd.DataFrame):

        if self.args.get("threshold"):
            return pd.DataFrame(
                self.binarizer.transform(df),
                columns=self.binarizer.get_feature_names_out(),
            )

        elif self.args.get("thresholds"):
            transformed_df = pd.DataFrame()
            for i, (col, threshold) in enumerate(self.args.get("thresholds").items()):
                if col in df.columns:
                    transformed_df[col] = self.binarizer[i].transform(df[[col]]).ravel()
            return transformed_df


class DAKBinsDiscretizer:
    _extend = False
    _allowed_data_types = [TabularDataType.NUMBER.value]
    _missing_values_allowed = False
    _dependent_variable_required = False

    class Args(ArgsBase):
        n_bins: int = Field(5, gt=1)
        encode: str = Field("ordinal", pattern="^(onehot|onehot-dense|ordinal)$")
        strategy: str = Field("quantile", pattern="^(uniform|quantile|kmeans)$")
        subsample: str | int | None = Field("warn")
        random_state: int | None = Field(None)

    def __init__(self, **kwargs):
        self.args = self.Args(**kwargs).dict()
        self.discretizer = None

    def fit(self, df: pd.DataFrame):
        self.discretizer = KBinsDiscretizer(**self.args)
        self.discretizer.fit(df)

    def transform(self, df: pd.DataFrame):
        transformed_df = pd.DataFrame(
            (
                self.discretizer.transform(df).toarray()
                if "onehot" in self.args.get("encode", [])
                else self.discretizer.transform(df)
            ),
            columns=self.discretizer.get_feature_names_out(),
        )
        return transformed_df


class PrepFeatureEngineeringDiscretizing(PrepBase):
    PREPROCESSOR_MAPPING = {
        "kbins_discretizer": DAKBinsDiscretizer,
        "binarizer": DABinarizer,
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
        if "onehot" in self.args.get("encode", []):
            transformed_df = self.preprocessor.transform(df[self.targeted_cols])
            df = df.drop(columns=self.targeted_cols)
            df = pd.concat([df, transformed_df], axis=1)
        else:
            df[self.targeted_cols] = self.preprocessor.transform(df[self.targeted_cols])
        return df
