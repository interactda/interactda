from sklearn.impute import KNNImputer
import pandas as pd
from tabular.dataset import TabularDataType, TabularDataset
from utils.tabular.preprocessing.preprocessing_base import PrepBase
from sklearn.linear_model import LinearRegression
from utils.tabular.preprocessing.validation_base import ArgsBase
from pydantic import Field
import numpy as np
from typing import List, Dict, Callable


class DAKNNImputer(KNNImputer):
    _extend = True
    _allowed_data_types = [TabularDataType.NUMBER.value]
    _allow_missing_values = False
    _dependent_variable_required = True
    _reference = "https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html"

    class Args(ArgsBase):
        independent_variables: List[str] = Field(..., min_items=1)
        missing_values: str | int | float | None = np.nan
        n_neighbors: int = Field(5, gt=1)
        weights: str | Callable = "uniform"
        metric: str = Field("nan_euclidean", pattern="nan_euclidean|nan_manhattan")
        add_indicator: bool = False
        keep_empty_features: bool = False

    def __init__(self, **kwargs):
        self.args = self.Args(**kwargs).dict()
        super().__init__(**self.args)

    def predict(self, df: pd.DataFrame):
        transformed_df = pd.DataFrame(
            self.transform(df), columns=df.columns, index=df.index
        )
        return transformed_df


class DALinearRegression(LinearRegression):
    _extend = True
    _allowed_data_types = [TabularDataType.NUMBER.value]
    _allow_missing_values = False
    _dependent_variable_required = True
    _reference = "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html"

    class Args(ArgsBase):
        independent_variables: list[str] = Field(...)

    def __init__(self, **kwargs):
        self.args = self.Args(**kwargs).dict()
        super().__init__(**self.args)


class PrepMissingValuesPredictImputer(PrepBase):
    PREPROCESSOR_MAPPING = {
        "linear_regression": DALinearRegression,
        "knn": DAKNNImputer,
    }

    def __init__(self, targeted_cols: List[str], strategy: str, args: Dict):
        self.targeted_cols = targeted_cols
        self.strategy = strategy
        self.args = args
        self.preprocessor = None
        self.dtypes = None

    def fit(self, df: pd.DataFrame):
        independent_vars = self.args.get("independent_variables")
        new_args = self.args.copy()
        new_args.pop("independent_variables")
        predictor_class = self.PREPROCESSOR_MAPPING[self.strategy]
        idep_dtypes = TabularDataset(df[independent_vars]).dtypes
        assert all(
            [
                idep_dtypes[idep_var] in predictor_class._allowed_data_types
                for idep_var in independent_vars
            ]
        ), f"All independent variables should be of type {predictor_class._allowed_data_types}, got {idep_dtypes}."

        self.preprocessor = []
        for col in self.targeted_cols:
            predictor = predictor_class(**new_args)
            without_na_ds = df[~df[col].isna()][col]
            if self.strategy == "linear_regression":
                predictor.fit(
                    df.loc[without_na_ds.index, independent_vars], without_na_ds
                )
            elif self.strategy == "knn":
                predictor.fit(
                    df.loc[without_na_ds.index, independent_vars + [col]], without_na_ds
                )
            self.preprocessor.append(predictor)

    def transform(self, df: pd.DataFrame):
        independent_vars = self.args.get("independent_variables")
        for idx, col in enumerate(self.targeted_cols):
            with_na_ds = df[df[col].isna()][col]
            if len(with_na_ds) == 0:
                continue
            if self.strategy == "linear_regression":
                df.loc[with_na_ds.index, col] = self.preprocessor[idx].predict(
                    df.loc[with_na_ds.index, independent_vars]
                )
            elif self.strategy == "knn":
                df.loc[with_na_ds.index, col] = self.preprocessor[idx].predict(
                    df.loc[with_na_ds.index, independent_vars + [col]]
                )[col]

        return df
