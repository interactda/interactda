from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
)
from tabular.dataset import TabularDataType
from utils.tabular.preprocessing.preprocessing_base import PrepBase
from typing import List, Dict, Callable
import pandas as pd
from utils.tabular.preprocessing.validation_base import ArgsBase
from pydantic import Field


class DALDA(LinearDiscriminantAnalysis):
    _extend = True
    _allowed_data_types = [TabularDataType.NUMBER.value]
    _missing_values_allowed = False
    _dependent_variable_required = True
    _only_on_training_set = False
    _reference = "https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html"

    class Args(ArgsBase):
        target: str = Field(...)
        solver: str = Field("svd", pattern="svd|lsqr|eigen")
        shrinkage: str | float | None = None
        priors: List[float] | None = None
        n_components: int | None = None
        store_covariance: bool = False
        tol: float = 1e-4
        covariance_estimator: Callable | None = None

    def __init__(self, **kwargs: str):
        self.args = self.Args(**kwargs).dict()
        self.args.pop("target")
        super().__init__(**self.args)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        transformed_array = super().transform(df)
        transformed_df = pd.DataFrame(
            transformed_array,
            columns=[f"dim_{i}" for i in range(transformed_array.shape[1])],
        )
        return transformed_df


class PrepDimensionReductionDiscriminantAnalysis(PrepBase):
    PREPROCESSOR_MAPPING = {"lda": DALDA}

    def __init__(self, targeted_cols: List[str], strategy: str, args: Dict):
        self.targeted_cols = targeted_cols
        self.strategy = strategy
        self.args = args
        self.preprocessor = None
        self.dtypes = None

    def fit(self, df: pd.DataFrame):
        self.targeted_cols = self.targeted_cols
        target = self.args["target"]
        self.preprocessor = self.PREPROCESSOR_MAPPING[self.strategy](**self.args)
        self.preprocessor.fit(df[self.targeted_cols], df[target])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.preprocessor.transform(df[self.targeted_cols])
        return df
