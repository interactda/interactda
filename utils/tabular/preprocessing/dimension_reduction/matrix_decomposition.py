from tabular.dataset import TabularDataType
from utils.tabular.preprocessing.preprocessing_base import PrepBase
from typing import List, Dict, Callable
import pandas as pd
from pydantic import Field
from utils.tabular.preprocessing.validation_base import ArgsBase
from sklearn.decomposition import PCA, NMF, KernelPCA, TruncatedSVD


class DAPCA(PCA):
    _extend = True
    _allowed_data_types = [TabularDataType.NUMBER.value]
    _missing_values_allowed = False
    _dependent_variable_required = False
    _only_on_training_set = False
    _reference = "https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html"

    class Args(ArgsBase):
        n_components: int | float | str
        svd_solver: str = Field("auto", pattern="auto|full|arpack|randomized")
        tol: float = 0.0
        n_oversamples: int | None = 10
        iterated_power: int | str = "auto"
        power_iteration_normalizer: str = Field("auto", pattern="auto|QR|LU|none")
        random_state: int | None = None
        whiten: bool = False

    def __init__(self, **kwargs: str):
        self.args = self.Args(**kwargs).dict()
        super().__init__(**self.args)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        transformed_array = super().transform(df)
        transformed_df = pd.DataFrame(
            transformed_array,
            columns=[f"dim_{i}" for i in range(transformed_array.shape[1])],
        )
        return transformed_df


class DAKernelPCA(KernelPCA):
    _extend = True
    _allowed_data_types = [TabularDataType.NUMBER.value]
    _missing_values_allowed = False
    _dependent_variable_required = False
    _only_on_training_set = False
    reference = "https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html"

    class Args(ArgsBase):
        n_components: int | float | str = Field(..., gt=0)
        kernel: str | Callable = "linear"
        gamma: float | None = None
        degree: float = 3
        coef0: float = 1
        kernel_params: dict | None = None
        alpha: float = 1.0
        fit_inverse_transform: bool = False
        eigen_solver: str = "auto"
        tol: float = 0
        max_iter: int | None = None
        iterated_power: int | str = "auto"
        remove_zero_eig: bool = False
        random_state: int | None = None
        copy_X: bool = True

    def __init__(self, **kwargs: str):
        self.args = self.Args(**kwargs).dict()
        super().__init__(**self.args)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        transformed_array = super().transform(df)
        transformed_df = pd.DataFrame(
            transformed_array,
            columns=[f"dim_{i}" for i in range(transformed_array.shape[1])],
        )
        return transformed_df


class DANMF(NMF):
    _extend = True
    _allowed_data_types = [TabularDataType.NUMBER.value]
    _missing_values_allowed = False
    _dependent_variable_required = False
    _only_on_training_set = False
    _reference = "https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html"

    class Args(ArgsBase):
        n_components: int | float | str = Field(..., gt=0)
        init: str | None = Field(None, pattern="nndsvd|nndsvda|nndsvdar|random")
        solver: str = Field("cd", pattern="cd|mu")
        beta_loss: str | float = "frobenius"
        tol: float = Field(1e-4, ge=0)
        max_iter: int = Field(200, gt=0)
        random_state: int | None = None
        alpha_W: float = Field(0.0, ge=0)
        alpha_H: float | str = "same"
        l1_ratio: float = Field(0, ge=0, le=1)
        verbose: int = Field(0, ge=0)
        shuffle: bool = False

    def __init__(self, **kwargs: str):
        self.args = self.Args(**kwargs).dict()
        super().__init__(**self.args)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        transformed_array = super().transform(df)
        transformed_df = pd.DataFrame(
            transformed_array,
            columns=[f"dim_{i}" for i in range(transformed_array.shape[1])],
        )
        return transformed_df


class DATruncatedSVD(TruncatedSVD):
    _extend = True
    _allowed_data_types = [TabularDataType.NUMBER.value]
    _missing_values_allowed = False
    _dependent_variable_required = False
    _only_on_training_set = False
    _reference = "https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html"

    class Args(ArgsBase):
        n_components: int = 2
        algorithm: str = Field("randomized", pattern="randomized|arpack")
        n_iter: int = 5
        n_oversamples: int = 10
        power_iteration_normalizer: str = Field("auto", pattern="auto|QR|LU|none")
        random_state: int | None = None
        tol: float = 0.0

    def __init__(self, **kwargs: str):
        self.args = self.Args(**kwargs).dict()
        super().__init__(**self.args)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        transformed_array = super().transform(df)
        transformed_df = pd.DataFrame(
            transformed_array,
            columns=[f"dim_{i}" for i in range(transformed_array.shape[1])],
        )
        return transformed_df


class PrepDimensionReductionMatrixDecomposition(PrepBase):
    PREPROCESSOR_MAPPING = {
        "pca": DAPCA,
        "nmf": DANMF,
        "kernel_pca": DAKernelPCA,
        "truncate_svd": DATruncatedSVD,
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
