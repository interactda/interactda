from enum import Enum
import pandas as pd
from warnings import filterwarnings

filterwarnings("ignore")


class TabularDataType(Enum):
    NUMBER = "number"
    CATEGORY = "category"
    DATETIME = "datetime"
    TEXT = "text"
    UNRECOGNIZED = "unrecognized"


class TabularDataset:
    def __init__(
        self,
        df,
        text_cols=None,
        text_unique_threshold=0.7,
        category_unique_cnt_threshold=50,
    ):
        self.df = df
        self.dtypes = self._define_dtypes(
            text_cols, text_unique_threshold, category_unique_cnt_threshold
        )

    def _define_dtypes(
        self, text_cols, text_unique_threshold, category_unique_cnt_threshold
    ):
        dtype_dict = {}

        for col in self.df.columns:
            tmp_col = self.df[col].dropna()

            if pd.api.types.is_any_real_numeric_dtype(tmp_col):
                dtype_dict[col] = TabularDataType.NUMBER.value

            elif pd.api.types.is_datetime64_any_dtype(tmp_col):
                dtype_dict[col] = TabularDataType.DATETIME.value

            elif pd.api.types.is_string_dtype(tmp_col):
                if not text_cols:
                    unique_ratio = len(tmp_col.unique()) / len(tmp_col)

                    if (
                        unique_ratio > text_unique_threshold
                        and tmp_col.nunique() > category_unique_cnt_threshold
                    ):
                        dtype_dict[col] = TabularDataType.TEXT.value
                    else:
                        dtype_dict[col] = TabularDataType.CATEGORY.value
                else:
                    if col in text_cols:
                        dtype_dict[col] = TabularDataType.TEXT.value
                    else:
                        dtype_dict[col] = TabularDataType.CATEGORY.value

            elif pd.api.types.is_bool_dtype(tmp_col):
                self.df[col] = self.df[col].apply(lambda x: 1 if x else 0)
                dtype_dict[col] = TabularDataType.NUMBER.value

            else:
                dtype_dict[col] = TabularDataType.UNRECOGNIZED.value

        return dtype_dict


if __name__ == "__main__":
    df = pd.read_csv(
        "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    )
    print(TabularDataset(df).dtypes)
    # {'PassengerId': <TabularDataType.UNRECOGNIZED: 'unrecognized'>, 'Survived': <TabularDataType.BINARY: 'binary'>, 'Pclass': <TabularDataType.NUMBER: 'number'>, 'Name': <TabularDataType.UNRECOGNIZED: 'unrecognized
