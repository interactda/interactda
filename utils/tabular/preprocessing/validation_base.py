from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import List, Dict, Any
from utils.tabular.prep_opts import PREPROCESSING_OPTIONS


class ArgsBase(BaseModel):
    class Config:
        extra = "forbid"


class PrepStepValidationBase(BaseModel):
    action: str = Field(...)
    targeted_cols: str | List[str] = Field(...)
    include: List[str] | None = None
    exclude: List[str] | None = None
    method: str = Field(...)
    strategy: str = Field(...)
    args: Dict[str, Any]

    class Config:
        extra = "forbid"

    @field_validator("action")
    def validate_action(cls, action):
        if action not in list(PREPROCESSING_OPTIONS.keys()):
            raise ValueError(
                "action must be one of the following: {}".format(
                    list(PREPROCESSING_OPTIONS.keys())
                )
            )
        return action

    @field_validator("method")
    def validate_method(cls, method, values):
        try:
            action = values.data["action"]
            method_opts = list(PREPROCESSING_OPTIONS[action].keys())
        except KeyError:
            raise ValueError(
                "action must be one of the following: {}".format(
                    list(PREPROCESSING_OPTIONS.keys())
                )
            )
        if method not in method_opts:
            raise ValueError(
                "method must be one of the following: {}".format(
                    PREPROCESSING_OPTIONS[action]
                )
            )
        return method

    @field_validator("strategy")
    def validate_strategy(cls, strategy, values):
        try:
            action = values.data["action"]
            method = values.data["method"]
            strategy_opts = PREPROCESSING_OPTIONS[action][method]
        except KeyError:
            raise ValueError(
                "method must be one of the following: {}".format(
                    list(PREPROCESSING_OPTIONS[action].keys())
                )
            )
        if strategy not in strategy_opts:
            raise ValueError(
                "strategy must be one of the following: {}".format(
                    PREPROCESSING_OPTIONS[action][method]
                )
            )
        return strategy


if __name__ == "__main__":
    try:
        PrepStepValidationBase(
            action="missing_values",
            targeted_cols="a",
            method="imputation",
            strategy="simple_imputer",
            args={"fill_value": 0},
        )
    except ValidationError as e:
        print(e)
