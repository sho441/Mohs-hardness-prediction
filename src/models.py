from typing import Dict, List
import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb
from sklearn.ensemble import (
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from src.config import (
    CATBOOST_DEFAULT_PARAMS,
    LGBM_DEFAULT_PARAMS,
    XGBOOST_DEFAULT_PARAMS,
)


def get_models(
    best_lgb_params: Dict = None,
) -> Dict[str, object]:
    """
    アンサンブル学習に使用する
    複数回帰モデルの辞書を返す。
    """
    lgb_params = (
        LGBM_DEFAULT_PARAMS.copy()
    )
    if best_lgb_params:
        lgb_params.update(
            best_lgb_params
        )

    cat_params = (
        CATBOOST_DEFAULT_PARAMS.copy()
    )
    xgb_params = (
        XGBOOST_DEFAULT_PARAMS.copy()
    )

    models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": (
            RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
            )
        ),
        "LightGBM": (
            lgb.LGBMRegressor(
                **lgb_params
            )
        ),
        "XGBoost": (
            xgb.XGBRegressor(
                **xgb_params
            )
        ),
        "CatBoost": (
            CatBoostRegressor(
                **cat_params
            )
        ),
    }
    return models


def get_scaling_required_models() -> (
    List[str]
):
    """
    StandardScalerによるスケーリングが
    推奨されるモデルのリストを返す。
    """
    return ["Ridge"]
