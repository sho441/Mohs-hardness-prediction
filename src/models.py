import optuna
import lightgbm as lgbm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
from src.config import OPTUNA_PARAMS_RANGE, LGBM_DEFAULT_PARAMS


def objective(trial, X, y, n_splits=5):
    """
    Optunaの最適化目的関数（CVのRMSEを最小化）。
    """
    params = LGBM_DEFAULT_PARAMS.copy()
    
    # Optunaによる探索パラメータの設定
    params["num_leaves"] = trial.suggest_int(
        "num_leaves",
        OPTUNA_PARAMS_RANGE["num_leaves_min"],
        OPTUNA_PARAMS_RANGE["num_leaves_max"]
    )
    params["max_depth"] = trial.suggest_int(
        "max_depth",
        OPTUNA_PARAMS_RANGE["max_depth_min"],
        OPTUNA_PARAMS_RANGE["max_depth_max"]
    )
    params["learning_rate"] = trial.suggest_float(
        "learning_rate",
        OPTUNA_PARAMS_RANGE["learning_rate_min"],
        OPTUNA_PARAMS_RANGE["learning_rate_max"],
        log=True
    )
    params["subsample"] = trial.suggest_float(
        "subsample",
        OPTUNA_PARAMS_RANGE["subsample_min"],
        OPTUNA_PARAMS_RANGE["subsample_max"]
    )
    params["colsample_bytree"] = trial.suggest_float(
        "colsample_bytree",
        OPTUNA_PARAMS_RANGE["colsample_bytree_min"],
        OPTUNA_PARAMS_RANGE["colsample_bytree_max"]
    )
    params["reg_alpha"] = trial.suggest_float(
        "reg_alpha",
        OPTUNA_PARAMS_RANGE["reg_alpha_low"],
        OPTUNA_PARAMS_RANGE["reg_alpha_high"],
        log=True
    )
    params["reg_lambda"] = trial.suggest_float(
        "reg_lambda",
        OPTUNA_PARAMS_RANGE["reg_lambda_low"],
        OPTUNA_PARAMS_RANGE["reg_lambda_high"],
        log=True
    )
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse_scores = []
    
    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgbm.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgbm.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        rmse_scores.append(rmse)
        
    return np.mean(rmse_scores)


def optimize_hyperparameters(X, y, n_trials=50):
    """
    Optunaを用いて最適なハイパーパラメータを探索する。
    """
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, X, y),
        n_trials=n_trials
    )
    
    best_params = LGBM_DEFAULT_PARAMS.copy()
    best_params.update(study.best_params)
    return best_params
