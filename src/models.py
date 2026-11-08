"""
Main training and inference script for Mohs hardness prediction.

This script executes data loading, feature engineering, cross-validation,
Optuna hyperparameter tuning, model blending, and creates a submission file.
"""

from typing import Dict, List
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from src.config import (
    N_SPLITS,
    OPTUNA_PARAMS_RANGE,
    RANDOM_STATE,
    SUBMISSION_PATH,
    TEST_PATH,
    TRAIN_PATH,
)
from src.features import preprocess_data
from src.models import (
    get_models,
    get_scaling_required_models,
)

# Optunaのログを抑制
optuna.logging.set_verbosity(
    optuna.logging.WARNING
)


def main():
    """メイン実行関数"""
    print("--- 1. データの読み込み ---")
    train_df = pd.read_csv(
        TRAIN_PATH
    )
    test_df = pd.read_csv(TEST_PATH)

    # ターゲットと特徴量の分離
    target_col = "Hardness"  # データセットのターゲット名に合わせて調整
    if target_col not in train_df.columns:
        # フォールバック: 最終列をターゲットとする
        target_col = (
            train_df.columns[-1]
        )

    y = train_df[target_col]
    X = train_df.drop(
        columns=[target_col]
    )

    # テストデータのID列等の退避
    test_id_col = (
        "id"
        if "id" in test_df.columns
        else (
            "Id"
            if "Id" in test_df.columns
            else None
        )
    )
    test_id = (
        test_df[test_id_col].copy()
        if test_id_col
        else None
    )

    print(
        "--- 2. 前処理および特徴量エンジニアリング ---"
    )
    X_processed, test_processed = (
        preprocess_data(X, test_df)
    )

    print(
        "--- 3. Optuna による LightGBM のハイパーパラメータチューニング ---"
    )
    outer_cv = KFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    def objective(trial):
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "random_state": RANDOM_STATE,
            "verbose": -1,
            "learning_rate": (
                trial.suggest_float(
                    "learning_rate",
                    OPTUNA_PARAMS_RANGE[
                        "learning_rate_min"
                    ],
                    OPTUNA_PARAMS_RANGE[
                        "learning_rate_max"
                    ],
                    log=True,
                )
            ),
            "num_leaves": (
                trial.suggest_int(
                    "num_leaves",
                    OPTUNA_PARAMS_RANGE[
                        "num_leaves_min"
                    ],
                    OPTUNA_PARAMS_RANGE[
                        "num_leaves_max"
                    ],
                )
            ),
            "max_depth": (
                trial.suggest_int(
                    "max_depth",
                    OPTUNA_PARAMS_RANGE[
                        "max_depth_min"
                    ],
                    OPTUNA_PARAMS_RANGE[
                        "max_depth_max"
                    ],
                )
            ),
            "subsample": (
                trial.suggest_float(
                    "subsample",
                    OPTUNA_PARAMS_RANGE[
                        "subsample_min"
                    ],
                    OPTUNA_PARAMS_RANGE[
                        "subsample_max"
                    ],
                )
            ),
            "colsample_bytree": (
                trial.suggest_float(
                    "colsample_bytree",
                    OPTUNA_PARAMS_RANGE[
                        "colsample_bytree_min"
                    ],
                    OPTUNA_PARAMS_RANGE[
                        "colsample_bytree_max"
                    ],
                )
            ),
            "reg_alpha": (
                trial.suggest_float(
                    "reg_alpha",
                    OPTUNA_PARAMS_RANGE[
                        "reg_alpha_low"
                    ],
                    OPTUNA_PARAMS_RANGE[
                        "reg_alpha_high"
                    ],
                    log=True,
                )
            ),
            "reg_lambda": (
                trial.suggest_float(
                    "reg_lambda",
                    OPTUNA_PARAMS_RANGE[
                        "reg_lambda_low"
                    ],
                    OPTUNA_PARAMS_RANGE[
                        "reg_lambda_high"
                    ],
                    log=True,
                )
            ),
        }

        fold_scores = []
        for (
            tr_idx,
            va_idx,
        ) in outer_cv.split(
            X_processed, y
        ):
            X_tr = X_processed.iloc[
                tr_idx
            ]
            y_tr = y.iloc[tr_idx]
            X_va = X_processed.iloc[
                va_idx
            ]
            y_va = y.iloc[va_idx]

            model = (
                lgb.LGBMRegressor(
                    **params,
                    n_estimators=500,
                )
            )
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[
                    lgb.early_stopping(
                        stopping_rounds=30,
                        verbose=False,
                    )
                ],
            )
            preds = model.predict(X_va)
            rmse = np.sqrt(
                mean_squared_error(
                    y_va, preds
                )
            )
            fold_scores.append(rmse)

        return np.mean(fold_scores)

    study = optuna.create_study(
        direction="minimize"
    )
    study.optimize(
        objective,
        n_trials=OPTUNA_PARAMS_RANGE[
            "n_trials"
        ],
    )
    best_lgb_params = study.best_params
    print(
        f"Best Optuna RMSE: {study.best_value:.4f}"
    )

    print(
        "--- 4. アンサンブルモデルの学習と OOF 予測 ---"
    )
    ensemble_models = get_models(
        best_lgb_params
    )
    scaling_needed = set(
        get_scaling_required_models()
    )

    kf = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    oof_preds = {
        name: np.zeros(
            len(X_processed)
        )
        for name in ensemble_models.keys()
    }
    test_preds = {
        name: np.zeros(
            len(test_processed)
        )
        for name in ensemble_models.keys()
    }
    y_trues = np.zeros(
        len(X_processed)
    )

    for fold, (
        tr_idx,
        va_idx,
    ) in enumerate(
        kf.split(X_processed, y)
    ):
        print(f"--- Fold {fold + 1} ---")
        X_tr = X_processed.iloc[
            tr_idx
        ].copy()
        y_tr = y.iloc[tr_idx].copy()
        X_va = X_processed.iloc[
            va_idx
        ].copy()
        y_va = y.iloc[va_idx].copy()
        y_trues[va_idx] = y_va.values

        X_te = test_processed.copy()

        # スケーリングが必要なモデル向けの処理
        for name, model in (
            ensemble_models.items()
        ):
            clf = clone(model)
            if name in scaling_needed:
                scaler = StandardScaler()
                X_tr_p = (
                    scaler.fit_transform(
                        X_tr
                    )
                )
                X_va_p = (
                    scaler.transform(X_va)
                )
                X_te_p = (
                    scaler.transform(X_te)
                )
            else:
                X_tr_p, X_va_p, X_te_p = (
                    X_tr,
                    X_va,
                    X_te,
                )

            if name == "LightGBM":
                clf.fit(
                    X_tr_p,
                    y_tr,
                    eval_set=[
                        (X_va_p, y_va)
                    ],
                    callbacks=[
                        lgb.early_stopping(
                            stopping_rounds=50,
                            verbose=False,
                        )
                    ],
                )
            elif name == "CatBoost":
                clf.fit(
                    X_tr_p,
                    y_tr,
                    eval_set=(
                        X_va_p,
                        y_va,
                    ),
                    early_stopping_rounds=50,
                    verbose=False,
                )
            else:
                clf.fit(X_tr_p, y_tr)

            oof_preds[name][
                va_idx
            ] = clf.predict(X_va_p)
            test_preds[name] += (
                clf.predict(X_te_p)
                / N_SPLITS
            )

    print(
        "\n--- 単体モデルの OOF RMSE スコア ---"
    )
    for name, preds in oof_preds.items():
        score = np.sqrt(
            mean_squared_error(
                y_trues, preds
            )
        )
        print(
            f"{name:15s}: {score:.5f}"
        )

    print(
        "--- 5. 重み付きアンサンブル（Blending）の最適化 ---"
    )
    model_names = list(
        ensemble_models.keys()
    )

    def obj_weights(weights):
        w = np.array(weights)
        w = w / np.sum(w)
        blended = np.zeros(
            len(y_trues)
        )
        for i, name in enumerate(
            model_names
        ):
            blended += (
                w[i] * oof_preds[name]
            )
        return np.sqrt(
            mean_squared_error(
                y_trues, blended
            )
        )

    init_w = [
        1.0 / len(model_names)
    ] * len(model_names)
    bounds = [(0.0, 1.0)] * len(
        model_names
    )
    constraints = {
        "type": "eq",
        "fun": lambda w: 1 - sum(w),
    }

    res = minimize(
        obj_weights,
        init_w,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    best_weights = res.x / np.sum(
        res.x
    )

    print(
        "\n最適化されたモデルごとの重み:"
    )
    for name, w in zip(
        model_names, best_weights
    ):
        print(f"{name:15s}: {w:.4f}")

    opt_blended_oof = np.zeros(
        len(y_trues)
    )
    opt_blended_test = np.zeros(
        len(test_processed)
    )
    for i, name in enumerate(
        model_names
    ):
        opt_blended_oof += (
            best_weights[i]
            * oof_preds[name]
        )
        opt_blended_test += (
            best_weights[i]
            * test_preds[name]
        )

    final_rmse = np.sqrt(
        mean_squared_error(
            y_trues, opt_blended_oof
        )
    )
    print(
        f"\nOptimal Weighted Ensemble OOF RMSE: {final_rmse:.5f}"
    )

    # 提出ファイルの作成
    if test_id is not None:
        sub = pd.DataFrame(
            {
                test_id_col: test_id,
                target_col: opt_blended_test,
            }
        )
        sub.to_csv(
            SUBMISSION_PATH,
            index=False,
        )
        print(
            f"提出ファイルを保存しました: {SUBMISSION_PATH}"
        )


if __name__ == "__main__":
    main()
