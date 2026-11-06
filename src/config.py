from pathlib import Path

# --- プロジェクトパス設定 ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SUBMISSION_PATH = DATA_DIR / "submission.csv"

# --- 共通設定 ---
RANDOM_STATE = 42
N_SPLITS = 5  # 交差検証の分割数

# --- LightGBM デフォルトパラメータ ---
LGBM_DEFAULT_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "random_state": RANDOM_STATE,
    "n_estimators": 1000,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbose": -1,
}

# --- Optuna チューニング用探索範囲 ---
OPTUNA_PARAMS_RANGE = {
    "num_leaves_min": 15,
    "num_leaves_max": 127,
    "max_depth_min": 3,
    "max_depth_max": 10,
    "learning_rate_min": 0.01,
    "learning_rate_max": 0.1,
    "subsample_min": 0.6,
    "subsample_max": 1.0,
    "colsample_bytree_min": 0.6,
    "colsample_bytree_max": 1.0,
    "reg_alpha_low": 1e-8,
    "reg_alpha_high": 10.0,
    "reg_lambda_low": 1e-8,
    "reg_lambda_high": 10.0,
    "n_trials": 50,  # Optunaの試行回数
}
