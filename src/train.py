import pandas as pd
import numpy as np
import lightgbm as lgbm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from src.config import (
    TRAIN_PATH, TEST_PATH, SUBMISSION_PATH,
    RANDOM_STATE, N_SPLITS, LGBM_DEFAULT_PARAMS,
    OPTUNA_PARAMS_RANGE
)
from src.features import preprocess_data
from src.models import optimize_hyperparameters


def main():
    print("1. データの読み込み中...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    # ターゲットと特徴量の分離（例: ターゲット名が 'target' の場合）
    target_col = "target"  # 実際のデータセットのカラム名に変更してください
    if target_col in train_df.columns:
        y = train_df[target_col]
        X = train_df.drop(columns=[target_col])
    else:
        # カラム名が異なる場合のフォールバック（例: 最終列をターゲットとする等）
        target_col = train_df.columns[-1]
        y = train_df[target_col]
        X = train_df.drop(columns=[target_col])
        
    # テストデータ側にIDカラム等があれば保持・除外の処理を記述
    
    print("2. 前処理および特徴量エンジニアリングの実行中...")
    X, test_processed = preprocess_data(X, test_df)
    
    print("3. Optunaによるハイパーパラメータ最適化中...")
    best_params = optimize_hyperparameters(
        X, y, n_trials=OPTUNA_PARAMS_RANGE["n_trials"]
    )
    print(f"最適パラメータ: {best_params}")
    
    print("4. 交差検証（Cross-Validation）による学習と評価...")
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(test_processed))
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgbm.LGBMRegressor(**best_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgbm.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        # 検証データの予測
        val_preds = model.predict(X_val)
        oof_preds[val_idx] = val_preds
        
        fold_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        cv_scores.append(fold_rmse)
        print(f"Fold {fold + 1} RMSE: {fold_rmse:.5f}")
        
        # テストデータの予測（各foldの平均をとる場合）
        test_preds += model.predict(test_processed) / N_SPLITS
        
    overall_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    print(f"==============================")
    print(f"全体 OOF RMSE: {overall_rmse:.5f}")
    print(f"==============================")
    
    # 5. 提出用ファイルの作成
    # submission = pd.DataFrame({"id": test_id, "target": test_preds})
    # submission.to_csv(SUBMISSION_PATH, index=False)
    # print(f"提出ファイルを保存しました: {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
