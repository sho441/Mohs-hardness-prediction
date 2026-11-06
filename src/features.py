import numpy as np
import pandas as pd


def handle_missing_and_infinite(df: pd.DataFrame) -> pd.DataFrame:
    """
    データフレーム内の欠損値および無限値を適切に処理する。
    """
    df = df.copy()
    # 無限値をNaNに置換
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 数値カラムの欠損値を中央値で補完（必要に応じて調整）
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    ドメイン知識や統計量に基づく派生特徴量を生成する。
    """
    df = df.copy()
    
    # ここに鉱物の物理特性や化学組成に関する特徴量生成ロジックを追加する
    # 例: 統計量や比率特徴量など
    # numeric_cols = df.select_dtypes(include=[np.number]).columns
    # if len(numeric_cols) >= 2:
    #     # 例としてのダミー処理（実際の特徴量に合わせて書き換えてください）
    #     pass
        
    return df


def preprocess_data(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    学習データとテストデータに対して一貫性のある前処理と
    特徴量エンジニアリングを一括実行する。
    """
    # 1. 欠損値・無限値の処理
    train_df = handle_missing_and_infinite(train_df)
    test_df = handle_missing_and_infinite(test_df)
    
    # 2. 特徴量エンジニアリング
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)
    
    return train_df, test_df
