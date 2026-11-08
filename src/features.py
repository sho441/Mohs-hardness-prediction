from typing import List, Tuple
import numpy as np
import pandas as pd


def clean_mineral_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[
    pd.DataFrame, pd.DataFrame
]:
    """
    データフレームの基本的なクリーニングと
    無限値・欠損値の安全な前処理を行う。
    """
    train = train_df.copy()
    test = test_df.copy()

    # 無限値をNaNに置換し、数値列の中央値で補完
    for df in [train, test]:
        df.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True,
        )
        numeric_cols = df.select_dtypes(
            include=[np.number]
        ).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[
                    col
                ].median()
                df[col].fillna(
                    median_val,
                    inplace=True,
                )

    return train, test


def add_composition_statistics(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_cols: List[str],
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    List[str],
]:
    """
    化学組成や物性値の統計的集約特徴量
    （行ごとの統計量）を生成する。
    """
    train = train_df.copy()
    test = test_df.copy()
    dfs = [train, test]
    added_features = []

    for df in dfs:
        # 元素・物性値の統計量（行方向）
        df["comp_mean"] = df[
            numeric_cols
        ].mean(axis=1)
        df["comp_std"] = df[
            numeric_cols
        ].std(axis=1)
        df["comp_max"] = df[
            numeric_cols
        ].max(axis=1)
        df["comp_min"] = df[
            numeric_cols
        ].min(axis=1)
        df["comp_sum"] = df[
            numeric_cols
        ].sum(axis=1)

    stat_cols = [
        "comp_mean",
        "comp_std",
        "comp_max",
        "comp_min",
        "comp_sum",
    ]
    added_features.extend(
        stat_cols
    )

    return (
        train,
        test,
        added_features,
    )


def add_interaction_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    base_cols: List[str],
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    List[str],
]:
    """
    主要な物性値・組成間の比率や積など、
    非線形な関係性を捉える相互作用特徴量を生成する。
    """
    train = train_df.copy()
    test = test_df.copy()
    dfs = [train, test]
    added_features = []

    if len(base_cols) >= 2:
        col1, col2 = (
            base_cols[0],
            base_cols[1],
        )
        for df in dfs:
            # ゼロ割を防ぐ安全な比率計算
            df["interaction_ratio"] = (
                df[col1]
                / (
                    df[col2].abs()
                    + 1e-6
                )
            )
            df[
                "interaction_product"
            ] = df[col1] * df[col2]

        interaction_cols = [
            "interaction_ratio",
            "interaction_product",
        ]
        added_features.extend(
            interaction_cols
        )

    return (
        train,
        test,
        added_features,
    )


def preprocess_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[
    pd.DataFrame, pd.DataFrame
]:
    """
    訓練データとテストデータに対して一貫性のある
    高度な前処理と特徴量エンジニアリングを一括実行する。
    """
    # 1. クリーニング
    train, test = (
        clean_mineral_data(
            train_df, test_df
        )
    )

    # 2. 数値列の抽出（目的変数やID等を除く）
    exclude_cols = [
        "id",
        "Id",
        "target",
        "Hardness",
    ]
    numeric_cols = [
        c
        for c in train.select_dtypes(
            include=[np.number]
        ).columns
        if c not in exclude_cols
    ]

    # 3. 統計的集約特徴量の追加
    (
        train,
        test,
        stat_feats,
    ) = add_composition_statistics(
        train, test, numeric_cols
    )

    # 4. 相互作用特徴量の追加（上位の数値列を活用）
    if len(numeric_cols) >= 2:
        (
            train,
            test,
            inter_feats,
        ) = (
            add_interaction_features(
                train,
                test,
                numeric_cols[:2],
            )
        )

    return train, test
