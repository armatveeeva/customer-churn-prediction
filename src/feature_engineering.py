import matplotlib.pyplot as plt
import pandas as pd

from src.constants import FE_COLORS
from src.data import create_advanced_features, load_and_prepare_data


def encode_and_select_features(df: pd.DataFrame) -> pd.Series:
    print("3. Кодирование и отбор признаков...")
    cols_to_exclude = ["Отток", "Отток_num"] + [
        c + "_num"
        for c in ["Онлайн-защита", "Онлайн-бекап", "Защита устройств", "Техподдержка", "Стриминг ТВ", "Стриминг кино"]
    ]

    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    cat_cols_to_encode = [col for col in cat_cols if col not in cols_to_exclude]

    df_encoded = pd.get_dummies(df, columns=cat_cols_to_encode, drop_first=True, dtype=int)
    df_numeric = df_encoded.select_dtypes(include="number")

    corr_with_target = df_numeric.corr()["Отток_num"].drop("Отток_num").abs().sort_values(ascending=False)
    return corr_with_target.head(15)


def plot_feature_importance(importance_series: pd.Series):
    print("4. Визуализация важности признаков...")
    plt.figure(figsize=(10, 8))
    importance_series = importance_series[::-1]

    new_features_keywords = [
        "Соотношение",
        "Количество",
        "Группа",
        "Есть оптоволокно",
        "Есть стриминг",
        "Способ оплаты_TE",
    ]
    colors = [
        FE_COLORS["accent"] if any(kw in str(x) for kw in new_features_keywords) else FE_COLORS["primary"]
        for x in importance_series.index
    ]

    bars = plt.barh(importance_series.index, importance_series.values, color=colors, edgecolor="black", linewidth=0.5)

    plt.title(
        "Топ-15 наиболее важных признаков для прогноза оттока\n(синие - исходные, оранжевые - созданные нами)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Абсолютная корреляция с целевой переменной", fontsize=12)
    plt.ylabel("Признак", fontsize=12)
    plt.tick_params(axis="both", which="major", labelsize=10)

    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = load_and_prepare_data()
    df = create_advanced_features(df)
    importance = encode_and_select_features(df)
    plot_feature_importance(importance)
