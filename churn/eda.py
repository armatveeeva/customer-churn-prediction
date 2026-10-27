"""Разведочный анализ данных (EDA) проекта прогнозирования оттока."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from churn.data import load_and_prepare_data

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sns.set_theme(style="darkgrid", palette="viridis", font="DejaVu Sans")
plt.rcParams.update(
    {
        "figure.figsize": (14, 6),
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
    }
)

COLORS = {"success": "#4ECDC4", "danger": "#FF6B6B", "neutral": "#95A5A6"}


def plot_target_distribution(df: pd.DataFrame) -> None:
    """Распределение целевой переменной."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    counts = df["Отток"].value_counts()

    bars = axes[0].bar(
        ["Остался", "Ушел"],
        [counts["Остался"], counts["Ушел"]],
        color=[COLORS["success"], COLORS["danger"]],
        edgecolor="black",
        linewidth=1.2,
    )
    axes[0].set_title("Распределение клиентов по статусу оттока", fontweight="bold", pad=20)
    axes[0].set_xlabel("Статус клиента")
    axes[0].set_ylabel("Количество клиентов")
    for bar, count in zip(bars, [counts["Остался"], counts["Ушел"]]):
        height = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 100,
            f"{count}\n({count / len(df) * 100:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    wedges, _, autotexts = axes[1].pie(
        [counts["Остался"], counts["Ушел"]],
        labels=["Остался", "Ушел"],
        autopct="%1.1f%%",
        startangle=90,
        colors=[COLORS["success"], COLORS["danger"]],
        explode=(0.05, 0.05),
        shadow=True,
    )
    for t in autotexts:
        t.set_color("white")
        t.set_fontsize(13)
        t.set_fontweight("bold")
    axes[1].set_title("Доля оттока клиентов", fontweight="bold", pad=20)

    plt.tight_layout()
    plt.show()


def plot_numerical_features(df: pd.DataFrame) -> None:
    """Boxplot числовых признаков по статусу оттока."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    num_cols = ["Срок обслуживания", "Ежемесячный платеж", "Общий платеж"]
    titles = ["Срок обслуживания (месяцы)", "Ежемесячный платеж ($)", "Общий платеж ($)"]

    for ax, col, title in zip(axes, num_cols, titles):
        sns.boxplot(
            data=df, x="Отток", y=col, ax=ax,
            palette=[COLORS["success"], COLORS["danger"]],
        )
        ax.set_title(title, fontweight="bold", pad=15)
        ax.set_xlabel("Статус клиента")

    fig.suptitle("Распределение числовых признаков по статусу оттока", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_categorical_features(df: pd.DataFrame) -> None:
    """Countplot категориальных признаков."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    cat_cols = ["Контракт", "Тип интернета", "Способ оплаты"]
    titles = ["Тип контракта", "Тип интернет-услуг", "Способ оплаты"]

    for ax, col, title in zip(axes, cat_cols, titles):
        sns.countplot(data=df, x=col, hue="Отток", ax=ax, palette=[COLORS["success"], COLORS["danger"]])
        ax.set_title(title, fontweight="bold", pad=15)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(["Остался", "Ушел"], loc="upper right")

    fig.suptitle("Анализ категориальных признаков", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Матрица корреляций с целевой переменной."""
    df_encoded = pd.get_dummies(df, drop_first=True, dtype=int)
    churn_cols = [c for c in df_encoded.columns if "Ушел" in c]
    if not churn_cols:
        logger.warning("Не удалось найти колонку оттока")
        return

    churn_col = churn_cols[0]
    corr_with_target = df_encoded.corr()[churn_col].drop(churn_col).sort_values(ascending=False)
    top_corr = corr_with_target.head(15).index.tolist() + [churn_col]

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        df_encoded[top_corr].corr(),
        annot=True, cmap="coolwarm", fmt=".2f",
        linewidths=0.5, center=0, square=True,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Матрица корреляций: Топ-15 признаков с целевой переменной", fontweight="bold", pad=20)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Запуск EDA-анализа."""
    logger.info("Начало EDA-анализа...")
    df = load_and_prepare_data()

    logger.info("1. Анализ целевой переменной...")
    plot_target_distribution(df)

    logger.info("2. Анализ числовых признаков...")
    plot_numerical_features(df)

    logger.info("3. Анализ категориальных признаков...")
    plot_categorical_features(df)

    logger.info("4. Построение матрицы корреляций...")
    plot_correlation_heatmap(df)

    logger.info("EDA-анализ завершён.")


if __name__ == "__main__":
    main()
