from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.constants import SERVICE_COLS
from src.data import create_advanced_features, load_and_prepare_data

RANDOM_STATE = 42
TARGET = "Отток_num"
CLASS_NAMES = ["Остался", "Ушел"]

sns.set_theme(style="darkgrid", palette="viridis", font="DejaVu Sans")
plt.rcParams["font.size"] = 11


@dataclass
class ModelResult:
    pipeline: Pipeline
    cv_mean: float
    cv_std: float
    test_accuracy: float
    test_roc_auc: float
    preds: np.ndarray
    probs: np.ndarray


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    print("3. Подготовка признаков для модели...")
    cols_to_drop = ["Отток", TARGET] + [c + "_num" for c in SERVICE_COLS]
    df_model = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    cat_cols = df_model.select_dtypes(exclude="number").columns.tolist()
    X = pd.get_dummies(df_model, columns=cat_cols, drop_first=True, dtype=int)
    y = df[TARGET]
    return X, y


def build_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Разбивает данные ДО расчёта target encoding, чтобы тест не влиял на признаки."""
    df = load_and_prepare_data()

    train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=RANDOM_STATE, stratify=df[TARGET])
    payment_means = df.loc[train_idx].groupby("Способ оплаты")[TARGET].mean()

    df = create_advanced_features(df, payment_means=payment_means)
    X, y = prepare_features(df)

    return X.loc[train_idx], X.loc[test_idx], y.loc[train_idx], y.loc[test_idx]


def build_models(y_train: pd.Series) -> dict[str, BaseEstimator]:
    weight_ratio = float((y_train == 0).sum() / (y_train == 1).sum())
    return {
        "Логистическая регрессия": LogisticRegression(
            class_weight="balanced", random_state=RANDOM_STATE, max_iter=1000
        ),
        "Случайный лес": RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_estimators=100),
        "XGBoost": XGBClassifier(
            scale_pos_weight=weight_ratio,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_estimators=100,
            max_depth=4,
        ),
    }


def evaluate_models(
    models: dict[str, BaseEstimator],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, ModelResult]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results: dict[str, ModelResult] = {}

    for name, model in models.items():
        print(f"\n--- {name} ---")
        # Масштабирование внутри Pipeline обучается заново на каждом фолде
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])

        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
        print(f"Кросс-валидация ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        probs = pipeline.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, preds)
        roc_auc = roc_auc_score(y_test, probs)
        print(f"Test Accuracy: {accuracy:.4f} | Test ROC-AUC: {roc_auc:.4f}")
        print(classification_report(y_test, preds, target_names=CLASS_NAMES))

        results[name] = ModelResult(
            pipeline=pipeline,
            cv_mean=float(cv_scores.mean()),
            cv_std=float(cv_scores.std()),
            test_accuracy=accuracy,
            test_roc_auc=roc_auc,
            preds=preds,
            probs=probs,
        )

    return results


def plot_roc_curves(results: dict[str, ModelResult], X_test: pd.DataFrame, y_test: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    for name, result in results.items():
        RocCurveDisplay.from_estimator(result.pipeline, X_test, y_test, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Случайное угадывание")
    ax.set_title("ROC-кривые моделей", fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Доля ложных срабатываний (FPR)", fontsize=12)
    ax.set_ylabel("Доля истинных срабатываний (TPR)", fontsize=12)
    ax.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(best_name: str, best: ModelResult, y_test: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    cm = confusion_matrix(y_test, best.preds)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES).plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Матрица ошибок ({best_name})", fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Предсказанный класс", fontsize=12)
    ax.set_ylabel("Истинный класс", fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_precision_recall(best_name: str, best: ModelResult, y_test: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    precision, recall, _ = precision_recall_curve(y_test, best.probs)
    pr_auc = average_precision_score(y_test, best.probs)
    ax.plot(recall, precision, label=f"{best_name} (PR-AUC: {pr_auc:.3f})", color="#2E86AB", linewidth=2.5)
    ax.set_title("Precision-Recall кривая лучшей модели", fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Полнота (Recall)", fontsize=12)
    ax.set_ylabel("Точность (Precision)", fontsize=12)
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_metrics_table(results: dict[str, ModelResult]) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    metrics_df = pd.DataFrame(
        {
            name: {
                "CV ROC-AUC": f"{r.cv_mean:.4f}",
                "Test Accuracy": f"{r.test_accuracy:.4f}",
                "Test ROC-AUC": f"{r.test_roc_auc:.4f}",
            }
            for name, r in results.items()
        }
    ).T
    table = ax.table(
        cellText=metrics_df.values,
        colLabels=metrics_df.columns,
        rowLabels=metrics_df.index,
        cellLoc="center",
        loc="center",
        colColours=["#2E86AB"] * len(metrics_df.columns),
        rowColours=["#f0f0f0"] * len(metrics_df.index),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.3, 2.0)
    ax.set_title("Сравнение метрик моделей", fontsize=14, fontweight="bold", pad=30)
    plt.tight_layout()
    plt.show()


def train_and_evaluate() -> None:
    X_train, X_test, y_train, y_test = build_dataset()

    print("4. Обучение моделей с кросс-валидацией...")
    models = build_models(y_train)
    results = evaluate_models(models, X_train, X_test, y_train, y_test)

    # Модель выбираем по CV на train, тест остаётся для финальной честной оценки
    best_name = max(results, key=lambda name: results[name].cv_mean)
    best = results[best_name]

    plot_roc_curves(results, X_test, y_test)
    plot_confusion_matrix(best_name, best, y_test)
    plot_precision_recall(best_name, best, y_test)
    plot_metrics_table(results)

    print(
        f"\nЛучшая модель (по CV): {best_name} | "
        f"CV ROC-AUC = {best.cv_mean:.4f}, Test ROC-AUC = {best.test_roc_auc:.4f}"
    )


if __name__ == "__main__":
    print("=== Начало обучения моделей ===\n")
    train_and_evaluate()
