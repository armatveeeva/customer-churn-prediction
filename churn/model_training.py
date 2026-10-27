"""Обучение и оценка моделей прогнозирования оттока."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from churn.data import create_advanced_features, load_and_prepare_data, prepare_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sns.set_theme(style="darkgrid", palette="viridis", font="DejaVu Sans")
plt.rcParams["font.size"] = 11


def train_and_evaluate() -> None:
    """Обучает 3 модели с кросс-валидацией и строит графики."""
    df = load_and_prepare_data()
    df = create_advanced_features(df)
    X, y = prepare_features(df)

    logger.info("4. Обучение моделей с кросс-валидацией...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    weight_ratio = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    models = {
        "Логистическая регрессия": LogisticRegression(
            class_weight="balanced", random_state=42, max_iter=1000
        ),
        "Случайный лес": RandomForestClassifier(
            class_weight="balanced", random_state=42, n_estimators=100
        ),
        "XGBoost": XGBClassifier(
            scale_pos_weight=weight_ratio,
            random_state=42,
            eval_metric="logloss",
            n_estimators=100,
            max_depth=4,
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results: dict[str, dict[str, str]] = {}
    best_model_name: str | None = None
    best_roc = 0.0
    best_preds: np.ndarray | None = None
    best_probs: np.ndarray | None = None

    for name, model in models.items():
        logger.info("\n--- %s ---", name)
        cv_scores = []
        for train_idx, val_idx in cv.split(X_train_scaled, y_train):
            X_cv_train = X_train_scaled[train_idx]
            X_cv_val = X_train_scaled[val_idx]
            y_cv_train = y_train.iloc[train_idx]
            y_cv_val = y_train.iloc[val_idx]

            model_clone = model.__class__(**model.get_params())
            model_clone.fit(X_cv_train, y_cv_train)
            y_cv_pred = model_clone.predict(X_cv_val)
            cv_scores.append(roc_auc_score(y_cv_val, y_cv_pred))

        mean_cv = np.mean(cv_scores)
        std_cv = np.std(cv_scores)
        logger.info("Кросс-валидация ROC-AUC: %.4f (+/- %.4f)", mean_cv, std_cv)

        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1]
        acc = accuracy_score(y_test, preds)
        roc = roc_auc_score(y_test, probs)

        logger.info("Test Accuracy: %.4f | Test ROC-AUC: %.4f", acc, roc)
        logger.info("\n%s", classification_report(y_test, preds, target_names=["Остался", "Ушел"]))

        results[name] = {
            "CV ROC-AUC": f"{mean_cv:.4f}",
            "Test Accuracy": f"{acc:.4f}",
            "Test ROC-AUC": f"{roc:.4f}",
        }

        if roc > best_roc:
            best_roc = roc
            best_model_name = name
            best_preds = preds
            best_probs = probs

    fig1, ax1 = plt.subplots(figsize=(10, 8))
    for name, model in models.items():
        RocCurveDisplay.from_estimator(model, X_test_scaled, y_test, ax=ax1, name=name)
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Случайное угадывание")
    ax1.set_title("ROC-кривые моделей", fontsize=14, fontweight="bold", pad=20)
    ax1.set_xlabel("Доля ложных срабатываний (FPR)", fontsize=12)
    ax1.set_ylabel("Доля истинных срабатываний (TPR)", fontsize=12)
    ax1.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    plt.show()

    if best_preds is not None and best_model_name is not None:
        fig2, ax2 = plt.subplots(figsize=(8, 7))
        cm = confusion_matrix(y_test, best_preds)
        disp = sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax2,
            xticklabels=["Остался", "Ушел"], yticklabels=["Остался", "Ушел"],
        )
        ax2.set_title(f"Матрица ошибок ({best_model_name})", fontsize=14, fontweight="bold", pad=20)
        ax2.set_xlabel("Предсказанный класс", fontsize=12)
        ax2.set_ylabel("Истинный класс", fontsize=12)
        plt.tight_layout()
        plt.show()

    if best_probs is not None and best_model_name is not None:
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        precision, recall, _ = precision_recall_curve(y_test, best_probs)
        pr_auc = roc_auc_score(y_test, best_probs)
        ax3.plot(recall, precision, label=f"{best_model_name} (PR-AUC: {pr_auc:.3f})", color="#2E86AB", linewidth=2.5)
        ax3.set_title("Precision-Recall кривая лучшей модели", fontsize=14, fontweight="bold", pad=20)
        ax3.set_xlabel("Полнота (Recall)", fontsize=12)
        ax3.set_ylabel("Точность (Precision)", fontsize=12)
        ax3.legend(loc="lower left", fontsize=11)
        ax3.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    fig4, ax4 = plt.subplots(figsize=(12, 6))
    ax4.axis("off")
    metrics_df = pd.DataFrame(results).T
    table = ax4.table(
        cellText=metrics_df.values,
        colLabels=metrics_df.columns,
        rowLabels=metrics_df.index,
        cellLoc="center", loc="center",
        colColours=["#2E86AB"] * len(metrics_df.columns),
        rowColours=["#f0f0f0"] * len(metrics_df.index),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.3, 2.0)
    ax4.set_title("Сравнение метрик моделей", fontsize=14, fontweight="bold", pad=30)
    plt.tight_layout()
    plt.show()

    logger.info("\nЛучшая модель: %s с ROC-AUC = %.4f", best_model_name, best_roc)


if __name__ == "__main__":
    logger.info("Начало обучения моделей\n")
    train_and_evaluate()
