import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data import create_advanced_features, load_and_prepare_data


def prepare_features(df: pd.DataFrame):
    print("3. Подготовка признаков для модели...")
    cols_to_drop = ["Отток", "Отток_num"] + [
        c + "_num"
        for c in ["Онлайн-защита", "Онлайн-бекап", "Защита устройств", "Техподдержка", "Стриминг ТВ", "Стриминг кино"]
    ]

    df_model = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")
    cat_cols = df_model.select_dtypes(exclude="number").columns.tolist()
    df_encoded = pd.get_dummies(df_model, columns=cat_cols, drop_first=True, dtype=int)

    X = df_encoded.drop(columns=["Отток_num"] if "Отток_num" in df_encoded.columns else [])
    y = df["Отток_num"]
    return X, y


def train_model(X_train, y_train):
    print("4. Обучение модели XGBoost...")
    weight_ratio = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    model = XGBClassifier(
        scale_pos_weight=weight_ratio, random_state=42, eval_metric="logloss", n_estimators=100, max_depth=4
    )
    model.fit(X_train, y_train)
    return model


def create_shap_explanations(model, X_train, X_test):
    print("5. Вычисление SHAP значений...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    return explainer, shap_values


def plot_summary_plot(shap_values, X_test):
    print("6. Построение Summary Plot (глобальная важность признаков)...")
    plt.figure(figsize=(14, 10))
    shap.summary_plot(shap_values, X_test, show=False, plot_size=(14, 10), color_bar_label="Влияние на прогноз")
    plt.title(
        "Влияние признаков на вероятность оттока клиентов\n(красный = высокое значение, синий = низкое)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("SHAP значение (влияние на вероятность оттока)", fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_force_plot(model, explainer, shap_values, X_test, y_test, sample_idx=0):
    print("7. Построение объяснения для одного клиента...")
    actual_label = "Ушел" if y_test.iloc[sample_idx] == 1 else "Остался"
    predicted_prob = model.predict_proba(X_test.iloc[[sample_idx]])[0][1]

    print(f"\nАнализ клиента #{sample_idx}:")
    print(f"Фактический статус: {actual_label}")
    print(f"Предсказанная вероятность оттока: {predicted_prob:.2%}")

    shap_df = pd.DataFrame(
        {
            "Признак": X_test.columns,
            "SHAP_значение": shap_values[sample_idx],
            "Значение_признака": X_test.iloc[sample_idx].values,
        }
    )
    shap_df["SHAP_абсолютное"] = shap_df["SHAP_значение"].abs()
    shap_df = shap_df.sort_values("SHAP_абсолютное", ascending=False).head(5)

    positive_features = shap_df[shap_df["SHAP_значение"] > 0].sort_values("SHAP_значение", ascending=True)
    negative_features = shap_df[shap_df["SHAP_значение"] < 0].sort_values("SHAP_значение", ascending=False)

    if len(positive_features) > 0:
        plt.figure(figsize=(14, 6))
        plt.subplots_adjust(left=0.35, right=0.95)
        plt.barh(
            range(len(positive_features)),
            positive_features["SHAP_значение"],
            color="#FF6B6B",
            edgecolor="black",
            linewidth=0.5,
            alpha=0.8,
            height=0.6,
        )
        plt.yticks(range(len(positive_features)), positive_features["Признак"], fontsize=11)
        plt.xlabel("SHAP значение (увеличивает риск оттока)", fontsize=12)
        plt.title(
            f"Клиент #{sample_idx}: Факторы, УВЕЛИЧИВАЮЩИЕ риск оттока\nСтатус: {actual_label} | Прогноз: {predicted_prob:.2%}",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        plt.axvline(x=0, color="black", linewidth=1)
        plt.grid(axis="x", alpha=0.3)
        plt.xlim(0, max(positive_features["SHAP_значение"].max() * 1.3, 0.5))
        for i, (_, row) in enumerate(positive_features.iterrows()):
            plt.text(
                row["SHAP_значение"] + 0.05,
                i,
                f"{row['SHAP_значение']:.3f}",
                va="center",
                fontsize=10,
                fontweight="bold",
            )
        plt.tight_layout()
        plt.show()

    if len(negative_features) > 0:
        plt.figure(figsize=(14, 6))
        plt.subplots_adjust(left=0.35, right=0.95)
        plt.barh(
            range(len(negative_features)),
            negative_features["SHAP_значение"],
            color="#4ECDC4",
            edgecolor="black",
            linewidth=0.5,
            alpha=0.8,
            height=0.6,
        )
        plt.yticks(range(len(negative_features)), negative_features["Признак"], fontsize=11)
        plt.xlabel("SHAP значение (уменьшает риск оттока)", fontsize=12)
        plt.title(
            f"Клиент #{sample_idx}: Факторы, УМЕНЬШАЮЩИЕ риск оттока\nСтатус: {actual_label} | Прогноз: {predicted_prob:.2%}",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        plt.axvline(x=0, color="black", linewidth=1)
        plt.grid(axis="x", alpha=0.3)
        plt.xlim(min(negative_features["SHAP_значение"].min() * 1.3, -0.5), 0)
        for i, (_, row) in enumerate(negative_features.iterrows()):
            plt.text(
                row["SHAP_значение"] - 0.05,
                i,
                f"{row['SHAP_значение']:.3f}",
                va="center",
                ha="right",
                fontsize=10,
                fontweight="bold",
            )
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    df = load_and_prepare_data()
    df = create_advanced_features(df)
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)

    model = train_model(X_train_df, y_train)
    explainer, shap_values = create_shap_explanations(model, X_train_df, X_test_df)

    plot_summary_plot(shap_values, X_test_df)
    plot_force_plot(model, explainer, shap_values, X_test_df, y_test, sample_idx=0)
