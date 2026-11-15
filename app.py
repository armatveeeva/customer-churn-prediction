import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap
import streamlit as st
from xgboost import XGBClassifier

from src.data import create_advanced_features, load_and_prepare_data

sns.set_theme(style="darkgrid", palette="viridis", font="DejaVu Sans")
plt.rcParams["font.size"] = 10

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.main-header { font-size: 2.5rem; font-weight: 700; color: #2c3e50; margin-bottom: 0.5rem; }
.sub-header { font-size: 1.1rem; color: #7f8c8d; margin-bottom: 1rem; }
.prediction-success { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; padding: 25px; border-radius: 12px; font-size: 1.3rem; font-weight: 600; text-align: center; box-shadow: 0 4px 15px rgba(17, 153, 142, 0.3); margin: 20px 0; }
.prediction-danger { background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%); color: white; padding: 25px; border-radius: 12px; font-size: 1.3rem; font-weight: 600; text-align: center; box-shadow: 0 4px 15px rgba(238, 9, 121, 0.3); margin: 20px 0; }
.sidebar-header { font-size: 1.1rem; font-weight: 600; color: #2c3e50; margin: 20px 0 12px 0; padding-bottom: 8px; border-bottom: 2px solid #3498db; }
.info-box { background: #ecf0f1; padding: 15px; border-radius: 8px; margin-bottom: 15px; }
.stButton>button { background: linear-gradient(135deg, #3498db 0%, #2c3e50 100%); color: white; border: none; padding: 12px 30px; border-radius: 8px; font-weight: 600; font-size: 1rem; transition: all 0.3s ease; width: 100%; }
.stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(52, 152, 219, 0.4); }
.footer { text-align: center; color: #95a5a6; padding: 30px; margin-top: 50px; border-top: 1px solid #ecf0f1; font-size: 0.9rem; }
</style>
"""


@st.cache_data
def get_payment_means():
    df = load_and_prepare_data()
    return df.groupby("Способ оплаты")["Отток_num"].mean()


@st.cache_resource
def get_model_and_explainer():
    model_path = "models/xgb_model.pkl"
    payment_means_path = "models/payment_means.pkl"
    feature_names_path = "models/feature_names.pkl"

    if os.path.exists(model_path) and os.path.exists(payment_means_path):
        model = joblib.load(model_path)
        payment_means = joblib.load(payment_means_path)
        feature_names = joblib.load(feature_names_path)
    else:
        df = load_and_prepare_data()
        payment_means = df.groupby("Способ оплаты")["Отток_num"].mean()
        df = create_advanced_features(df, payment_means)

        cols_to_drop = ["Отток", "Отток_num"] + [
            c + "_num"
            for c in [
                "Онлайн-защита",
                "Онлайн-бекап",
                "Защита устройств",
                "Техподдержка",
                "Стриминг ТВ",
                "Стриминг кино",
            ]
        ]
        df_model = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

        cat_cols = df_model.select_dtypes(exclude="number").columns.tolist()
        df_encoded = pd.get_dummies(df_model, columns=cat_cols, drop_first=True, dtype=int)

        X = df_encoded.drop(columns=["Отток_num"] if "Отток_num" in df_encoded.columns else [])
        y = df["Отток_num"]

        weight_ratio = len(y[y == 0]) / len(y[y == 1])
        model = XGBClassifier(
            scale_pos_weight=weight_ratio,
            random_state=42,
            eval_metric="logloss",
            n_estimators=100,
            max_depth=4,
        )
        model.fit(X, y)

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, model_path)
        joblib.dump(payment_means, payment_means_path)
        joblib.dump(X.columns.tolist(), feature_names_path)
        feature_names = X.columns.tolist()

    explainer = shap.TreeExplainer(model)
    return model, explainer, feature_names, payment_means


st.set_page_config(page_title="Прогнозирование оттока клиентов", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Прогнозирование оттока клиентов</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Интерактивная система машинного обучения с интерпретируемыми результатами</p>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["Прогноз", "Статистика данных", "Модели и метрики"])

with tab1:
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); padding: 20px; border-radius: 12px; border-left: 4px solid #667eea; margin: 20px 0;">
            <h4 style="margin: 0 0 10px 0; color: #2c3e50;">О данных</h4>
            <p style="margin: 0; color: #555; line-height: 1.6;">
                Проект построен на основе открытого датасета <strong>IBM Telco Customer Churn</strong>, содержащего исторические данные о 7 043 клиентах телекоммуникационной компании.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model, explainer, feature_names, payment_means = get_model_and_explainer()

    with st.sidebar:
        st.markdown('<div class="sidebar-header">Личные данные</div>', unsafe_allow_html=True)
        gender = st.selectbox("Пол", ["Мужской", "Женский"])
        senior = st.selectbox("Пенсионер", ["Да", "Нет"])
        partner = st.selectbox("Партнер", ["Да", "Нет"])
        dependents = st.selectbox("Иждивенцы", ["Да", "Нет"])

        st.markdown('<div class="sidebar-header">Финансовые параметры</div>', unsafe_allow_html=True)
        tenure = st.slider("Срок обслуживания (месяцев)", 0, 72, 10)
        monthly_charges = st.slider("Ежемесячный платеж ($)", 18, 120, 50)
        contract = st.selectbox("Тип контракта", ["Помесячный", "1 год", "2 года"])
        payment_method = st.selectbox(
            "Способ оплаты",
            ["Электронный чек", "Чек почтой", "Банковский перевод", "Кредитная карта"],
        )
        paperless = st.selectbox("Электронный счет", ["Да", "Нет"])

        st.markdown('<div class="sidebar-header">Услуги связи</div>', unsafe_allow_html=True)
        internet_service = st.selectbox("Тип интернета", ["DSL", "Оптоволокно", "Нет"])
        phone_service = st.selectbox("Телефония", ["Да", "Нет"])
        multiple_lines = st.selectbox("Несколько линий", ["Да", "Нет", "Нет телефона"])
        online_security = st.selectbox("Онлайн-защита", ["Да", "Нет", "Нет интернета"])
        online_backup = st.selectbox("Онлайн-бекап", ["Да", "Нет", "Нет интернета"])
        device_protection = st.selectbox("Защита устройств", ["Да", "Нет", "Нет интернета"])
        tech_support = st.selectbox("Техподдержка", ["Да", "Нет", "Нет интернета"])
        streaming_tv = st.selectbox("Стриминг ТВ", ["Да", "Нет", "Нет интернета"])
        streaming_movies = st.selectbox("Стриминг кино", ["Да", "Нет", "Нет интернета"])

    col_input, col_metrics = st.columns([1, 3])

    with col_input:
        st.markdown("### Параметры клиента")
        st.markdown(
            f"""
            <div class="info-box">
            <b>Срок обслуживания:</b> {tenure} месяцев<br>
            <b>Ежемесячный платеж:</b> ${monthly_charges}<br>
            <b>Тип контракта:</b> {contract}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Рассчитать прогноз"):
            input_data = {
                "Пол": gender,
                "Пенсионер": senior,
                "Партнер": partner,
                "Иждивенцы": dependents,
                "Срок обслуживания": tenure,
                "Телефония": phone_service,
                "Несколько линий": multiple_lines,
                "Тип интернета": internet_service,
                "Онлайн-защита": online_security,
                "Онлайн-бекап": online_backup,
                "Защита устройств": device_protection,
                "Техподдержка": tech_support,
                "Стриминг ТВ": streaming_tv,
                "Стриминг кино": streaming_movies,
                "Контракт": contract,
                "Электронный счет": paperless,
                "Способ оплаты": payment_method,
                "Ежемесячный платеж": float(monthly_charges),
                "Общий платеж": float(tenure * monthly_charges),
            }

            input_df = pd.DataFrame([input_data])
            input_df = create_advanced_features(input_df, payment_means)

            cols_to_drop = ["Отток", "Отток_num"] + [
                c + "_num"
                for c in [
                    "Онлайн-защита",
                    "Онлайн-бекап",
                    "Защита устройств",
                    "Техподдержка",
                    "Стриминг ТВ",
                    "Стриминг кино",
                ]
            ]
            input_df = input_df.drop(columns=[c for c in cols_to_drop if c in input_df.columns], errors="ignore")

            cat_cols = input_df.select_dtypes(exclude="number").columns.tolist()
            input_encoded = pd.get_dummies(input_df, columns=cat_cols, drop_first=True, dtype=int)

            for col in feature_names:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0

            input_encoded = input_encoded[feature_names]

            prob = model.predict_proba(input_encoded)[0][1]
            prediction = "Уйдет" if prob > 0.5 else "Останется"

            with col_metrics:
                st.markdown("### Результат прогноза")
                if prediction == "Уйдет":
                    st.markdown(
                        f'<div class="prediction-danger">Вероятность оттока: {prob:.1%}<br><span style="font-size: 0.9rem">Клиент с высокой вероятностью уйдет</span></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="prediction-success">Вероятность оттока: {prob:.1%}<br><span style="font-size: 0.9rem">Клиент с высокой вероятностью останется</span></div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("---")
                st.markdown("### Анализ факторов влияния")

                shap_values = explainer.shap_values(input_encoded)
                shap_df = pd.DataFrame({"Признак": feature_names, "SHAP_значение": shap_values[0]})
                shap_df["SHAP_абсолютное"] = shap_df["SHAP_значение"].abs()
                shap_df = shap_df.sort_values("SHAP_абсолютное", ascending=False).head(6)

                positive_features = shap_df[shap_df["SHAP_значение"] > 0].sort_values("SHAP_значение", ascending=True)
                negative_features = shap_df[shap_df["SHAP_значение"] < 0].sort_values("SHAP_значение", ascending=False)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("##### Факторы, увеличивающие риск оттока")
                    if len(positive_features) > 0:
                        fig1, ax1 = plt.subplots(figsize=(6, 4))
                        ax1.barh(
                            range(len(positive_features)),
                            positive_features["SHAP_значение"],
                            color="#FF6B6B",
                            edgecolor="black",
                            linewidth=0.5,
                            alpha=0.8,
                            height=0.6,
                        )
                        ax1.set_yticks(range(len(positive_features)))
                        ax1.set_yticklabels(positive_features["Признак"], fontsize=9)
                        ax1.axvline(x=0, color="black", linewidth=1)
                        ax1.grid(axis="x", alpha=0.3)
                        ax1.set_xlim(0, max(positive_features["SHAP_значение"].max() * 1.3, 0.1))
                        for i, (_, row) in enumerate(positive_features.iterrows()):
                            ax1.text(
                                row["SHAP_значение"] + 0.02,
                                i,
                                f"{row['SHAP_значение']:.3f}",
                                va="center",
                                fontsize=8,
                                fontweight="bold",
                            )
                        plt.tight_layout()
                        st.pyplot(fig1)
                    else:
                        st.info("Нет факторов, увеличивающих риск")

                with col2:
                    st.markdown("##### Факторы, снижающие риск оттока")
                    if len(negative_features) > 0:
                        fig2, ax2 = plt.subplots(figsize=(6, 4))
                        ax2.barh(
                            range(len(negative_features)),
                            negative_features["SHAP_значение"],
                            color="#4ECDC4",
                            edgecolor="black",
                            linewidth=0.5,
                            alpha=0.8,
                            height=0.6,
                        )
                        ax2.set_yticks(range(len(negative_features)))
                        ax2.set_yticklabels(negative_features["Признак"], fontsize=9)
                        ax2.axvline(x=0, color="black", linewidth=1)
                        ax2.grid(axis="x", alpha=0.3)
                        ax2.set_xlim(min(negative_features["SHAP_значение"].min() * 1.3, -0.1), 0)
                        for i, (_, row) in enumerate(negative_features.iterrows()):
                            ax2.text(
                                row["SHAP_значение"] - 0.02,
                                i,
                                f"{row['SHAP_значение']:.3f}",
                                va="center",
                                ha="right",
                                fontsize=8,
                                fontweight="bold",
                            )
                        plt.tight_layout()
                        st.pyplot(fig2)
                    else:
                        st.info("Нет факторов, снижающих риск")

with tab2:
    st.markdown("## Статистика данных")
    df = load_and_prepare_data()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Всего клиентов", value=f"{len(df):,}")
    with col2:
        churn_count = len(df[df["Отток_num"] == 1])
        st.metric(label="Доля оттока", value=f"{churn_count / len(df) * 100:.1f}%")
    with col3:
        st.metric(
            label="Средний срок обслуживания",
            value=f"{df['Срок обслуживания'].mean():.0f} мес",
        )

    st.markdown("---")
    st.markdown("### Распределение целевой переменной")
    fig, ax = plt.subplots(figsize=(8, 6))
    churn_counts = df["Отток"].value_counts()
    colors = ["#4ECDC4" if x == "Остался" else "#FF6B6B" for x in churn_counts.index]
    ax.bar(churn_counts.index, churn_counts.values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_title("Распределение клиентов по статусу оттока", fontsize=14, fontweight="bold")
    for i, v in enumerate(churn_counts.values):
        ax.text(
            i,
            v + 50,
            f"{v} ({v / len(df) * 100:.1f}%)",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )
    plt.tight_layout()
    st.pyplot(fig)

with tab3:
    st.markdown("## Модели и метрики")
    st.markdown("В проекте были обучены и сравнены три модели машинного обучения с обработкой дисбаланса классов.")
    metrics_data = {
        "Модель": ["Логистическая регрессия", "Случайный лес", "XGBoost"],
        "CV ROC-AUC": ["0.7709", "0.7337", "0.7533"],
        "Test Accuracy": ["0.7374", "0.7651", "0.7466"],
        "Test ROC-AUC": ["0.8473", "0.8206", "0.8299"],
        "Recall (Ушел)": ["0.79", "0.64", "0.73"],
    }
    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)

st.markdown(
    '<div class="footer">Проект по машинному обучению</div>',
    unsafe_allow_html=True,
)
