import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ssl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, RocCurveDisplay

ssl._create_default_https_context = ssl._create_unverified_context

sns.set_theme(style="darkgrid", palette="viridis", font='DejaVu Sans')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

COLUMN_NAMES_RU = {
    'gender': 'Пол', 'SeniorCitizen': 'Пенсионер', 'Partner': 'Партнер',
    'Dependents': 'Иждивенцы', 'tenure': 'Срок обслуживания',
    'PhoneService': 'Телефония', 'MultipleLines': 'Несколько линий',
    'InternetService': 'Тип интернета', 'OnlineSecurity': 'Онлайн-защита',
    'OnlineBackup': 'Онлайн-бекап', 'DeviceProtection': 'Защита устройств',
    'TechSupport': 'Техподдержка', 'StreamingTV': 'Стриминг ТВ',
    'StreamingMovies': 'Стриминг кино', 'Contract': 'Контракт',
    'PaperlessBilling': 'Электронный счет', 'PaymentMethod': 'Способ оплаты',
    'MonthlyCharges': 'Ежемесячный платеж', 'TotalCharges': 'Общий платеж',
    'Churn': 'Отток'
}

VALUE_MAPS = {
    'Пол': {'Male': 'Мужской', 'Female': 'Женский'},
    'Пенсионер': {1: 'Да', 0: 'Нет'},
    'Партнер': {'Yes': 'Да', 'No': 'Нет'},
    'Иждивенцы': {'Yes': 'Да', 'No': 'Нет'},
    'Телефония': {'Yes': 'Да', 'No': 'Нет'},
    'Несколько линий': {'Yes': 'Да', 'No': 'Нет', 'No phone service': 'Нет телефона'},
    'Тип интернета': {'DSL': 'DSL', 'Fiber optic': 'Оптоволокно', 'No': 'Нет'},
    'Онлайн-защита': {'Yes': 'Да', 'No': 'Нет', 'No internet service': 'Нет интернета'},
    'Онлайн-бекап': {'Yes': 'Да', 'No': 'Нет', 'No internet service': 'Нет интернета'},
    'Защита устройств': {'Yes': 'Да', 'No': 'Нет', 'No internet service': 'Нет интернета'},
    'Техподдержка': {'Yes': 'Да', 'No': 'Нет', 'No internet service': 'Нет интернета'},
    'Стриминг ТВ': {'Yes': 'Да', 'No': 'Нет', 'No internet service': 'Нет интернета'},
    'Стриминг кино': {'Yes': 'Да', 'No': 'Нет', 'No internet service': 'Нет интернета'},
    'Контракт': {'Month-to-month': 'Помесячный', 'One year': '1 год', 'Two year': '2 года'},
    'Электронный счет': {'Yes': 'Да', 'No': 'Нет'},
    'Способ оплаты': {
        'Electronic check': 'Электронный чек', 'Mailed check': 'Чек почтой',
        'Bank transfer (automatic)': 'Банковский перевод', 'Credit card (automatic)': 'Кредитная карта'
    },
    'Отток': {'Yes': 'Ушел', 'No': 'Остался'}
}


def load_and_prepare_data():
    print("1. Загрузка и базовая подготовка данных...")
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    df.drop('customerID', axis=1, inplace=True)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    df.rename(columns=COLUMN_NAMES_RU, inplace=True)
    for col, mapping in VALUE_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    df['Отток_num'] = (df['Отток'] == 'Ушел').astype(int)
    return df


def create_advanced_features(df):
    print("2. Применение продвинутых признаков...")

    bins = [0, 12, 24, 48, 100]
    labels = ['0-12 мес', '13-24 мес', '25-48 мес', '48+ мес']
    df['Группа срока'] = pd.cut(df['Срок обслуживания'], bins=bins, labels=labels, right=False)

    df['Соотношение платежей'] = df['Ежемесячный платеж'] / (df['Общий платеж'] + 1e-5)

    service_cols = ['Онлайн-защита', 'Онлайн-бекап', 'Защита устройств', 'Техподдержка', 'Стриминг ТВ', 'Стриминг кино']
    for col in service_cols:
        df[col + '_num'] = (df[col] == 'Да').astype(int)
    df['Количество услуг'] = df[[c + '_num' for c in service_cols]].sum(axis=1)

    df['Есть оптоволокно'] = (df['Тип интернета'] == 'Оптоволокно').astype(int)
    df['Есть стриминг'] = ((df['Стриминг ТВ'] == 'Да') | (df['Стриминг кино'] == 'Да')).astype(int)

    payment_means = df.groupby('Способ оплаты')['Отток_num'].mean()
    df['Способ оплаты_TE'] = df['Способ оплаты'].map(payment_means)

    return df


def train_and_evaluate():
    print("3. Подготовка данных для обучения...")
    df = load_and_prepare_data()
    df = create_advanced_features(df)

    cols_to_drop = ['Отток', 'Отток_num'] + [c + '_num' for c in
                                             ['Онлайн-защита', 'Онлайн-бекап', 'Защита устройств', 'Техподдержка',
                                              'Стриминг ТВ', 'Стриминг кино']]
    df_model = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    cat_cols = df_model.select_dtypes(exclude='number').columns.tolist()
    df_encoded = pd.get_dummies(df_model, columns=cat_cols, drop_first=True, dtype=int)

    X = df_encoded.drop(columns=['Отток_num'] if 'Отток_num' in df_encoded.columns else [])
    y = df['Отток_num']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("4. Обучение моделей...")
    weight_ratio = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    models = {
        'Логистическая регрессия': LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
        'Случайный лес': RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100),
        'XGBoost': XGBClassifier(scale_pos_weight=weight_ratio, random_state=42, eval_metric='logloss',
                                 n_estimators=100, max_depth=4)
    }

    plt.figure(figsize=(10, 8))

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, preds)
        roc = roc_auc_score(y_test, probs)

        print(f"\n--- {name} ---")
        print(f"Accuracy: {acc:.4f} | ROC-AUC: {roc:.4f}")
        print(classification_report(y_test, preds, target_names=['Остался', 'Ушел']))

        RocCurveDisplay.from_estimator(model, X_test_scaled, y_test, ax=plt.gca(), name=name)

    plt.plot([0, 1], [0, 1], 'k--', label='Случайное угадывание', alpha=0.5)
    plt.title('Сравнение моделей: ROC-кривые', fontsize=14, fontweight='bold', pad=20)

    # ПОЛНОСТЬЮ РУССКИЕ ПОДПИСИ ОСЕЙ
    plt.xlabel('Доля ложных срабатываний (FPR)', fontsize=12)
    plt.ylabel('Доля истинных срабатываний (TPR)', fontsize=12)

    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    train_and_evaluate()
