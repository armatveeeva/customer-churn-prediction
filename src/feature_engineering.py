import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

sns.set_theme(style="darkgrid", palette="viridis", font='DejaVu Sans')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

COLORS = {'primary': '#2E86AB', 'secondary': '#A23B72', 'accent': '#F18F01'}

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
    print("2. Создание продвинутых признаков (Feature Engineering)...")

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


def encode_and_select_features(df):
    print("3. Кодирование и отбор признаков...")

    cols_to_exclude = ['Отток', 'Отток_num'] + [c + '_num' for c in
                                                ['Онлайн-защита', 'Онлайн-бекап', 'Защита устройств', 'Техподдержка',
                                                 'Стриминг ТВ', 'Стриминг кино']]

    cat_cols = df.select_dtypes(exclude='number').columns.tolist()
    cat_cols_to_encode = [col for col in cat_cols if col not in cols_to_exclude]

    df_encoded = pd.get_dummies(df, columns=cat_cols_to_encode, drop_first=True, dtype=int)

    df_numeric = df_encoded.select_dtypes(include='number')

    corr_with_target = df_numeric.corr()['Отток_num'].drop('Отток_num').abs().sort_values(ascending=False)

    return corr_with_target.head(15)


def plot_feature_importance(importance_series):
    print("4. Визуализация важности признаков...")

    plt.figure(figsize=(10, 8))

    importance_series = importance_series[::-1]

    new_features_keywords = ['Соотношение', 'Количество', 'Группа', 'Есть оптоволокно', 'Есть стриминг',
                             'Способ оплаты_TE']
    colors = [COLORS['accent'] if any(kw in x for kw in new_features_keywords) else COLORS['primary'] for x in
              importance_series.index]

    bars = plt.barh(importance_series.index, importance_series.values, color=colors, edgecolor='black', linewidth=0.5)

    plt.title('Топ-15 наиболее важных признаков для прогноза оттока\n(синие - исходные, оранжевые - созданные нами)',
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Абсолютная корреляция с целевой переменной', fontsize=12)
    plt.ylabel('Признак', fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=10)

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.005, bar.get_y() + bar.get_height() / 2, f'{width:.3f}',
                 va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    df = load_and_prepare_data()
    df = create_advanced_features(df)
    importance = encode_and_select_features(df)
    plot_feature_importance(importance)

