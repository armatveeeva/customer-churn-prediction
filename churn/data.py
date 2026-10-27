import pandas as pd
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

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


def load_and_prepare_data() -> pd.DataFrame:
    """
    Загружает датасет IBM Telco Customer Churn и выполняет базовую очистку:
    - Удаляет customerID
    - Приводит TotalCharges к числовому типу и заполняет пропуски медианой
    - Переименовывает колонки на русский язык (COLUMN_NAMES_RU)
    - Применяет маппинг значений (VALUE_MAPS)
    - Создает бинарную целевую переменную 'Отток_num'
    """
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


def create_advanced_features(df: pd.DataFrame, payment_means: pd.Series = None) -> pd.DataFrame:
    """
    Создает продвинутые признаки (Feature Engineering):
    - Группировка срока обслуживания
    - Соотношение ежемесячного и общего платежа
    - Подсчет количества дополнительных услуг
    - Бинарные флаги наличия оптоволокна и стриминга
    - Target Encoding для способа оплаты (опционально, для Streamlit)
    """
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

    if payment_means is not None and 'Способ оплаты' in df.columns:
        df['Способ оплаты_TE'] = df['Способ оплаты'].map(payment_means)
        df['Способ оплаты_TE'] = df['Способ оплаты_TE'].fillna(payment_means.mean())

    return df


def prepare_features_for_model(df: pd.DataFrame, return_payment_means: bool = False):
    """
    Подготавливает DataFrame для обучения модели:
    - Удаляет целевые и промежуточные числовые признаки (_num)
    - Применяет One-Hot Encoding к категориальным признакам
    """
    print("3. Кодирование и подготовка признаков для модели...")

    cols_to_drop = ['Отток', 'Отток_num'] + [c + '_num' for c in
                                             ['Онлайн-защита', 'Онлайн-бекап', 'Защита устройств', 'Техподдержка',
                                              'Стриминг ТВ', 'Стриминг кино']]

    df_model = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    cat_cols = df_model.select_dtypes(exclude='number').columns.tolist()
    df_encoded = pd.get_dummies(df_model, columns=cat_cols, drop_first=True, dtype=int)

    X = df_encoded.drop(columns=['Отток_num'], errors='ignore')
    y = df['Отток_num']

    if return_payment_means:
        payment_means = df.groupby('Способ оплаты')['Отток_num'].mean()
        return X, y, payment_means

    return X, y
