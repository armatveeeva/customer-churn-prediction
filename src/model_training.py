import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ssl
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, RocCurveDisplay, confusion_matrix, \
    ConfusionMatrixDisplay, precision_recall_curve

ssl._create_default_https_context = ssl._create_unverified_context

sns.set_theme(style="darkgrid", palette="viridis", font='DejaVu Sans')
plt.rcParams['font.size'] = 11

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


def prepare_features(df):
    print("3. Подготовка признаков для модели...")

    cols_to_drop = ['Отток', 'Отток_num'] + [c + '_num' for c in
                                             ['Онлайн-защита', 'Онлайн-бекап', 'Защита устройств', 'Техподдержка',
                                              'Стриминг ТВ', 'Стриминг кино']]
    df_model = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    cat_cols = df_model.select_dtypes(exclude='number').columns.tolist()
    df_encoded = pd.get_dummies(df_model, columns=cat_cols, drop_first=True, dtype=int)

    X = df_encoded.drop(columns=['Отток_num'] if 'Отток_num' in df_encoded.columns else [])
    y = df['Отток_num']

    return X, y


def train_and_evaluate():
    df = load_and_prepare_data()
    df = create_advanced_features(df)

    X, y = prepare_features(df)

    print("4. Обучение моделей с кросс-валидацией...")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    weight_ratio = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    models = {
        'Логистическая регрессия': LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
        'Случайный лес': RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100),
        'XGBoost': XGBClassifier(scale_pos_weight=weight_ratio, random_state=42, eval_metric='logloss',
                                 n_estimators=100, max_depth=4)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}
    best_model_name = None
    best_roc = 0
    best_preds = None
    best_probs = None

    for name, model in models.items():
        print(f"\n--- {name} ---")

        cv_scores = []
        for train_idx, val_idx in cv.split(X_train_scaled, y_train):
            X_cv_train = X_train_scaled[train_idx]
            X_cv_val = X_train_scaled[val_idx]
            y_cv_train = y_train.iloc[train_idx]
            y_cv_val = y_train.iloc[val_idx]

            model_clone = model.__class__(**model.get_params())
            model_clone.fit(X_cv_train, y_cv_train)
            y_cv_pred = model_clone.predict(X_cv_val)
            score = roc_auc_score(y_cv_val, y_cv_pred)
            cv_scores.append(score)

        mean_cv_score = np.mean(cv_scores)
        std_cv_score = np.std(cv_scores)
        print(f"Кросс-валидация ROC-AUC: {mean_cv_score:.4f} (+/- {std_cv_score:.4f})")

        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, preds)
        roc = roc_auc_score(y_test, probs)

        print(f"Test Accuracy: {acc:.4f} | Test ROC-AUC: {roc:.4f}")
        print(classification_report(y_test, preds, target_names=['Остался', 'Ушел']))

        results[name] = {
            'CV ROC-AUC': f"{mean_cv_score:.4f}",
            'Test Accuracy': f"{acc:.4f}",
            'Test ROC-AUC': f"{roc:.4f}"
        }

        if roc > best_roc:
            best_roc = roc
            best_model_name = name
            best_preds = preds
            best_probs = probs

    # График 1: ROC-кривые всех моделей
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    for name, model in models.items():
        RocCurveDisplay.from_estimator(model, X_test_scaled, y_test, ax=ax1, name=name)
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Случайное угадывание')
    ax1.set_title('ROC-кривые моделей', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Доля ложных срабатываний (FPR)', fontsize=12)
    ax1.set_ylabel('Доля истинных срабатываний (TPR)', fontsize=12)
    ax1.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    plt.show()

    # График 2: Матрица ошибок лучшей модели
    fig2, ax2 = plt.subplots(figsize=(8, 7))
    cm = confusion_matrix(y_test, best_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Остался', 'Ушел'])
    disp.plot(ax=ax2, cmap='Blues', values_format='d')
    ax2.set_title(f'Матрица ошибок ({best_model_name})', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('Предсказанный класс', fontsize=12)
    ax2.set_ylabel('Истинный класс', fontsize=12)
    plt.tight_layout()
    plt.show()

    # График 3: Precision-Recall кривая
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    precision, recall, thresholds = precision_recall_curve(y_test, best_probs)
    pr_auc = roc_auc_score(y_test, best_probs)
    ax3.plot(recall, precision, label=f'{best_model_name} (PR-AUC: {pr_auc:.3f})', color='#2E86AB', linewidth=2.5)
    ax3.set_title('Precision-Recall кривая лучшей модели', fontsize=14, fontweight='bold', pad=20)
    ax3.set_xlabel('Полнота (Recall)', fontsize=12)
    ax3.set_ylabel('Точность (Precision)', fontsize=12)
    ax3.legend(loc='lower left', fontsize=11)
    ax3.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # График 4: Таблица метрик (отдельная фигура)
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    ax4.axis('off')
    metrics_df = pd.DataFrame(results).T
    table = ax4.table(
        cellText=metrics_df.values,
        colLabels=metrics_df.columns,
        rowLabels=metrics_df.index,
        cellLoc='center',
        loc='center',
        colColours=['#2E86AB'] * len(metrics_df.columns),
        rowColours=['#f0f0f0'] * len(metrics_df.index)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.3, 2.0)
    ax4.set_title('Сравнение метрик моделей', fontsize=14, fontweight='bold', pad=30)
    plt.tight_layout()
    plt.show()

    print(f"\nЛучшая модель: {best_model_name} с ROC-AUC = {best_roc:.4f}")


if __name__ == "__main__":
    print("=== Начало обучения моделей ===\n")
    train_and_evaluate()
