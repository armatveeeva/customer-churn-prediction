import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

sns.set_theme(style="darkgrid", palette="viridis", font='DejaVu Sans')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

COLORS = {'success': '#4ECDC4', 'danger': '#FF6B6B', 'neutral': '#95A5A6'}

COLUMN_NAMES_RU = {
    'gender': 'Пол',
    'SeniorCitizen': 'Пенсионер',
    'Partner': 'Партнер',
    'Dependents': 'Иждивенцы',
    'tenure': 'Срок обслуживания',
    'PhoneService': 'Телефония',
    'MultipleLines': 'Несколько линий',
    'InternetService': 'Тип интернета',
    'OnlineSecurity': 'Онлайн-защита',
    'OnlineBackup': 'Онлайн-бекап',
    'DeviceProtection': 'Защита устройств',
    'TechSupport': 'Техподдержка',
    'StreamingTV': 'Стриминг ТВ',
    'StreamingMovies': 'Стриминг кино',
    'Contract': 'Контракт',
    'PaperlessBilling': 'Электронный счет',
    'PaymentMethod': 'Способ оплаты',
    'MonthlyCharges': 'Ежемесячный платеж',
    'TotalCharges': 'Общий платеж',
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
        'Electronic check': 'Электронный чек',
        'Mailed check': 'Чек почтой',
        'Bank transfer (automatic)': 'Банковский перевод',
        'Credit card (automatic)': 'Кредитная карта'
    },
    'Отток': {'Yes': 'Ушел', 'No': 'Остался'}
}


def load_and_clean_data():
    print("Загрузка данных...")
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)

    df.drop('customerID', axis=1, inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    df.rename(columns=COLUMN_NAMES_RU, inplace=True)

    for col, mapping in VALUE_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    return df


def plot_target_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    counts = df['Отток'].value_counts()
    bars = axes[0].bar(['Остался', 'Ушел'], [counts['Остался'], counts['Ушел']],
                       color=[COLORS['success'], COLORS['danger']], edgecolor='black', linewidth=1.2)

    axes[0].set_title('Распределение клиентов по статусу оттока', fontsize=14, fontweight='bold', pad=20)
    axes[0].set_xlabel('Статус клиента', fontsize=12)
    axes[0].set_ylabel('Количество клиентов', fontsize=12)
    axes[0].tick_params(axis='both', which='major', labelsize=11)

    for i, (bar, count) in enumerate(zip(bars, [counts['Остался'], counts['Ушел']])):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2., height + 100,
                     f'{count}\n({count / len(df) * 100:.1f}%)',
                     ha='center', va='bottom', fontsize=12, fontweight='bold')

    wedges, texts, autotexts = axes[1].pie([counts['Остался'], counts['Ушел']],
                                           labels=['Остался', 'Ушел'],
                                           autopct='%1.1f%%',
                                           startangle=90,
                                           colors=[COLORS['success'], COLORS['danger']],
                                           explode=(0.05, 0.05),
                                           shadow=True)

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(13)
        autotext.set_fontweight('bold')

    axes[1].set_title('Доля оттока клиентов', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.show()


def plot_numerical_features(df):
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    num_cols = ['Срок обслуживания', 'Ежемесячный платеж', 'Общий платеж']
    col_titles = ['Срок обслуживания (месяцы)', 'Ежемесячный платеж ($)', 'Общий платеж ($)']

    for i, (col, title) in enumerate(zip(num_cols, col_titles)):
        sns.boxplot(data=df, x='Отток', y=col, ax=axes[i],
                    palette=[COLORS['success'], COLORS['danger']],
                    hue='Отток', legend=False)
        axes[i].set_title(title, fontsize=14, fontweight='bold', pad=15)
        axes[i].set_xlabel('Статус клиента', fontsize=11)
        axes[i].set_ylabel('Значение', fontsize=11)
        axes[i].tick_params(axis='both', which='major', labelsize=10)

    plt.subplots_adjust(top=0.88)
    fig.suptitle('Распределение числовых признаков по статусу оттока',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_categorical_features(df):
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    cat_cols = ['Контракт', 'Тип интернета', 'Способ оплаты']
    col_titles = ['Тип контракта', 'Тип интернет-услуг', 'Способ оплаты']

    for i, (col, title) in enumerate(zip(cat_cols, col_titles)):
        sns.countplot(data=df, x=col, hue='Отток', ax=axes[i],
                      palette=[COLORS['success'], COLORS['danger']])
        axes[i].set_title(title, fontsize=14, fontweight='bold', pad=15)
        axes[i].set_xlabel('', fontsize=11)
        axes[i].set_ylabel('Количество клиентов', fontsize=11)
        axes[i].tick_params(axis='x', rotation=45, labelsize=9)
        axes[i].tick_params(axis='y', labelsize=10)
        axes[i].legend(['Остался', 'Ушел'], loc='upper right', fontsize=10)

    plt.subplots_adjust(top=0.88)
    fig.suptitle('Анализ категориальных признаков',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df):
    df_encoded = pd.get_dummies(df, drop_first=True, dtype=int)

    churn_cols = [col for col in df_encoded.columns if 'Ушел' in col]
    churn_col = churn_cols[0] if churn_cols else None

    if churn_col:
        corr_with_target = df_encoded.corr()[churn_col].drop(churn_col).sort_values(ascending=False)

        plt.figure(figsize=(14, 12))
        top_corr = corr_with_target.head(15).index.tolist() + [churn_col]

        corr_matrix = df_encoded[top_corr].corr()

        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm',
                    fmt='.2f', linewidths=0.5, center=0,
                    square=True, cbar_kws={"shrink": 0.8})

        plt.title('Матрица корреляций: Топ-15 признаков с целевой переменной',
                  fontsize=14, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(fontsize=9)
        plt.tight_layout()
        plt.show()
    else:
        print("Не удалось найти колонку оттока")


if __name__ == "__main__":
    print("Начало EDA-анализа...\n")

    df = load_and_clean_data()

    print("1. Анализ целевой переменной...")
    plot_target_distribution(df)

    print("2. Анализ числовых признаков...")
    plot_numerical_features(df)

    print("3. Анализ категориальных признаков...")
    plot_categorical_features(df)

    print("4. Построение матрицы корреляций...")
    plot_correlation_heatmap(df)

    print("EDA-анализ завершен.")
