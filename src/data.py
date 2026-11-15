from pathlib import Path

import pandas as pd

from src.constants import COLUMN_NAMES_RU, DATA_URL, SERVICE_COLS, VALUE_MAPS

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "telco_churn.csv"


def load_raw_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Читает CSV из локальной папки data/. Если файла нет, скачивает его один раз и сохраняет."""
    if not path.exists():
        print(f"Скачивание данных в {path}...")
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(DATA_URL).to_csv(path, index=False)
    return pd.read_csv(path)


def load_and_prepare_data(path: Path = DATA_PATH) -> pd.DataFrame:
    print("1. Загрузка и базовая подготовка данных...")
    df = load_raw_data(path)
    df.drop("customerID", axis=1, inplace=True)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    df.rename(columns=COLUMN_NAMES_RU, inplace=True)
    for col, mapping in VALUE_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    df["Отток_num"] = (df["Отток"] == "Ушел").astype(int)
    return df


def create_advanced_features(
    df: pd.DataFrame,
    payment_means: pd.Series | None = None,
) -> pd.DataFrame:
    print("2. Создание продвинутых признаков...")

    bins = [0, 12, 24, 48, 100]
    labels = ["0-12 мес", "13-24 мес", "25-48 мес", "48+ мес"]
    df["Группа срока"] = pd.cut(df["Срок обслуживания"], bins=bins, labels=labels, right=False)

    df["Соотношение платежей"] = df["Ежемесячный платеж"] / (df["Общий платеж"] + 1e-5)

    for col in SERVICE_COLS:
        df[col + "_num"] = (df[col] == "Да").astype(int)
    df["Количество услуг"] = df[[c + "_num" for c in SERVICE_COLS]].sum(axis=1)

    df["Есть оптоволокно"] = (df["Тип интернета"] == "Оптоволокно").astype(int)
    df["Есть стриминг"] = ((df["Стриминг ТВ"] == "Да") | (df["Стриминг кино"] == "Да")).astype(int)

    if payment_means is None:
        payment_means = df.groupby("Способ оплаты")["Отток_num"].mean()

    df["Способ оплаты_TE"] = df["Способ оплаты"].map(payment_means)
    df["Способ оплаты_TE"] = df["Способ оплаты_TE"].fillna(payment_means.mean())

    return df
