import numpy as np
import pandas as pd
import pytest

N_ROWS = 400


def make_prepared_df(n_rows: int = N_ROWS, seed: int = 0) -> pd.DataFrame:
    """Синтетический датафрейм в том виде, в котором его возвращает load_and_prepare_data."""
    rng = np.random.default_rng(seed)

    yes_no = ["Да", "Нет"]
    service = ["Да", "Нет", "Нет интернета"]
    payment_methods = ["Электронный чек", "Чек почтой", "Банковский перевод", "Кредитная карта"]

    tenure = rng.integers(0, 73, n_rows)
    monthly = rng.uniform(20, 120, n_rows).round(2)
    churn = rng.random(n_rows) < 0.27

    df = pd.DataFrame(
        {
            "Пол": rng.choice(["Мужской", "Женский"], n_rows),
            "Пенсионер": rng.choice(yes_no, n_rows),
            "Партнер": rng.choice(yes_no, n_rows),
            "Иждивенцы": rng.choice(yes_no, n_rows),
            "Срок обслуживания": tenure,
            "Телефония": rng.choice(yes_no, n_rows),
            "Несколько линий": rng.choice(["Да", "Нет", "Нет телефона"], n_rows),
            "Тип интернета": rng.choice(["DSL", "Оптоволокно", "Нет"], n_rows),
            "Онлайн-защита": rng.choice(service, n_rows),
            "Онлайн-бекап": rng.choice(service, n_rows),
            "Защита устройств": rng.choice(service, n_rows),
            "Техподдержка": rng.choice(service, n_rows),
            "Стриминг ТВ": rng.choice(service, n_rows),
            "Стриминг кино": rng.choice(service, n_rows),
            "Контракт": rng.choice(["Помесячный", "1 год", "2 года"], n_rows),
            "Электронный счет": rng.choice(yes_no, n_rows),
            "Способ оплаты": rng.choice(payment_methods, n_rows),
            "Ежемесячный платеж": monthly,
            "Общий платеж": (monthly * tenure).round(2),
            "Отток": np.where(churn, "Ушел", "Остался"),
        }
    )
    df["Отток_num"] = churn.astype(int)
    return df


@pytest.fixture
def prepared_df() -> pd.DataFrame:
    return make_prepared_df()
